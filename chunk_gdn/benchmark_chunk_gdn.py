"""
Benchmark `chunk_gdn_lib.so` vs the torch reference in `test_chunk_gdn.py`.

Timing: `torch.npu.Event` pairs (same pattern as `mla_prefill/test_mla_prefill.py`).
Per case, the torch reference is timed **before** launching the custom kernel so a baseline
row is still recorded if the kernel faults the device.

Metrics: TFLOP/s from `estimate_chunk_gdn_flops` (same scalar for both columns) and **effective**
GiB/s from user-visible operands, tiling, and outputs only (**excludes** GM `workspaceGM`).
CSV: `benchmark_chunk_gdn.csv`.
"""

from __future__ import annotations

import csv
import ctypes
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import torch_npu

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from test_chunk_gdn import (
    build_tiling_and_workspace,
    cgdr_benchmark_bf16,
    cgdr_golden_native,
)

LIB_PATH = os.path.join(_HERE, "chunk_gdn_lib.so")


def as_ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def benchmark_with_events(fn, warmup_iters: int = 5, benchmark_iters: int = 20) -> float:
    start_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    for _ in range(warmup_iters):
        fn()
    torch.npu.synchronize()
    for i in range(benchmark_iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.npu.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return float(sum(times_ms) / len(times_ms))


def estimate_chunk_gdn_flops(T: int, nk: int, nv: int, dk: int, dv: int, chunk: int) -> float:
    """Scalar work estimate for roofline TFLOP/s (one multiply-add = 2 FLOPs).

    Not a cycle-accurate kernel count: dominant matmul-shaped terms aligned with the chunked
    Q/K/V recurrence (chunk tiles + along-sequence state). Used consistently for custom kernel
    and torch reference rows in CSV.
    """
    C = chunk
    nc = (T + C - 1) // C
    # Chunk-local tiles (Stage1-style QK / inner products): nv × nc chunks × C×C × dk
    qk_blocks = 2.0 * nv * nc * (C * C * dk)
    # Chunk mixing / value paths at O(C²·dv)
    mix_blocks = 2.0 * nv * nc * (C * C * dv)
    # Along-sequence state / projection depth (nk appears in K heads; keep nk in key-like paths)
    state_like = 2.0 * T * nv * (dk * dv + nk * dk * dk + dk * dv)
    return qk_blocks + mix_blocks + state_like


def estimate_effective_io_bytes(
    T: int,
    B: int,
    nk: int,
    nv: int,
    dk: int,
    dv: int,
    tiling_nbytes: int,
) -> int:
    """Bytes for **useful** I/O only: logical inputs read + final outputs written + tiling struct.

    Includes: `query`, `key`, `value`, `beta`, `g`, `initial_state`, `seqlens`, `tilingGM`,
    `out`, `final_state`. **Excludes** the large `workspaceGM` scratch buffer and any other
    internal GM traffic (not part of the operator’s external data interface).
    """
    b2 = 2  # bf16
    f4 = 4  # float32
    i4 = 4  # int32
    read_bytes = (
        T * nk * dk * b2
        + T * nk * dk * b2
        + T * nv * dv * b2
        + T * nv * b2
        + T * nv * f4
        + B * nv * dv * dk * b2
        + B * i4
        + int(tiling_nbytes)
    )
    write_bytes = T * nv * dv * b2 + B * nv * dv * dk * b2
    return read_bytes + write_bytes


def ms_to_tflops_per_s(flops: float, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    return flops / (ms * 1e-3) / 1e12


def ms_to_effective_gibs(effective_bytes: int, ms: float) -> float:
    """GiB/s from effective (user-visible) byte count and kernel time."""
    if ms <= 0:
        return float("nan")
    return effective_bytes / (ms * 1e-3) / (1024.0**3)


def run_benchmarks(*, run_custom_kernel: bool = True) -> None:
    device_id = int(os.environ.get("NPU_ID", "0"))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    try:
        cube_core_num = int(
            torch.npu.get_device_properties(torch.npu.current_device()).cube_core_num
        )
    except Exception:
        cube_core_num = 24
    ai_core_num = max(1, cube_core_num // 3)

    lib = None
    if run_custom_kernel:
        lib = ctypes.CDLL(LIB_PATH)
        lib.call_kernel.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.call_kernel.restype = None
    cases = [
        {"name": "gdn_b1_s64_h4", "B": 1, "seqlen": 64, "nk": 4, "nv": 4, "dk": 128, "dv": 128, "chunk": 64},
        {"name": "gdn_b1_s512_h4", "B": 1, "seqlen": 512, "nk": 4, "nv": 4, "dk": 128, "dv": 128, "chunk": 64},
        {"name": "gdn_b1_s2048_h4", "B": 1, "seqlen": 2048, "nk": 4, "nv": 4, "dk": 128, "dv": 128, "chunk": 64},
        {"name": "gdn_b1_s4096_h4", "B": 1, "seqlen": 4096, "nk": 4, "nv": 4, "dk": 128, "dv": 128, "chunk": 64},
    ]

    error_warn_threshold = 1.0e-2
    results = []
    skipped_cases: list[str] = []

    for i, case in enumerate(cases):
        stream_ptr = torch.npu.current_stream()._as_parameter_
        B = case["B"]
        seqlen = case["seqlen"]
        nk, nv, dk, dv, chunk_size = case["nk"], case["nv"], case["dk"], case["dv"], case["chunk"]
        T = B * seqlen
        scale = 1.0 / math.sqrt(float(dk))

        # Match `test_chunk_gdn.run_one_case`: allocate on-device (same RNG + layout as unit test).
        q = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
        k = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
        v = torch.rand((T, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
        g = (torch.rand((T, nv), dtype=torch.float32, device=device) * -1.0).contiguous()
        beta = torch.rand((T, nv), dtype=torch.bfloat16, device=device).contiguous()
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        initial_state = torch.rand((B, nv, dv, dk), dtype=torch.bfloat16, device=device).contiguous()
        actual_seq_lengths = torch.full((B,), int(seqlen), dtype=torch.int32, device=device)

        flops_est = estimate_chunk_gdn_flops(T, nk, nv, dk, dv, chunk_size)

        has_gamma = 1
        tiling_bytes, tiling_size, workspace_size = build_tiling_and_workspace(
            ai_core_num=ai_core_num,
            B=B,
            T=T,
            nk=nk,
            nv=nv,
            dk=dk,
            dv=dv,
            has_gamma=has_gamma,
            chunk_size=chunk_size,
            scale=scale,
        )

        effective_io_bytes = estimate_effective_io_bytes(T, B, nk, nv, dk, dv, tiling_size)

        # Mix kernel (`KERNEL_TYPE_MIX_AIC_1_2`): launch with one block per AI core.
        block_dim = ai_core_num

        workspace = None
        tilingGM = None
        out = None
        final_state = None

        if run_custom_kernel:
            workspace = torch.empty((workspace_size,), dtype=torch.uint8, device=device)
            workspace.zero_()

            tiling_host_u8 = np.frombuffer(tiling_bytes, dtype=np.uint8).copy()
            assert tiling_size % 8 == 0
            tiling_host_i64 = tiling_host_u8.view(np.int64)
            tilingGM = torch.empty((tiling_size // 8,), dtype=torch.int64, device=device)
            tilingGM.copy_(torch.from_numpy(tiling_host_i64).to(device=device))

            out = torch.empty((T, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
            final_state = torch.empty((B, nv, dv, dk), dtype=torch.bfloat16, device=device).contiguous()

        def run_custom() -> None:
            assert lib is not None and stream_ptr is not None
            assert workspace is not None and tilingGM is not None and out is not None and final_state is not None
            lib.call_kernel(
                block_dim,
                stream_ptr,
                as_ptr(q),
                as_ptr(k),
                as_ptr(v),
                as_ptr(beta),
                as_ptr(initial_state),
                as_ptr(actual_seq_lengths),
                as_ptr(g),
                as_ptr(out),
                as_ptr(final_state),
                as_ptr(workspace),
                as_ptr(tilingGM),
            )

        def run_ref() -> None:
            cgdr_benchmark_bf16(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

        mean_abs_err = float("nan")
        max_abs_err = float("nan")
        max_abs_err_state = float("nan")
        custom_ms = float("nan")
        custom_tflops = float("nan")
        custom_eff_gibs = float("nan")

        # Time torch reference before any custom-kernel launch so we still get baseline ms if the .so faults.
        try:
            ref_ms = benchmark_with_events(run_ref)
        except RuntimeError as e:
            print(f"WARNING[{case['name']}]: skipped (timing ref): {e}")
            skipped_cases.append(case["name"])
            continue

        ref_tflops = ms_to_tflops_per_s(flops_est, ref_ms)
        ref_eff_gibs = ms_to_effective_gibs(effective_io_bytes, ref_ms)

        if run_custom_kernel:
            try:
                run_custom()
                torch.npu.synchronize()
            except RuntimeError as e:
                print(
                    f"ERROR[{case['name']}]: custom kernel failed on smoke run: {e}\n"
                    "Stopping further cases (NPU may be unusable; use a fresh process or "
                    "`python benchmark_chunk_gdn.py --torch-ref-only` for reference-only numbers)."
                )
                skipped_cases.append(case["name"])
                skipped_cases.extend(c["name"] for c in cases[i + 1 :])
                results.append(
                    {
                        "case": case["name"],
                        "B": B,
                        "seqlen": seqlen,
                        "T": T,
                        "nk": nk,
                        "nv": nv,
                        "dk": dk,
                        "dv": dv,
                        "chunk": chunk_size,
                        "block_dim": block_dim,
                        "ai_core_num": ai_core_num,
                        "flops_estimate": flops_est,
                        "effective_io_bytes": effective_io_bytes,
                        "workspace_bytes_excluded_from_bw": workspace_size,
                        "custom_ms": float("nan"),
                        "torch_ref_ms": ref_ms,
                        "custom_tflops_est": float("nan"),
                        "torch_ref_tflops_est": ref_tflops,
                        "custom_effective_gibs": float("nan"),
                        "torch_ref_effective_gibs": ref_eff_gibs,
                        "mean_abs_err": float("nan"),
                        "max_abs_err_out": float("nan"),
                        "max_abs_err_state": float("nan"),
                    }
                )
                print(
                    f"[{case['name']}] torch_ref={ref_ms:.3f} ms ({ref_tflops:.4f} TFLOP/s est, {ref_eff_gibs:.4f} GiB/s eff); "
                    "custom kernel not measured (smoke failed)"
                )
                break

            o_golden, state_golden = cgdr_golden_native(
                q, k, v, g, beta, scale, initial_state, actual_seq_lengths
            )

            mean_abs_err = torch.mean(torch.abs(out.to(torch.float32) - o_golden)).item()
            max_abs_err = torch.max(torch.abs(out.to(torch.float32) - o_golden)).item()
            max_abs_err_state = torch.max(torch.abs(final_state.to(torch.float32) - state_golden)).item()

            if max_abs_err > error_warn_threshold or max_abs_err_state > error_warn_threshold:
                print(
                    f"WARNING[{case['name']}]: skipped (golden mismatch): "
                    f"out max_abs={max_abs_err:.6f}, state max_abs={max_abs_err_state:.6f}, "
                    f"threshold={error_warn_threshold}"
                )
                skipped_cases.append(case["name"])
                continue

            try:
                custom_ms = benchmark_with_events(run_custom)
            except RuntimeError as e:
                print(
                    f"ERROR[{case['name']}]: timing custom kernel failed: {e}\n"
                    "Stopping further cases."
                )
                skipped_cases.append(case["name"])
                skipped_cases.extend(c["name"] for c in cases[i + 1 :])
                results.append(
                    {
                        "case": case["name"],
                        "B": B,
                        "seqlen": seqlen,
                        "T": T,
                        "nk": nk,
                        "nv": nv,
                        "dk": dk,
                        "dv": dv,
                        "chunk": chunk_size,
                        "block_dim": block_dim,
                        "ai_core_num": ai_core_num,
                        "flops_estimate": flops_est,
                        "effective_io_bytes": effective_io_bytes,
                        "workspace_bytes_excluded_from_bw": workspace_size,
                        "custom_ms": float("nan"),
                        "torch_ref_ms": ref_ms,
                        "custom_tflops_est": float("nan"),
                        "torch_ref_tflops_est": ref_tflops,
                        "custom_effective_gibs": float("nan"),
                        "torch_ref_effective_gibs": ref_eff_gibs,
                        "mean_abs_err": mean_abs_err,
                        "max_abs_err_out": max_abs_err,
                        "max_abs_err_state": max_abs_err_state,
                    }
                )
                print(
                    f"[{case['name']}] torch_ref={ref_ms:.3f} ms ({ref_tflops:.4f} TFLOP/s est, {ref_eff_gibs:.4f} GiB/s eff); "
                    "custom kernel not measured (timing failed)"
                )
                break

            custom_tflops = ms_to_tflops_per_s(flops_est, custom_ms)
            custom_eff_gibs = ms_to_effective_gibs(effective_io_bytes, custom_ms)

        if run_custom_kernel:
            print(
                f"[{case['name']}] kernel={custom_ms:.3f} ms ({custom_tflops:.4f} TFLOP/s est, {custom_eff_gibs:.4f} GiB/s eff), "
                f"torch_ref={ref_ms:.3f} ms ({ref_tflops:.4f} TFLOP/s est, {ref_eff_gibs:.4f} GiB/s eff), "
                f"mean_abs_err={mean_abs_err:.6f}, max_abs_err={max_abs_err:.6f}"
            )
        else:
            print(
                f"[{case['name']}] torch_ref={ref_ms:.3f} ms ({ref_tflops:.4f} TFLOP/s est, {ref_eff_gibs:.4f} GiB/s eff) "
                f"(custom kernel not timed)"
            )

        results.append(
            {
                "case": case["name"],
                "B": B,
                "seqlen": seqlen,
                "T": T,
                "nk": nk,
                "nv": nv,
                "dk": dk,
                "dv": dv,
                "chunk": chunk_size,
                "block_dim": block_dim if run_custom_kernel else "",
                "ai_core_num": ai_core_num,
                "flops_estimate": flops_est,
                "effective_io_bytes": effective_io_bytes,
                "workspace_bytes_excluded_from_bw": workspace_size,
                "custom_ms": custom_ms,
                "torch_ref_ms": ref_ms,
                "custom_tflops_est": custom_tflops,
                "torch_ref_tflops_est": ref_tflops,
                "custom_effective_gibs": custom_eff_gibs,
                "torch_ref_effective_gibs": ref_eff_gibs,
                "mean_abs_err": mean_abs_err,
                "max_abs_err_out": max_abs_err,
                "max_abs_err_state": max_abs_err_state,
            }
        )

    def _csv_val(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return ""
        return x

    if results:
        csv_path = os.path.join(_HERE, "benchmark_chunk_gdn.csv")
        rows = [{k: _csv_val(v) for k, v in row.items()} for row in results]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote benchmark csv: {csv_path}")
    else:
        print("WARNING: no successful benchmark cases; csv not written.")
    if skipped_cases:
        print(f"NOTE: skipped cases: {skipped_cases}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Benchmark chunk_gdn kernel vs torch reference.")
    ap.add_argument(
        "--torch-ref-only",
        action="store_true",
        help="Time only the torch reference (no custom kernel .so). Useful if the device is unstable after kernel runs.",
    )
    args = ap.parse_args()
    run_benchmarks(run_custom_kernel=not args.torch_ref_only)
