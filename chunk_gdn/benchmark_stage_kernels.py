"""
Benchmark standalone Stage1 / Stage2 / Stage3 shared libraries (`stage{1,2,3}_lib.so`).

Each stage is timed in isolation (Stage2 and Stage3 restore tensors from snapshots so repeated
launches are valid). Uses `torch.npu.Event` timing like `benchmark_chunk_gdn.py`.

TFLOP/s uses per-stage heuristic FLOP estimates (same definition family as the full-op benchmark).
GiB/s uses **operand footprint**: sum of GM sizes of inputs/outputs/tiling **excluding** the
uint8 `workspace` scratch buffers (implementation detail, not external I/O).
CSV: `benchmark_stage_kernels.csv`.
"""

from __future__ import annotations

import csv
import ctypes
import math
import os
import sys

import torch
import torch.nn.functional as F

import torch_npu

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from chunk_gdn_common import (
    ai_core_num_from_device,
    default_matmul_tiling,
    stage1_workspace_bytes,
    stage3_workspace_bytes,
    tiling_to_device,
    as_ptr,
)
from test_chunk_gdn import ChunkGatedDeltaRuleTilingData

LIB1 = os.path.join(_HERE, "stage1_lib.so")
LIB2 = os.path.join(_HERE, "stage2_lib.so")
LIB3 = os.path.join(_HERE, "stage3_lib.so")


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


def ms_to_tflops_per_s(flops: float, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    return flops / (ms * 1e-3) / 1e12


def ms_to_operand_gibs(operand_bytes: int, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    return operand_bytes / (ms * 1e-3) / (1024.0**3)


def estimate_stage1_flops(T: int, nv: int, dk: int, dv: int, chunk: int) -> float:
    """Heuristic MACs×2 for Stage1 (chunk QK, decay, inner products)."""
    nc = (T + chunk - 1) // chunk
    return 2.0 * nv * nc * (chunk * chunk * dk + chunk * dk * dv + chunk * dv * dk)


def estimate_stage2_flops(T: int, nv: int, dk: int, dv: int) -> float:
    """Heuristic for recurrent state + attn_inter matmul chains along T."""
    return 2.0 * nv * T * (dk * dv * 6 + dk * dk * 2)


def estimate_stage3_flops(T: int, nv: int, dv: int, chunk: int) -> float:
    """Masked qkt @ v style contraction (length T, chunk tile)."""
    return 2.0 * nv * T * chunk * dv * 4


def nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def run_benchmarks() -> None:
    device_id = int(os.environ.get("NPU_ID", "0"))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    ai_core_num = ai_core_num_from_device()

    # Staged tests use nk=nv=1, dk=dv=64, chunk=64; vary sequence length T.
    cases = [
        {"name": "s1_t64", "T": 64},
        {"name": "s1_t512", "T": 512},
        {"name": "s1_t2048", "T": 2048},
        {"name": "s1_t4096", "T": 4096},
    ]

    B, nk, nv, dk, dv, chunk = 1, 1, 1, 64, 64, 64
    matmul_dim = 64

    lib1 = ctypes.CDLL(LIB1)
    lib1.call_stage1.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 13
    lib1.call_stage1.restype = None

    lib2 = ctypes.CDLL(LIB2)
    lib2.call_stage2.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 8
    lib2.call_stage2.restype = None

    lib3 = ctypes.CDLL(LIB3)
    lib3.call_stage3.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 8
    lib3.call_stage3.restype = None

    stream = torch.npu.current_stream()._as_parameter_

    rows: list[dict] = []

    for case in cases:
        T = case["T"]
        name = case["name"]
        scale = 1.0 / math.sqrt(float(dk))

        tiling = ChunkGatedDeltaRuleTilingData()
        tiling.aiCoreNum = ai_core_num
        tiling.t = T
        tiling.nk = nk
        tiling.dk = dk
        tiling.nv = nv
        tiling.dv = dv
        tiling.b = B
        tiling.hasGamma = 0
        tiling.chunkSize = chunk
        tiling.maxGroupLength = T
        tiling.stageOneParaNum = 2
        tiling.scale = float(scale)
        tiling.matmulTilingFp32 = default_matmul_tiling(ai_core_num, matmul_dim)
        tiling_tensor = tiling_to_device(tiling, device)
        tiling_nbytes = tiling_tensor.numel() * tiling_tensor.element_size()

        mask_elems = chunk * chunk * ai_core_num * 2

        query = torch.randn((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
        key = torch.randn((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
        value = torch.randn((T, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
        beta = torch.ones((T, nv), dtype=torch.bfloat16, device=device).contiguous()

        stage_one_mask = torch.zeros((mask_elems,), dtype=torch.float32, device=device).contiguous()
        tri = torch.tril(torch.ones((chunk, chunk), dtype=torch.float32, device=device))
        stage_one_mask[: chunk * chunk].copy_(tri.flatten())
        stage_one_mask[chunk * chunk : 2 * chunk * chunk].copy_(tri.flatten())

        qkt = torch.empty((nv, T, chunk), dtype=torch.float32, device=device).contiguous()
        g_cum_exp = torch.empty((nv, T), dtype=torch.float32, device=device).contiguous()
        k_cum_decay = torch.empty((nv, T, dk), dtype=torch.float32, device=device).contiguous()
        v_inner = torch.empty((nv, T, dv), dtype=torch.float32, device=device).contiguous()
        q_prime = torch.empty((nv, T, dk), dtype=torch.float32, device=device).contiguous()
        kg = torch.empty((nv, T, dk), dtype=torch.float32, device=device).contiguous()

        ws1 = stage1_workspace_bytes(ai_core_num, chunk, dk, dv)
        workspace1 = torch.empty((ws1,), dtype=torch.uint8, device=device)

        op1_bytes = (
            nbytes(query)
            + nbytes(key)
            + nbytes(value)
            + nbytes(beta)
            + nbytes(stage_one_mask)
            + nbytes(qkt)
            + nbytes(g_cum_exp)
            + nbytes(k_cum_decay)
            + nbytes(v_inner)
            + nbytes(q_prime)
            + nbytes(kg)
            + tiling_nbytes
        )
        f1 = estimate_stage1_flops(T, nv, dk, dv, chunk)

        def run_s1() -> None:
            lib1.call_stage1(
                ai_core_num,
                stream,
                as_ptr(query),
                as_ptr(key),
                as_ptr(value),
                as_ptr(beta),
                ctypes.c_void_p(0),
                as_ptr(stage_one_mask),
                as_ptr(qkt),
                as_ptr(g_cum_exp),
                as_ptr(k_cum_decay),
                as_ptr(v_inner),
                as_ptr(q_prime),
                as_ptr(kg),
                as_ptr(workspace1),
                as_ptr(tiling_tensor),
            )

        s1_ms = benchmark_with_events(run_s1)
        rows.append(
            {
                "case": name,
                "stage": "1",
                "T": T,
                "ms": s1_ms,
                "flops_est": f1,
                "tflops_est": ms_to_tflops_per_s(f1, s1_ms),
                "operand_bytes": op1_bytes,
                "operand_gibs": ms_to_operand_gibs(op1_bytes, s1_ms),
                "ai_core_num": ai_core_num,
            }
        )

        # --- Stage2: snapshot after Stage1, restore each timed iteration ---
        torch.npu.synchronize()
        lib1.call_stage1(
            ai_core_num,
            stream,
            as_ptr(query),
            as_ptr(key),
            as_ptr(value),
            as_ptr(beta),
            ctypes.c_void_p(0),
            as_ptr(stage_one_mask),
            as_ptr(qkt),
            as_ptr(g_cum_exp),
            as_ptr(k_cum_decay),
            as_ptr(v_inner),
            as_ptr(q_prime),
            as_ptr(kg),
            as_ptr(workspace1),
            as_ptr(tiling_tensor),
        )
        torch.npu.synchronize()

        snap_qp = q_prime.clone()
        snap_vi = v_inner.clone()
        snap_g = g_cum_exp.clone()
        snap_kcd = k_cum_decay.clone()
        snap_kg = kg.clone()

        cur_state = torch.zeros((nv, dv, dk), dtype=torch.float32, device=device).contiguous()
        attn_inter = torch.zeros((nv, T, dv), dtype=torch.float32, device=device).contiguous()
        workspace2 = torch.zeros((4096,), dtype=torch.uint8, device=device)
        f2 = estimate_stage2_flops(T, nv, dk, dv)
        op2_bytes = (
            nbytes(q_prime)
            + nbytes(v_inner)
            + nbytes(g_cum_exp)
            + nbytes(k_cum_decay)
            + nbytes(cur_state)
            + nbytes(kg)
            + nbytes(attn_inter)
            + tiling_nbytes
        )

        def run_s2() -> None:
            q_prime.copy_(snap_qp)
            v_inner.copy_(snap_vi)
            g_cum_exp.copy_(snap_g)
            k_cum_decay.copy_(snap_kcd)
            kg.copy_(snap_kg)
            cur_state.zero_()
            attn_inter.zero_()
            lib2.call_stage2(
                ai_core_num,
                stream,
                as_ptr(q_prime),
                as_ptr(v_inner),
                as_ptr(g_cum_exp),
                as_ptr(k_cum_decay),
                as_ptr(cur_state),
                as_ptr(kg),
                as_ptr(attn_inter),
                as_ptr(workspace2),
                as_ptr(tiling_tensor),
            )

        s2_ms = benchmark_with_events(run_s2)
        rows.append(
            {
                "case": name,
                "stage": "2",
                "T": T,
                "ms": s2_ms,
                "flops_est": f2,
                "tflops_est": ms_to_tflops_per_s(f2, s2_ms),
                "operand_bytes": op2_bytes,
                "operand_gibs": ms_to_operand_gibs(op2_bytes, s2_ms),
                "ai_core_num": ai_core_num,
            }
        )

        # --- Stage3: run S1+S2 once, snapshot, restore each timed iteration ---
        q_prime.copy_(snap_qp)
        v_inner.copy_(snap_vi)
        g_cum_exp.copy_(snap_g)
        k_cum_decay.copy_(snap_kcd)
        kg.copy_(snap_kg)
        cur_state.zero_()
        attn_inter.zero_()
        lib2.call_stage2(
            ai_core_num,
            stream,
            as_ptr(q_prime),
            as_ptr(v_inner),
            as_ptr(g_cum_exp),
            as_ptr(k_cum_decay),
            as_ptr(cur_state),
            as_ptr(kg),
            as_ptr(attn_inter),
            as_ptr(workspace2),
            as_ptr(tiling_tensor),
        )
        torch.npu.synchronize()

        stage_three_mask = torch.zeros((mask_elems,), dtype=torch.float32, device=device).contiguous()
        stage_three_mask[: chunk * chunk].copy_(tri.flatten())
        stage_three_mask[chunk * chunk : 2 * chunk * chunk].copy_(tri.flatten())

        out_bf16 = torch.empty((T, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
        ws3 = stage3_workspace_bytes(ai_core_num, chunk)
        workspace3 = torch.zeros((ws3,), dtype=torch.uint8, device=device)

        snap_qkt = qkt.clone()
        snap_ge = g_cum_exp.clone()
        snap_ai = attn_inter.clone()
        snap_vi2 = v_inner.clone()

        f3 = estimate_stage3_flops(T, nv, dv, chunk)
        op3_bytes = (
            nbytes(qkt)
            + nbytes(g_cum_exp)
            + nbytes(attn_inter)
            + nbytes(v_inner)
            + nbytes(stage_three_mask)
            + nbytes(out_bf16)
            + tiling_nbytes
        )

        def run_s3() -> None:
            qkt.copy_(snap_qkt)
            g_cum_exp.copy_(snap_ge)
            attn_inter.copy_(snap_ai)
            v_inner.copy_(snap_vi2)
            out_bf16.zero_()
            lib3.call_stage3(
                ai_core_num,
                stream,
                as_ptr(qkt),
                as_ptr(g_cum_exp),
                as_ptr(attn_inter),
                as_ptr(v_inner),
                as_ptr(stage_three_mask),
                as_ptr(out_bf16),
                as_ptr(workspace3),
                as_ptr(tiling_tensor),
            )

        s3_ms = benchmark_with_events(run_s3)
        rows.append(
            {
                "case": name,
                "stage": "3",
                "T": T,
                "ms": s3_ms,
                "flops_est": f3,
                "tflops_est": ms_to_tflops_per_s(f3, s3_ms),
                "operand_bytes": op3_bytes,
                "operand_gibs": ms_to_operand_gibs(op3_bytes, s3_ms),
                "ai_core_num": ai_core_num,
            }
        )

        print(
            f"[{name}] T={T}  stage1={rows[-3]['ms']:.3f} ms ({rows[-3]['tflops_est']:.4f} TFLOP/s est, "
            f"{rows[-3]['operand_gibs']:.4f} GiB/s op), "
            f"stage2={rows[-2]['ms']:.3f} ms ({rows[-2]['tflops_est']:.4f} TFLOP/s est, "
            f"{rows[-2]['operand_gibs']:.4f} GiB/s op), "
            f"stage3={rows[-1]['ms']:.3f} ms ({rows[-1]['tflops_est']:.4f} TFLOP/s est, "
            f"{rows[-1]['operand_gibs']:.4f} GiB/s op)"
        )

    csv_path = os.path.join(_HERE, "benchmark_stage_kernels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    run_benchmarks()
