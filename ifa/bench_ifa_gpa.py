#!/usr/bin/env python3
"""
Benchmark `torch_npu.npu_incre_flash_attention` for Grouped-Query Attention (GQA) shapes
typical of the Qwen3 dense series (decode: q_len=1, KV cache length = kv_seq).

Timer matches `cann_ops_standalone/mla_prefill_cce/test_mla_prefill.py` (Event-based mean).

Roofline-style metrics:
  - Achieved TFLOP/s: theory FLOPs / time (QK+PV matmuls).
  - Bandwidth GiB/s: minimum tensor traffic (Q+K+V+O) / time (fp16 by default).

GQA vs MHA/MQA (this script’s counting):
  - **FLOPs** depend on **num_heads (Hq)** only: each query head still does full QK^T and PV
    against its KV sequence using the **shared** KV head for that group — same matmul
    count as MHA with Hq heads. (Using Hkv in the FLOP formula would be wrong for GQA.)
  - **Bytes** use **Hq** for Q and O, and **num_kv_heads (Hkv)** for K and V — the KV cache
    is compressed along heads. Counting K/V with Hq would be **MHA** traffic, not GQA.
  - **MQA** is Hkv=1 (extreme GQA); intensity is highest for fixed Hq when KV is narrowest.

For decode with long `kv_seq`, bytes ≈ 2·B·S_kv·Hkv·D (K+V) plus Q+O; then
`AI = FLOPs/bytes` scales like **Hq/Hkv** (plus a small `Hq` term in the denominator).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


def benchmark_with_events(fn, warmup_iters: int = 5, benchmark_iters: int = 20) -> float:
    """Mean kernel time in ms (same pattern as test_mla_prefill.py)."""
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


def gqa_decode_matmul_flops(
    batch: int, num_query_heads: int, q_seq: int, kv_seq: int, head_dim: int
) -> float:
    """QK + PV matmul FLOPs for GQA incremental attention (same as `test_mla_prefill.estimate_flops_fused_infer`).

    Uses **num_query_heads (Hq)** for both terms: every query head performs a full QK and PV
    matmul along `kv_seq`; KV heads are shared across groups and do **not** divide FLOPs.
    """
    qk = 2.0 * batch * num_query_heads * q_seq * kv_seq * head_dim
    pv = 2.0 * batch * num_query_heads * q_seq * kv_seq * head_dim
    return qk + pv


def gqa_tensor_bytes_bsh(
    batch: int,
    q_seq: int,
    kv_seq: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    elem_size: int,
) -> float:
    """Minimum moved bytes for BSH IFA: Q, K, V in + O out.

    **GQA:** K/V width is `num_kv_heads * head_dim` (not `num_query_heads * head_dim`).
    MHA is the special case Hq == Hkv; MQA is Hkv == 1.
    """
    q_b = batch * q_seq * num_query_heads * head_dim * elem_size
    k_b = batch * kv_seq * num_kv_heads * head_dim * elem_size
    v_b = batch * kv_seq * num_kv_heads * head_dim * elem_size
    o_b = batch * q_seq * num_query_heads * head_dim * elem_size
    return float(q_b + k_b + v_b + o_b)


def theory_intensity_vs_kv_heads(
    batch: int,
    q_seq: int,
    kv_seq: int,
    num_query_heads: int,
    num_kv_heads_list: tuple[int, ...],
    head_dim: int,
    elem_size: int,
) -> list[tuple[int, float, float, float]]:
    """Return (Hkv, flops, bytes, AI) for fixed Hq — AI falls as Hkv grows (same FLOPs, more KV bytes)."""
    rows: list[tuple[int, float, float, float]] = []
    for hkv in num_kv_heads_list:
        f = gqa_decode_matmul_flops(batch, num_query_heads, q_seq, kv_seq, head_dim)
        b = gqa_tensor_bytes_bsh(batch, q_seq, kv_seq, num_query_heads, hkv, head_dim, elem_size)
        rows.append((hkv, f, b, f / b))
    return rows


@dataclass(frozen=True)
class Qwen3GqaCase:
    name: str
    batch: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    kv_seq: int


def default_cases() -> list[Qwen3GqaCase]:
    """Typical Qwen3 GQA decode shapes (q_seq=1); kv_seq sweep for one mid-size config."""
    # From Qwen3 family: 8 KV heads across many sizes; head_dim 128. Multiple (nq, nkv) pairs from public configs.
    return [
        # Qwen3-0.6B / 1.7B style: 16 Q / 8 KV
        Qwen3GqaCase("Qwen3-0.6B_gqa_b1_h16_kv8_d128_kv2048", 1, 16, 8, 128, 2048),
        Qwen3GqaCase("Qwen3-1.7B_gqa_b1_h16_kv8_d128_kv4096", 1, 16, 8, 128, 4096),
        # Qwen3-4B / 8B style: 32 Q / 8 KV
        Qwen3GqaCase("Qwen3-4B_gqa_b1_h32_kv8_d128_kv2048", 1, 32, 8, 128, 2048),
        Qwen3GqaCase("Qwen3-8B_gqa_b1_h32_kv8_d128_kv4096", 1, 32, 8, 128, 4096),
        Qwen3GqaCase("Qwen3-8B_gqa_b1_h32_kv8_d128_kv8192", 1, 32, 8, 128, 8192),
        # Qwen3-14B style: 40 Q / 8 KV
        Qwen3GqaCase("Qwen3-14B_gqa_b1_h40_kv8_d128_kv2048", 1, 40, 8, 128, 2048),
        # Qwen3-32B style: 64 Q / 8 KV
        Qwen3GqaCase("Qwen3-32B_gqa_b1_h64_kv8_d128_kv2048", 1, 64, 8, 128, 2048),
        # Synthetic MHA (Hq=Hkv) for roofline comparison vs GQA — not any Qwen3-8B checkpoint:
        # Qwen/Qwen3-8B is 32 query / 8 KV heads per official config.json.
        Qwen3GqaCase("mha_synth_b1_h32_kv32_d128_kv2048", 1, 32, 32, 128, 2048),
        # Larger batches (same GQA 32/8 — improves achieved GiB/s; theoretical AI unchanged)
        Qwen3GqaCase("Qwen3-8B_gqa_b4_h32_kv8_d128_kv2048", 4, 32, 8, 128, 2048),
        Qwen3GqaCase("Qwen3-8B_gqa_b8_h32_kv8_d128_kv2048", 8, 32, 8, 128, 2048),
        Qwen3GqaCase("Qwen3-8B_gqa_b16_h32_kv8_d128_kv2048", 16, 32, 8, 128, 2048),
        Qwen3GqaCase("Qwen3-8B_gqa_b32_h32_kv8_d128_kv2048", 32, 32, 8, 128, 2048),
        Qwen3GqaCase("Qwen3-8B_gqa_b64_h32_kv8_d128_kv2048", 64, 32, 8, 128, 2048),
    ]


def run_one(
    case: Qwen3GqaCase,
    dtype: torch.dtype,
    warmup_iters: int,
    benchmark_iters: int,
) -> None:
    b = case.batch
    nq = case.num_heads
    nkv = case.num_kv_heads
    d = case.head_dim
    s_kv = case.kv_seq
    q_seq = 1

    scale = 1.0 / math.sqrt(float(d))

    q = torch.randn(b, q_seq, nq * d, dtype=dtype, device="npu")
    k = torch.randn(b, s_kv, nkv * d, dtype=dtype, device="npu")
    v = torch.randn(b, s_kv, nkv * d, dtype=dtype, device="npu")

    def forward():
        return torch_npu.npu_incre_flash_attention(
            q,
            k,
            v,
            num_heads=nq,
            num_key_value_heads=nkv,
            input_layout="BSH",
            scale_value=scale,
        )

    # correctness / alloc
    out = forward()
    torch.npu.synchronize()
    assert out.shape == (b, q_seq, nq * d), f"bad output shape {out.shape}"

    ms = benchmark_with_events(forward, warmup_iters=warmup_iters, benchmark_iters=benchmark_iters)
    t_s = ms * 1e-3

    flops = gqa_decode_matmul_flops(b, nq, q_seq, s_kv, d)
    elem_size = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    nbytes = gqa_tensor_bytes_bsh(b, q_seq, s_kv, nq, nkv, d, elem_size)

    tflops = flops / t_s / 1e12
    gibs = (nbytes / t_s) / (1024**3)
    ai = flops / nbytes  # FLOP / byte (roofline intensity vs this byte model)

    ratio = nq / nkv
    print(
        f"{case.name}: {ms:.4f} ms | {tflops:.4f} TFLOP/s | {gibs:.4f} GiB/s (Q+K+V+O) | "
        f"AI={ai:.4f} F/B | Hq/Hkv={ratio:.1f} | flops={flops:.6e} bytes={nbytes:.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark npu_incre_flash_attention (GQA / Qwen3-style).")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (default float16).")
    args = parser.parse_args()

    if torch_npu is None:
        print("torch_npu is not installed; this benchmark requires Ascend NPU.", file=sys.stderr)
        sys.exit(1)

    if not torch.npu.is_available():
        print("NPU is not available.", file=sys.stderr)
        sys.exit(1)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    torch.npu.set_device("npu:0")

    print(
        "npu_incre_flash_attention GQA benchmark — timer: benchmark_with_events "
        "(ref: mla_prefill_cce/test_mla_prefill.py:274-286)"
    )
    print(f"dtype={dtype} warmup={args.warmup} benchmark_iters={args.iters}")
    elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    print(
        "Theory sanity A (fixed Hq=32, B=1, q=1, kv_seq=2048, d=128): "
        "same FLOPs for all rows; K/V bytes scale with Hkv. "
        "Arithmetic intensity grows with Hq/Hkv (here: 32.0 > 4.0 > 1.0)."
    )
    for hkv, f, b, ai in theory_intensity_vs_kv_heads(
        1, 1, 2048, 32, (1, 8, 32), 128, elem
    ):
        print(f"  Hkv={hkv:2d}  (Hq/Hkv={32/hkv:.1f})  AI={ai:.4f} F/B  flops={f:.6e}  bytes={b:.0f}")
    print(
        "Theory sanity B (fixed Hkv=8, B=1, kv_seq=2048, d=128): "
        "FLOPs and Q/O bytes grow with Hq; KV bytes fixed ⇒ AI still rises with Hq/Hkv."
    )
    for hq in (16, 32, 64):
        f = gqa_decode_matmul_flops(1, hq, 1, 2048, 128)
        b = gqa_tensor_bytes_bsh(1, 1, 2048, hq, 8, 128, elem)
        ai = f / b
        print(f"  Hq={hq:2d}  (Hq/Hkv={hq/8:.1f})  AI={ai:.4f} F/B  flops={f:.6e}  bytes={b:.0f}")
    print("---")
    for c in default_cases():
        try:
            run_one(c, dtype, args.warmup, args.iters)
        except RuntimeError as e:
            print(f"{c.name}: FAILED — {e}")


if __name__ == "__main__":
    main()
