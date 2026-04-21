"""
Launch standalone `incre_flash_attention_lib.so` with tiling from `gen_incre_flash_tiling`.

This is the intended end-to-end path (custom AscendC kernel + host-generated tiling),
distinct from `torch_npu.npu_incre_flash_attention` which uses the prebuilt OPP binary.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from typing import Sequence

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _gen_tiling_hex_and_meta(
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_seq_len: int,
    block_size: int,
    block_table_width: int,
    total_k_blocks: int,
    dtype: torch.dtype,
    inner_precise: int = 1,
) -> tuple[bytes, int, int]:
    gen_bin = os.path.join(_HERE, "gen_incre_flash_tiling")
    if not os.path.isfile(gen_bin):
        raise FileNotFoundError(f"Missing {gen_bin}; run ./compile_gen_tiling.sh")

    dt = "bf16" if dtype == torch.bfloat16 else "fp16"
    cmd = [
        gen_bin,
        str(batch),
        str(num_heads),
        str(num_kv_heads),
        str(head_dim),
        str(kv_seq_len),
        str(block_size),
        str(block_table_width),
        str(total_k_blocks),
        dt,
        str(inner_precise),
    ]
    out = subprocess.check_output(cmd, text=True)
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        raise RuntimeError(f"gen_incre_flash_tiling: unexpected output:\n{out}")
    h = lines[0]
    raw = bytes.fromhex(h)
    workspace = int(lines[1])
    block_dim = int(lines[2])
    return raw, workspace, block_dim


def load_incre_flash_lib() -> ctypes.CDLL:
    path = os.path.join(_HERE, "incre_flash_attention_lib.so")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}; run ./compile.sh")
    return ctypes.CDLL(path)


def run_incre_flash_attention_custom(
    q: torch.Tensor,
    k_page: torch.Tensor,
    v_page: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale_value: float,
    block_table: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    block_size: int,
    workspace: torch.Tensor,
    tiling_gm: torch.Tensor,
    block_dim: int,
) -> None:
    """In-place attention into `out`. All tensors must be on the same NPU device."""
    lib = load_incre_flash_lib()
    fn = lib.call_incre_flash_attention
    fn.argtypes = [
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
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    fn.restype = None

    stream = torch.npu.current_stream()
    fn(
        ctypes.c_uint32(int(block_dim)),
        stream._as_parameter_,
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k_page.data_ptr()),
        ctypes.c_void_p(v_page.data_ptr()),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
        ctypes.c_void_p(actual_seq_lengths.data_ptr()),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(0),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(tiling_gm.data_ptr()),
    )


def prepare_paged_ifa(
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_seq_len: int,
    block_size: int,
    dtype: torch.dtype,
    device: str,
    inner_precise: int = 1,
) -> tuple[torch.Tensor, int, int, int]:
    """Build tiling buffer and return (tiling_uint8, workspace_bytes, block_dim, total_k_blocks)."""
    if kv_seq_len % block_size != 0:
        raise ValueError("kv_seq_len must be divisible by block_size")
    nb = kv_seq_len // block_size
    block_table_width = nb
    total_k_blocks = batch * nb
    raw, ws, bd = _gen_tiling_hex_and_meta(
        batch,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        block_size,
        block_table_width,
        total_k_blocks,
        dtype,
        inner_precise=inner_precise,
    )
    tiling = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device=device)
    return tiling, ws, bd, total_k_blocks


def main_quick_check() -> None:
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        print("NPU required", file=sys.stderr)
        sys.exit(1)
    torch.npu.set_device(int(os.environ.get("NPU_ID", "0")))
    b, nh, nkv, d = 1, 32, 8, 128
    kv_seq, bs = 2048, 128
    dtype = torch.float16
    dev = "npu"

    tiling, ws, bd, _ = prepare_paged_ifa(b, nh, nkv, d, kv_seq, bs, dtype, dev)
    q = torch.randn(b, 1, nh * d, dtype=dtype, device=dev)
    nb = kv_seq // bs
    k_page = torch.randn(b * nb, bs, nkv * d, dtype=dtype, device=dev)
    v_page = torch.randn_like(k_page)
    bt = (
        torch.arange(nb, dtype=torch.int32, device=dev)
        .unsqueeze(0)
        .expand(b, -1)
        .clone()
    )
    bt = bt + (torch.arange(b, dtype=torch.int32, device=dev) * nb).unsqueeze(1)
    act = torch.tensor([kv_seq] * b, dtype=torch.int64, device=dev)
    out = torch.empty(b, 1, nh * d, dtype=dtype, device=dev)
    ws_t = torch.empty((ws,), dtype=torch.uint8, device=dev)

    scale = 1.0 / (d**0.5)
    ref = torch_npu_ref(q, k_page, v_page, nh, nkv, scale, bt, act, bs)

    run_incre_flash_attention_custom(
        q,
        k_page,
        v_page,
        out,
        nh,
        nkv,
        scale,
        bt,
        act,
        bs,
        ws_t,
        tiling,
        bd,
    )
    torch.npu.synchronize()
    print("max abs vs torch_npu:", (out - ref).abs().max().item())


def torch_npu_ref(q, k_page, v_page, nh, nkv, scale, bt, act, bs):
    import torch_npu

    return torch_npu.npu_incre_flash_attention(
        q,
        k_page,
        v_page,
        num_heads=nh,
        num_key_value_heads=nkv,
        input_layout="BSH",
        scale_value=scale,
        block_table=bt,
        actual_seq_lengths=act.cpu().tolist(),
        block_size=bs,
    )


if __name__ == "__main__":
    main_quick_check()
