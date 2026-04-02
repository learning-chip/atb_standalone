import ctypes
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import torch_npu


here = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(here, "chunk_gdn_lib.so")


# =========================
# Reference implementation
# =========================

MIN_ERR = 1e-3
CV_MAX_RE = 5
CV_AVER_RE = 1.5
CV_RMSE = 1.5
CV_SMALL_VAL = 2
err_threshold = 2**(-8)


def get_max_re(golden: torch.Tensor, actual: torch.Tensor):
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    return torch.max(abs_error.flatten())


def get_avg_re(golden: torch.Tensor, actual: torch.Tensor):
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    return torch.mean(abs_error)


def get_rmse(golden: torch.Tensor, actual: torch.Tensor):
    sqr_err = torch.pow((actual - golden), 2)
    return torch.sqrt(torch.mean(sqr_err))


def get_smra(golden: torch.Tensor, actual: torch.Tensor):
    abs_A = torch.abs(golden)
    mask_A = abs_A < 2**(-10)
    num_a = torch.sum(mask_A).item()

    abs_B = torch.abs(golden - actual)
    mask_B = abs_B > 1e-16
    num_b = torch.sum(mask_A & mask_B).item()

    return num_b / num_a if num_a > 0 else 0


def get_eb(golden_high_type: torch.Tensor, actual: torch.Tensor):
    golden_nmax = torch.clamp(torch.abs(golden_high_type), min=1)
    actual_error = actual - golden_high_type
    return torch.mean(actual_error / golden_nmax)


def compare_cv(golden: torch.Tensor, golden_high_type: torch.Tensor, actual: torch.Tensor, name=None):
    golden = golden.to(torch.float32)
    golden_high_type = golden_high_type.to(torch.float32)
    actual = actual.to(torch.float32)

    max_re_npu = get_max_re(golden, actual)
    max_re_high_type = get_max_re(golden, golden_high_type)
    avg_re_npu = get_avg_re(golden, actual)
    avg_re_high_type = get_avg_re(golden, golden_high_type)
    rmse_npu = get_rmse(golden, actual)
    rmse_high_type = get_rmse(golden, golden_high_type)
    smra_npu = get_smra(golden, actual)
    smra_high_type = get_smra(golden, golden_high_type)

    max_re_rate = max_re_npu / max(max_re_high_type, err_threshold)
    avg_re_rate = avg_re_npu / max(avg_re_high_type, err_threshold)
    rmse_rate = rmse_npu / max(rmse_high_type, err_threshold)
    smra_rate = smra_npu / max(smra_high_type, err_threshold)

    EB = get_eb(golden_high_type, actual)
    _ = EB  # kept for debugging parity with upstream

    result = (max_re_rate < CV_MAX_RE) and (avg_re_rate < CV_AVER_RE) and (rmse_rate < CV_RMSE)
    result = result and smra_rate < CV_SMALL_VAL

    if name is not None:
        print(
            f"compare[{name}]: "
            f"max_re_rate={float(max_re_rate):.3f} avg_re_rate={float(avg_re_rate):.3f} "
            f"rmse_rate={float(rmse_rate):.3f} smra_rate={float(smra_rate):.3f} "
            f"(golden_high=max_re={float(max_re_high_type):.3e})"
        )

    return bool(result)


def chunk_gated_delta_rule_native(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,  # kept for signature compatibility
):
    initial_dtype = query.dtype
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # reshape to chunks along head dimension
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def chunk_gated_delta_rule_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    cu_seqlens=None,
):
    # Mirrors ops-transformer golden logic, but runs entirely via Torch ops.
    num_heads = q.shape[-2]
    num_value_heads = v.shape[-2]

    if num_value_heads // num_heads > 1:
        q = q.repeat_interleave(num_value_heads // num_heads, dim=2)
        k = k.repeat_interleave(num_value_heads // num_heads, dim=2)

    batch_size = initial_state.shape[0]
    core_attn_out = []
    last_recurrent_state = torch.empty_like(initial_state)

    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        cur_q = q[:, start:end, ...]
        cur_k = k[:, start:end, ...]
        cur_v = v[:, start:end, ...]
        cur_g = g[:, start:end, ...]
        cur_beta = beta[:, start:end, ...]
        cur_state = initial_state[b_idx].unsqueeze(0)

        cur_core_attn_out, cur_last_recurrent_state = chunk_gated_delta_rule_native(
            query=cur_q,
            key=cur_k,
            value=cur_v,
            g=cur_g,
            beta=cur_beta,
            initial_state=cur_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out.append(cur_core_attn_out)
        last_recurrent_state[b_idx] = cur_last_recurrent_state

    tar_dtype = core_attn_out[0].dtype
    tar_device = core_attn_out[0].device
    tar_shape = list(core_attn_out[0].shape)
    tar_shape[1] = cu_seqlens[-1]
    final_cor_attn_out = torch.empty(tar_shape, dtype=tar_dtype, device=tar_device)

    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        final_cor_attn_out[:, start:end, ...] = core_attn_out[b_idx]

    return final_cor_attn_out, last_recurrent_state


def cgdr_golden_native(q, k, v, g, beta, scale, initial_state, actual_seq_lengths, use_float64=False):
    # Golden: compute float32 reference (by forcing query/key/value/beta/g to float32).
    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)

    # Force computation dtype to float32 (matching upstream golden default).
    q_ = q.to(torch.float32)
    k_ = k.to(torch.float32)
    v_ = v.to(torch.float32)
    g_ = g.to(torch.float32)
    beta_ = beta.to(torch.float32)

    o_golden, state_golden = chunk_gated_delta_rule_npu(
        q_.unsqueeze(0),
        k_.unsqueeze(0),
        v_.unsqueeze(0),
        g_.unsqueeze(0),
        beta_.unsqueeze(0),
        scale=scale,
        initial_state=initial_state.transpose(-1, -2).clone().to(v_.dtype),
        cu_seqlens=cu_seqlens,
    )
    o_golden = o_golden[0]
    state_golden = state_golden.transpose(-1, -2)
    return o_golden.to(torch.float32), state_golden.to(torch.float32)


def cgdr_benchmark_bf16(q, k, v, g, beta, scale, initial_state, actual_seq_lengths):
    # High-type: compute using bf16 inputs so the reference output includes bf16 rounding,
    # then convert to float32 for compare_cv.
    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)

    o_bench, state_bench = chunk_gated_delta_rule_npu(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        scale=scale,
        initial_state=initial_state.transpose(-1, -2).clone(),
        cu_seqlens=cu_seqlens,
    )
    o_bench = o_bench[0].to(torch.float32)
    state_bench = state_bench.transpose(-1, -2).to(torch.float32)
    return o_bench, state_bench


# =========================
# Kernel call + tiling data
# =========================


class TCubeTiling(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("usedCoreNum", ctypes.c_int32),
        ("M", ctypes.c_int32),
        ("N", ctypes.c_int32),
        ("Ka", ctypes.c_int32),
        ("Kb", ctypes.c_int32),
        ("singleCoreM", ctypes.c_int32),
        ("singleCoreN", ctypes.c_int32),
        ("singleCoreK", ctypes.c_int32),
        ("baseM", ctypes.c_int32),
        ("baseN", ctypes.c_int32),
        ("baseK", ctypes.c_int32),
        ("depthA1", ctypes.c_int32),
        ("depthB1", ctypes.c_int32),
        ("stepM", ctypes.c_int32),
        ("stepN", ctypes.c_int32),
        ("isBias", ctypes.c_int32),
        ("transLength", ctypes.c_int32),
        ("iterateOrder", ctypes.c_int32),
        ("shareMode", ctypes.c_int32),
        ("shareL1Size", ctypes.c_int32),
        ("shareL0CSize", ctypes.c_int32),
        ("shareUbSize", ctypes.c_int32),
        ("batchM", ctypes.c_int32),
        ("batchN", ctypes.c_int32),
        ("singleBatchM", ctypes.c_int32),
        ("singleBatchN", ctypes.c_int32),
        ("stepKa", ctypes.c_int32),
        ("stepKb", ctypes.c_int32),
        ("depthAL1CacheUB", ctypes.c_int32),
        ("depthBL1CacheUB", ctypes.c_int32),
        ("dbL0A", ctypes.c_int32),
        ("dbL0B", ctypes.c_int32),
        ("dbL0C", ctypes.c_int32),
        ("ALayoutInfoB", ctypes.c_int32),
        ("ALayoutInfoS", ctypes.c_int32),
        ("ALayoutInfoN", ctypes.c_int32),
        ("ALayoutInfoG", ctypes.c_int32),
        ("ALayoutInfoD", ctypes.c_int32),
        ("BLayoutInfoB", ctypes.c_int32),
        ("BLayoutInfoS", ctypes.c_int32),
        ("BLayoutInfoN", ctypes.c_int32),
        ("BLayoutInfoG", ctypes.c_int32),
        ("BLayoutInfoD", ctypes.c_int32),
        ("CLayoutInfoB", ctypes.c_int32),
        ("CLayoutInfoS1", ctypes.c_int32),
        ("CLayoutInfoN", ctypes.c_int32),
        ("CLayoutInfoG", ctypes.c_int32),
        ("CLayoutInfoS2", ctypes.c_int32),
        ("BatchNum", ctypes.c_int32),
        ("mxTypePara", ctypes.c_int32),
    ]


class ChunkGatedDeltaRuleTilingData(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("aiCoreNum", ctypes.c_int64),
        ("t", ctypes.c_int64),
        ("nk", ctypes.c_int64),
        ("dk", ctypes.c_int64),
        ("nv", ctypes.c_int64),
        ("dv", ctypes.c_int64),
        ("b", ctypes.c_int64),
        ("hasGamma", ctypes.c_int64),
        ("chunkSize", ctypes.c_int64),
        ("maxGroupLength", ctypes.c_int64),
        ("interWorkspaceSz", ctypes.c_int64),
        ("stageWorkspaceSz", ctypes.c_int64),
        ("stageOneParaNum", ctypes.c_int64),
        ("scale", ctypes.c_float),
        ("matmulTilingFp32", TCubeTiling),
    ]


def build_tiling_and_workspace(
    *,
    ai_core_num: int,
    B: int,
    T: int,
    nk: int,
    nv: int,
    dk: int,
    dv: int,
    has_gamma: int,
    chunk_size: int,
    scale: float,
):
    # Matches op_host tiling.cpp constants:
    # c=64, p=P_NUM=2, STAGE_ONE_TWO=2, STAGE_ONE_THREE=3, stageOneParaNum=STAGE_ONE_TWO=2
    c = int(chunk_size)
    p = 2
    stage_one_two = 2
    stage_one_three = 3
    mask_num = 4  # MASK_NUM
    stage_one_para_num = stage_one_two

    max_group_len = p * ai_core_num * c
    size_high = 4  # float32

    # interWorkspaceSz
    s = max_group_len
    inter = (
        size_high * nv * s  # gCumExp
        + size_high * nv * s * dk  # kCumDecay
        + size_high * nv * s * dv  # vInner
        + size_high * nv * s * dk  # qPrime
        + size_high * nv * s * dv  # attnInter
        + size_high * nv * s * dk  # kg
        + size_high * nv * s * c  # qkt
        + size_high * B * nv * dv * dk  # highState
        + size_high * c * c * ai_core_num * mask_num  # mask (stageOne+stageThree)
    )

    # stageWorkspaceSz
    stage_ws = size_high * c * (stage_one_two * c + stage_one_three * dk + dv) * stage_one_para_num * ai_core_num

    # tiling struct
    tiling = ChunkGatedDeltaRuleTilingData()
    tiling.aiCoreNum = ai_core_num
    tiling.t = T
    tiling.nk = nk
    tiling.dk = dk
    tiling.nv = nv
    tiling.dv = dv
    tiling.b = B
    tiling.hasGamma = has_gamma
    tiling.chunkSize = c
    tiling.maxGroupLength = max_group_len
    tiling.interWorkspaceSz = inter
    tiling.stageWorkspaceSz = stage_ws
    tiling.stageOneParaNum = stage_one_para_num
    tiling.scale = float(scale)

    # Populate matmul tiling with a "safe-ish" set of values.
    # The kernel's stage MT path uses these fields during Init().
    #
    # Upstream op_host does: mm_.GetTiling(...), then overrides a few fields
    # (dbL0C/stepKa/stepKb/depthA1/depthB1/stepM/stepN). We mirror that and
    # also fill the most commonly used shape/layout fields so we don't leave
    # large portions of TCubeTiling as zeros.
    tiling.matmulTilingFp32 = TCubeTiling()
    mm = tiling.matmulTilingFp32

    # Base shapes for MATMUL_BASE_M/N/K in op_host.
    baseM = 128
    baseN = 128
    baseK = 128
    mm.baseM = baseM
    mm.baseN = baseN
    mm.baseK = baseK

    mm.M = baseM
    mm.N = baseN
    mm.Ka = baseK
    mm.Kb = baseK
    mm.usedCoreNum = ai_core_num

    # Heuristic single-core shapes: split M across cores, keep N/K intact.
    mm.singleCoreM = (baseM + ai_core_num - 1) // ai_core_num
    mm.singleCoreN = baseN
    mm.singleCoreK = baseK

    # Mirrors op_host overrides (after GetTiling()).
    mm.dbL0C = 1
    mm.stepKa = 1
    mm.stepKb = 1
    mm.depthA1 = 1
    mm.depthB1 = 1
    mm.stepM = 1
    mm.stepN = 1

    # Sizes used for some copy/transpose paths (bytes for FP32).
    mm.shareL0CSize = baseM * baseN * 4  # 128*128*4
    mm.transLength = mm.shareL0CSize

    # Keep sharing/cache/simple defaults consistent with op_host.
    mm.shareMode = 0
    mm.shareUbSize = 0
    mm.shareL1Size = 0
    mm.iterateOrder = 0

    # Batch-related fields: matmul is SetDim(1) in op_host.
    mm.batchM = 1
    mm.batchN = 1
    mm.singleBatchM = 1
    mm.singleBatchN = 1

    # Bias disabled.
    mm.isBias = 0

    tiling.matmulTilingFp32 = mm

    tiling_bytes = bytes(tiling)
    tiling_size = len(tiling_bytes)

    # `workspaceGM` is interpreted by AscendC's `GetUserWorkspace(workspaceGM)`,
    # which applies a 16MB "system workspace" offset internally.
    # Allocate that prefix so the device-side pointer math stays in-bounds.
    system_ws = 16 * 1024 * 1024
    workspace_size = system_ws + inter + stage_ws

    return tiling_bytes, tiling_size, workspace_size


def run_one_case(params):
    B, seqlen, nk, nv, dk, dv, chunk_size = params

    T = B * seqlen
    scale = 1.0 / math.sqrt(float(dk))

    q = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
    k = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
    v = torch.rand((T, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
    # g is float32 in golden
    g = (torch.rand((T, nv), dtype=torch.float32, device=device) * -1.0).contiguous()
    beta = torch.rand((T, nv), dtype=torch.bfloat16, device=device).contiguous()
    # Normalize q/k like upstream golden.
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    initial_state = torch.rand((B, nv, dv, dk), dtype=torch.bfloat16, device=device).contiguous()
    actual_seq_lengths = torch.full((B,), int(seqlen), dtype=torch.int32, device=device)

    # ====================
    # Golden (native)
    # ====================
    o_golden, state_golden = cgdr_golden_native(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)
    o_bench, state_bench = cgdr_benchmark_bf16(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

    # ====================
    # Kernel call
    # ====================
    stream_ptr = torch.npu.current_stream()._as_parameter_
    # The kernel implementation uses internal AIC/AIV block-index mapping (divisions by TASK_RATIO/AIC_AIV_1_1).
    # Follow the same launch convention as other standalone Ascend tests:
    # use the computed core count directly.
    block_dim = ai_core_num

    # output tensors
    out = torch.empty((T, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
    final_state = torch.empty((B, nv, dv, dk), dtype=torch.bfloat16, device=device).contiguous()

    # tiling + workspace
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

    workspace = torch.empty((workspace_size,), dtype=torch.uint8, device=device)
    workspace.zero_()

    tiling_host_u8 = np.frombuffer(tiling_bytes, dtype=np.uint8).copy()
    # `tilingGM` is read by device code as int64/float via fixed offsets.
    # Allocate as int64 to guarantee 8-byte alignment of the base address.
    assert tiling_size % 8 == 0
    tiling_host_i64 = tiling_host_u8.view(np.int64)
    tilingGM = torch.empty((tiling_size // 8,), dtype=torch.int64, device=device)
    tilingGM.copy_(torch.from_numpy(tiling_host_i64).to(device=device))

    def as_ptr(t: torch.Tensor):
        return ctypes.c_void_p(t.data_ptr())

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
    torch.npu.synchronize()

    # ====================
    # Compare
    # ====================
    ok_out = compare_cv(o_golden, o_bench, out, name="o")
    ok_state = compare_cv(state_golden, state_bench, final_state, name="state")

    if not ok_out or not ok_state:
        # Give helpful diagnostic info.
        max_diff_o = torch.max(torch.abs(out.to(torch.float32) - o_golden)).item()
        max_diff_s = torch.max(torch.abs(final_state.to(torch.float32) - state_golden)).item()
        raise AssertionError(f"chunk_gdn failed: max_diff_o={max_diff_o}, max_diff_s={max_diff_s}")

    print(f"PASS: B={B} seqlen={seqlen} nk={nk} nv={nv} dk={dk} dv={dv} chunk={chunk_size}")


if __name__ == "__main__":
    # Device selection: override with `NPU_ID=...` if needed.
    # Example: torch.npu.set_device("npu:7") if NPU 7 is free.
    device_id = int(os.environ.get("NPU_ID", "0"))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    try:
        cube_core_num = int(torch.npu.get_device_properties("npu").cube_core_num)
    except Exception:
        cube_core_num = 24
    ai_core_num = cube_core_num // 3

    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # query
        ctypes.c_void_p,  # key
        ctypes.c_void_p,  # value
        ctypes.c_void_p,  # beta
        ctypes.c_void_p,  # initialState
        ctypes.c_void_p,  # seqlens
        ctypes.c_void_p,  # gOptional (gamma)
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # finalState
        ctypes.c_void_p,  # workspaceGM
        ctypes.c_void_p,  # tilingGM
    ]
    lib.call_kernel.restype = None

    # Mirrors ops-transformer pytest paramset (but we only run correctness, not performance).
    test_params = [
        (1, 64, 4, 4, 128, 128, 64),
        (1, 16384, 4, 4, 128, 128, 64),
    ]

    for params in test_params:
        run_one_case(params)

    print("chunk_gdn all tests passed.")

