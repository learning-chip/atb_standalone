# PTO port: `pa_gqaonly_pto` (paged attention GQA decode)

## Goal

Match `atb_pa_gqaonly_cce` behavior and performance while moving implementation toward **PTO-ISA** APIs (`pto::TLOAD`, `pto::TMATMUL`, `pto::TADD`, …), aligned with `pto-kernels/csrc` and `pto-isa-master/kernels/manual/a2a3/flash_atten`.

## Current status (2026-04-23)

| Milestone | State |
|-----------|--------|
| Layout under `pa_gqaonly_pto/` (tiling Python, tests, bench, host wrapper) | Done — same surface as CCE tree |
| `bisheng` shared library `pa_lib.so` | Done — `bash compile.sh` |
| `test_pa_accuracy.py` | **PASS** all fp16 cases on NPU (`ASCEND_DEVICE_ID=0` verified this session) |
| `bench_pa_performance.py` | Runs; IFA comparison lines print as before |
| PTO-ISA headers on cube TU | Done — `-I${PTO_ISA_ROOT}/include` + `kernel/pa_gqa_pto_tile_helpers.hpp` |
| PTO-ISA headers on vector TU | Done — same header included from `pa_kernel.cce` under `__DAV_C220_VEC__` (prep for UB tile ops) |
| Cube QK / PV matmul: `mad` → `pto::TMatmul` | **Done** — both call sites use `pa_pto::tmatmul_fp32acc<IN_DTYPE>` in `pa_gqa_pto_tile_helpers.hpp` (GEMV path when `m==1`) |
| Replace `copy_gm_to_cbuf` / `load_cbuf_to_*` with `pto::TLOAD` / `pto::TMOV` | **Not done** — still AscendC helpers in `pa_kernel.cce` |
| Replace `pa_l0c_to_gm_nd_fp32` (`copy_matrix_cc_to_gm`) with `pto::TSTORE` | **Not done** — needs `GlobalTensor` + `TileAcc` shape wiring to match ND strides |
| Vector softmax / mask (`vadd`, `vexp`, strided `vadd`, `vcmax`, …) → `pto::TADD` / `TEXP` / row ops | **Not done** — UB layout uses repeat/strides; migrate with `TPartAdd` / row tiles or staged loops (see `pto_macro_fa_softmax.hpp`) |

## Implementation notes (cube matmul)

- `pto::TMATMUL` on A2A3 lowers to the same `mad` intrinsic via `pto::TMatmul` (`pto-isa-master/include/pto/npu/a2a3/TMatmul.hpp`). The win is a single PTO-shaped entry point and correct **m==1** handling: generic `TMatmul` with `isGemv=false` widens `m` to 16 on A2A3; this kernel uses **`isGemv=true` when `m==1`** so behavior matches the previous explicit `mad(..., m=1, ...)`.
- Static tile caps in `pa_gqa_pto_tile_helpers.hpp`: `kPaPtoMadMaxM=256`, `kPaPtoMadMaxK=1024`, `kPaPtoMadMaxN=4096`. If future tilings exceed these, **raise the constants** (or refactor to tiling-derived constexprs) or `CheckDynamicMad` / tile asserts will fail at compile or runtime.

## AscendC intrinsic → PTO-ISA mapping (target)

Reference: `pto-isa-master/include/pto/common/pto_instr.hpp`, `pto-kernels/csrc/kernel/kernel_simple_matmul.cpp`, `pto-isa-master/kernels/manual/a2a3/flash_atten/pto_macro_*.hpp`.

| AscendC / kernel_operator (current) | PTO-ISA direction | Notes |
|-------------------------------------|-------------------|--------|
| `copy_gm_to_cbuf` / `pa_gm_to_l1_nd_nd` | `pto::TLOAD` into `Tile<TileType::Mat, …>` or L1 tile | Use `GlobalTensor` + `TileShape`; dynamic sizes use `DYNAMIC` shapes where supported |
| `pa_gm_to_l1_nd_nz` | `pto::TLOAD` with NZ layout parameters | ND→NZ path is performance-critical |
| `load_cbuf_to_ca` / `pa_l1_to_l0_a_vector` | `pto::TMOV` L1→L0A | Match `simple_matmul` MTE1→M pipe flags |
| `load_cbuf_to_cb` / `pa_l1_to_l0_b_vector` | `pto::TMOV` L1→L0B | Transpose variants need `load_cbuf_to_cb_transpose` equivalent plan |
| `mad(...)` (QK and PV) | **`pto::TMatmul` / `TMATMUL` (done for both sites)** | `pa_pto::tmatmul_fp32acc` in `pa_gqa_pto_tile_helpers.hpp` |
| `copy_matrix_cc_to_gm` / `pa_l0c_to_gm_nd_fp32` | `pto::TSTORE` from `TileAcc` | FP32 L0C → GM ND |
| `vadd`, `vadds` | `pto::TADD`, `pto::TADDS` | Vec softmax / rescale in `UnpadAttentionDecoderAiv` |
| `vsub` | `pto::TSUB` | `pto_instr.hpp` |
| `vmul`, `vmuls` | `pto::TMUL`, `pto::TMULS` | |
| `vdiv` | `pto::TDIV` | |
| `vexp` | `pto::TEXP` | Online softmax |
| `vmax`, `vmaxs` | `pto::TMAX`, `pto::TMAXS` | Row max for softmax |
| `pipe_barrier` / `set_flag` / `wait_flag` | Same intrinsics + `pto` event helpers where used in samples | `flash_atten` uses explicit CV FIFO flags |

## Incremental migration plan (suggested order)

1. ~~**Cube QK/PV matmul**~~ — **Done:** `pa_pto::tmatmul_fp32acc` wraps `TMatmul` with correct GEMV branch for `m==1`.
2. **Cube L0C → GM** — `pa_l0c_to_gm_nd_fp32` → `pto::TSTORE` / `TStoreAcc` with runtime `GlobalTensor` shapes matching current `dstStride` / `srcStride`.
3. **Cube data path** — `pa_gm_to_l1_*` / `pa_l1_to_l0_*` → `TLOAD` / `TMOV` where fractal/NZ matches PTO assumptions; keep AscendC for odd transpose paths until parity harness exists.
4. **Vector softmax** — replace contiguous-friendly blocks first; keep `pipe_barrier(PIPE_V)` like `pto_macro_fa_softmax.hpp` for A2A3 vec.
5. **Split-KV / MHA paths** — tiling keys 16/17; keep tiling Python identical (`pa_tiling.py`).

## Repro commands

```bash
export PTO_ISA_ROOT=/workdir/pto-isa-master   # optional; default in compile.sh
cd /workdir/cann_ops_standalone/paged_attention/pa_gqaonly_pto
bash compile.sh
export ASCEND_DEVICE_ID=0   # or 4–7 per cluster policy
python3 test_pa_accuracy.py
python3 bench_pa_performance.py --device 0 --warmup 5 --iters 20
```

## Session log

- Established `pa_gqaonly_pto` from `atb_pa_gqaonly_cce` sources; added PTO include path and cube-side `pa_gqa_pto_tile_helpers.hpp`.
- Verified `test_pa_accuracy.py` all fp16 cases PASS on NPU.
- Verified `bench_pa_performance.py` runs vs IFA baseline.
- **2026-04-23 (this session):** Routed both cube `mad` sites through `pa_pto::tmatmul_fp32acc` (`pto::TMatmul`, GEMV when `m==1`); included `pto/pto-inst.hpp` on the vector translation unit for the next UB migration step; re-ran compile, full fp16 accuracy suite, and bench — all green.
