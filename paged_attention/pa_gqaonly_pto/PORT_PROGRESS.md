# PTO port: `pa_gqaonly_pto` (paged attention GQA decode)

## Goal

Match `atb_pa_gqaonly_cce` behavior and performance while moving implementation toward **PTO-ISA** APIs (`pto::TLOAD`, `pto::TMATMUL`, `pto::TADD`, …), aligned with `pto-kernels/csrc` and `pto-isa-master/kernels/manual/a2a3/flash_atten`.

## Current status (2026-04-23)

| Milestone | State |
|-----------|--------|
| Layout under `pa_gqaonly_pto/` (tiling Python, tests, bench, host wrapper) | Done — same surface as CCE tree |
| `bisheng` shared library `pa_lib.so` | Done — `bash compile.sh` |
| `test_pa_accuracy.py` | **PASS** all fp16 cases on NPU (run with `ASCEND_DEVICE_ID=7` in this environment) |
| `bench_pa_performance.py` | Runs; timings are NPU-dependent but compare correctly vs IFA |
| PTO-ISA headers on cube TU | Done — `-I${PTO_ISA_ROOT}/include` + `kernel/pa_gqa_pto_tile_helpers.hpp` includes `<pto/pto-inst.hpp>` under `__DAV_C220_CUBE__` |
| Replace AscendC `mad` / `copy_*` / `vadd` bodies with `pto::TMATMUL` / `pto::TLOAD` / `pto::TADD` | **Not done** — kernel logic is still the proven AscendC implementation in `kernel/pa_kernel.cce` (~2.9k lines) |

## AscendC intrinsic → PTO-ISA mapping (target)

Reference: `pto-isa-master/include/pto/common/pto_instr.hpp`, `pto-kernels/csrc/kernel/kernel_simple_matmul.cpp`, `pto-isa-master/kernels/manual/a2a3/flash_atten/pto_macro_*.hpp`.

| AscendC / kernel_operator (current) | PTO-ISA direction | Notes |
|-------------------------------------|-------------------|--------|
| `copy_gm_to_cbuf` / `pa_gm_to_l1_nd_nd` | `pto::TLOAD` into `Tile<TileType::Mat, …>` or L1 tile | Use `GlobalTensor` + `TileShape`; dynamic sizes use `DYNAMIC` shapes where supported |
| `pa_gm_to_l1_nd_nz` | `pto::TLOAD` with NZ layout parameters | ND→NZ path is performance-critical |
| `load_cbuf_to_ca` / `pa_l1_to_l0_a_vector` | `pto::TMOV` L1→L0A | Match `simple_matmul` MTE1→M pipe flags |
| `load_cbuf_to_cb` / `pa_l1_to_l0_b_vector` | `pto::TMOV` L1→L0B | Transpose variants need `load_cbuf_to_cb_transpose` equivalent plan |
| `mad(...)` (QK and PV) | `pto::TMATMUL` / `pto::TMATMUL_ACC` | See `TMATMUL` in `kernel_simple_matmul.cpp` |
| `copy_matrix_cc_to_gm` / `pa_l0c_to_gm_nd_fp32` | `pto::TSTORE` from `TileAcc` | FP32 L0C → GM ND |
| `vadd`, `vadds` | `pto::TADD`, `pto::TADDS` | Vec softmax / rescale in `UnpadAttentionDecoderAiv` |
| `vsub` | `pto::TSUB` | `pto_instr.hpp` |
| `vmul`, `vmuls` | `pto::TMUL`, `pto::TMULS` | |
| `vdiv` | `pto::TDIV` | |
| `vexp` | `pto::TEXP` | Online softmax |
| `vmax`, `vmaxs` | `pto::TMAX`, `pto::TMAXS` | Row max for softmax |
| `pipe_barrier` / `set_flag` / `wait_flag` | Same intrinsics + `pto` event helpers where used in samples | `flash_atten` uses explicit CV FIFO flags |

## Incremental migration plan (suggested order)

1. **Cube QK tile** — isolate one `mad` path (single-head, fixed `qk_round_n`) and prove `TMATMUL` + `TLOAD`/`TMOV` parity vs current `s_gm` slice (small unit harness if needed).
2. **Cube PV tile** — same for PV `mad` and `o_tmp_gm` writeback.
3. **Vector softmax** — replace `vadd`/`vexp`/… blocks in `UnpadAttentionDecoderAiv` with UB `Tile<TileType::Vec, …>` and `pto::TADD` / `pto::TEXP` / …; mirror sync from `fa_performance_kernel.cpp`.
4. **Split-KV / MHA paths** — tiling keys 16/17; keep tiling Python identical (`pa_tiling.py`).

## Repro commands

```bash
export PTO_ISA_ROOT=/workdir/pto-isa-master   # optional; default in compile.sh
cd /workdir/cann_ops_standalone/paged_attention/pa_gqaonly_pto
bash compile.sh
export ASCEND_DEVICE_ID=7
python3 test_pa_accuracy.py
python3 bench_pa_performance.py --device 7 --warmup 5 --iters 20
```

## Session log

- Established `pa_gqaonly_pto` from `atb_pa_gqaonly_cce` sources; added PTO include path and cube-side `pa_gqa_pto_tile_helpers.hpp`.
- Verified `test_pa_accuracy.py` all cases PASS on `npu:7`.
- Verified `bench_pa_performance.py` runs vs IFA baseline.
