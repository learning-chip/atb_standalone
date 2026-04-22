# Progress Report: `atb_pa_gqaonly_cce` Raw Intrinsic Port

## Goal (completed)

Create `/workdir/cann_ops_standalone/atb_pa_gqaonly_cce/` — a **raw-intrinsic** version of
`/workdir/cann_ops_standalone/atb_pa_gqaonly/` that:

- Uses **no** `AscendC::` namespace calls in compiled code (comments may reference AscendC for traceability).
- Compiles with `bisheng` targeting `dav-c220`.
- Passes all unit tests in `test_atb_pa_standalone.py` with `max_err ≤ 0.02`.
- Delivers performance in line with the AscendC standalone build (same `pa_lib.so` benchmark script).

Reference style: `/workdir/cann_ops_standalone/mla_prefill_cce/`.

---

## Current Status (2026-04-22)

**Done.** Compilation succeeds. **All six fp16 unit tests pass.** Quick benchmark (`bench_pa_standalone.py --warmup 2 --iters 5`) shows timings within normal run-to-run variance vs `atb_pa_gqaonly`.

### Root cause of prior numerical failures

`ub_to_gm_align` in `kernels/utils/kernel/iterators/gm_to_ub_iterator.inc` incorrectly called
`copy_ubuf_to_gm` with `lenBurst` / gaps intended for the **align** UB→GM path.

AscendC `DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams)` maps to
`DataCopyPadUB2GMImpl` → `copy_ubuf_to_gm_align_b8` / `b16` / `b32` with **block length in bytes**
(see `/workdir/asc-devkit/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h`).

Using the non-align intrinsic misinterpreted transfer sizes and strides, corrupting GM writes
(mask path, stage-2 output, `CopyScale`, etc.) → NaN for GQA and large errors for MHA.

**Fix:** Dispatch `ub_to_gm_align` to `copy_ubuf_to_gm_align_b8` / `b16` / `b32` by `sizeof(DType)`.

### Minor follow-up

- `load_cbuf_to_cb_transpose` calls in `paged_attention_decoder_nd_common.cce` use `(addr_cal_mode_t)0`
  instead of bare `inc` for clarity (equivalent to `__cce_scalar::addr_cal_mode_t::inc`).

---

## Verification commands

```bash
cd /workdir/cann_ops_standalone/atb_pa_gqaonly_cce
bash compile.sh
python test_atb_pa_standalone.py
python bench_pa_standalone.py   # optional; add --warmup / --iters as needed
```

---

## File map (unchanged architecture)

| Area | Path |
|------|------|
| Main kernels | `op_kernel/paged_attention_decoder_nd_common.cce`, `op_kernel/paged_attention_mask_mix.cce` |
| Tensor views / buffer | `kernels/utils/kernel/mem.h` |
| SIMD / MMA / utils / common | `kernels/utils/kernel/simd.h`, `mma.h`, `utils.h`, `common.h`, `hardware.h`, `layout.h` |
| Iterators | `kernels/utils/kernel/iterator.h`, `kernels/utils/kernel/iterators/*.inc` |

---

## Todo list (all complete)

- [x] Copy non-kernel scaffolding from `atb_pa_gqaonly` to `atb_pa_gqaonly_cce`
- [x] Kernel utilities without `AscendC::`
- [x] Iterator `.inc` files with raw intrinsics
- [x] Convert `paged_attention_decoder_nd_common.cce`
- [x] Compile with bisheng
- [x] Fix correctness (`ub_to_gm_align` → align intrinsics)
- [x] All unit tests pass
- [x] Performance sanity vs AscendC build (same bench script; variance expected)

---

## Note for future maintenance

When porting UB↔GM “pad” / “align” AscendC APIs, always match the **exact** underlying intrinsic from
`kernel_operator_data_copy_impl.h` (`copy_*_align_*` vs plain `copy_*`), including **byte** vs
**block** length conventions for each intrinsic family.
