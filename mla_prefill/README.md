# MLA Prefill standalone

This directory provides a self-contained `mla_prefill` kernel demo
with lightweight compile + ctypes call style.

## Build and run

```bash
bash compile.sh
python test_mla_prefill.py
```

## Original code reference

- https://gitcode.com/cann/ascend-transformer-boost/blob/br_release_cann_8.5.0_20260527/src/kernels/mixkernels/multi_latent_attention/op_kernel/mla_prefill.cce
- https://gitcode.com/cann/ascend-transformer-boost/blob/master/src/kernels/mixkernels/multi_latent_attention/mla_operation.cpp
- https://gitcode.com/cann/ascend-transformer-boost/blob/br_release_cann_8.5.0_20260527/tests/apitest/kernelstest/mix/test_mla_prefill_rope.py

## Files

- `mla_prefill.cce`: vendored kernel body copy for standalone build
- `mla_prefill_kernel.cce`: local include entry for `mla_prefill.cce`
- `mla_prefill_wrapper.cpp`: lightweight exported `call_kernel(...)` launcher
- `kernels/utils/kernel/*`, `mixkernels/unpad_flash_attention/op_kernel/fa_common.cce`: vendored dependency files
- `compile.sh`: compile script using `bisheng`
- `test_mla_prefill.py`: lightweight smoke runner inspired by the original test data layout
