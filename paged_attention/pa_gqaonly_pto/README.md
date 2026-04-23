# Paged attention (GQA decode) — PTO migration workspace

This directory is the **PTO-oriented** copy of `../atb_pa_gqaonly_cce`: same host ABI, tiling (`pa_tiling.py`), tests, and benchmark scripts, with the kernel tree prepared for incremental migration to **PTO-ISA** (`pto/pto-inst.hpp`).

## Build

Requires CANN `bisheng` and `ASCEND_TOOLKIT_HOME`. Optional: **PTO-ISA** tree for headers (defaults to `/workdir/pto-isa-master`).

```bash
export PTO_ISA_ROOT=/path/to/pto-isa-master   # optional
bash ./compile.sh
```

This produces `pa_lib.so` (gitignored) with `call_kernel` / `get_ffts_info` exports.

## Run

```bash
export ASCEND_DEVICE_ID=7    # pick a free NPU
python3 ./test_pa_accuracy.py
python3 ./bench_pa_performance.py --device 7 --warmup 5 --iters 20
```

## PTO port status

Most of the kernel remains AscendC for tiling and MTE paths. PTO-ISA is on the include path via `kernel/pa_gqa_pto_tile_helpers.hpp` on **both** cube and vector translation units. **Cube QK and PV matmul** already go through `pa_pto::tmatmul_fp32acc` (`pto::TMatmul`, with a GEMV-style path when `m==1`). Data movement (`copy_gm_to_cbuf`, L1→L0 loads) and vector softmax still use AscendC intrinsics; see **`PORT_PROGRESS.md`** for the mapping and remaining steps.

## References

- Style / APIs: `pto-kernels/csrc/kernel/kernel_simple_matmul.cpp`, `pto-isa-master/kernels/manual/a2a3/flash_atten/`
- ISA surface: `pto-isa-master/include/pto/npu/a2a3/`, `pto-isa-master/docs/isa`
