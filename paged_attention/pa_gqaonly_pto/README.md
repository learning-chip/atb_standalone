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

The numerical implementation still lives in AscendC (`kernel/pa_kernel.cce`) for full tiling and cube/vector split fidelity. PTO-ISA is **on the include path** and included on the **cube** translation unit via `kernel/pa_gqa_pto_tile_helpers.hpp` so new code can call `pto::TLOAD`, `pto::TMATMUL`, etc., alongside existing intrinsics during migration.

See **`PORT_PROGRESS.md`** for intrinsic→PTO mapping, test results, and the staged plan to replace `mad` / `copy_*` / `vadd` with PTO APIs.

## References

- Style / APIs: `pto-kernels/csrc/kernel/kernel_simple_matmul.cpp`, `pto-isa-master/kernels/manual/a2a3/flash_atten/`
- ISA surface: `pto-isa-master/include/pto/npu/a2a3/`, `pto-isa-master/docs/isa`
