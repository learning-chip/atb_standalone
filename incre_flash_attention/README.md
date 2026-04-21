# Standalone `incre_flash_attention` (AscendC)

This directory vendors the AscendC sources for **incremental flash attention** (same family as CANN OPP `ops_transformer/ascendc/incre_flash_attention`) and builds a shared library with `bisheng`, similar to `cann_ops_standalone/chunk_gdn` and `mla_prefill`.

## Layout

| Path | Role |
|------|------|
| `op_kernel/` | Kernel sources (from CANN `opp/built-in/.../incre_flash_attention/`; aligned with `ops-transformer/attention/incre_flash_attention/op_kernel` for arch32) |
| `kernel_tiling/kernel_tiling.h` | Device-side tiling structs (`IncreFlashAttentionTilingDataV2`, …) extending `AscendC::tiling` from CANN `highlevel_api/kernel_tiling/kernel_tiling.h` |
| `kernels/utils/kernel/` | Iterator helpers (from `mla_prefill`); local renames avoid clashing with `AscendC::DataFormat` |
| `lib/matmul_intf.h`, `lib/matrix/matmul/tiling.h` | Include shims to CANN `adv_api/matmul/*` |
| `incre_flash_attention_wrapper.cpp` | Host wrapper exporting `call_incre_flash_attention` |
| `ifa_torch.py` | Python API: calls `torch_npu.npu_incre_flash_attention` (production path; framework supplies tiling) |
| `ifa_tiling_common.py` | ctypes mirrors of `IncreFlashAttentionTilingDataV2` (and nested types) + `tiling_to_device`; layout matches `kernel_tiling/kernel_tiling.h` / `op_host/incre_flash_attention_tiling.h` |
| `compile.sh` | Builds `incre_flash_attention_lib.so` |

### Tiling buffer (custom `.so` launch)

`kernel_tiling/kernel_tiling.h` defines **`GET_TILING_DATA_MEMBER`** and **`GET_TILING_DATA_WITH_STRUCT`** as explicit byte copies from the GM tiling pointer into local structs (same self-contained pattern as `cann_ops_standalone/chunk_gdn/kernel_tiling/kernel_tiling.h`, generalized with the null-pointer offset idiom for nested members such as `tilingBase.bmm1TilingData`).

Python can build a device `uint8` tensor with `ifa_tiling_common.tiling_to_device` once you have a filled `IncreFlashAttentionTilingDataV2` (values must still match what the CANN op host would emit for your shape; porting `incre_flash_attention_tiling.cpp` / context is out of scope here).

## Build (bisheng)

Requires `ASCEND_TOOLKIT_HOME` (CANN toolkit root, e.g. `/usr/local/Ascend/cann-8.5.1`).

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.1
cd /workdir/cann_ops_standalone/incre_flash_attention
./compile.sh
```

Output: `incre_flash_attention_lib.so`.

**Note:** The OPP kernel relies on dynamic tiling keys (`TILING_KEY_IS` / `REGISTER_TILING_DEFAULT`); launching `call_incre_flash_attention` from Python still requires a valid tiling buffer (normally produced by the CANN op host inside `torch_npu`). For benchmarks and tests, use `ifa_torch.py` / `torch_npu` as below. For ctypes layout and GM packing, see `ifa_tiling_common.py` and the tiling macros at the bottom of `kernel_tiling/kernel_tiling.h`.

## Run tests (NPU)

Pick a free device (`npu-smi info`), then:

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.1
cd /workdir/cann_ops_standalone/incre_flash_attention
./compile.sh
python3 test_incre_flash_attention.py
```

## Paged GQA benchmark (parity with `ifa/bench_ifa_gpa_paged.py`)

```bash
cd /workdir/cann_ops_standalone/incre_flash_attention
python3 benchmark_ifa_paged.py --device 2 --warmup 5 --iters 20
```

This uses the same timer and shapes as `cann_ops_standalone/ifa/bench_ifa_gpa_paged.py` (via `ifa_torch` → `torch_npu.npu_incre_flash_attention`), so throughput numbers are directly comparable.

## Local patches (standalone)

The following edits are applied so the tree builds **outside** the full `ops-transformer` / OPP CMake:

- `op_kernel/incre_flash_attention.cpp`: `incre_flash_attention` calls `incre_flash_attention_FIAS_OBP` directly (avoids calling one `__global__` kernel from another under bisheng).
- `op_kernel/incre_flash_attention_obp.h`: always include arch32 paths (OPP tree here has no `arch20/` fallback).
- `op_kernel/arch32/paged_attention_antiquantkv.h`: drop duplicate iterator includes (already pulled from `iterator.h`).
- `op_kernel/arch32/incre_flash_attention_preload_dd.h`: `EVENT_ID4`–`EVENT_ID7` constants use numeric values where identifiers are not in scope early in the include graph.
- `kernels/utils/kernel/layout.h` / `iterator.h`: `IfaKernelDataFormat` alias to avoid `AscendC::DataFormat` ambiguity.
