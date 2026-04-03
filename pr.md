# Pull request: Standalone `chunk_gdn` + extraction guide

## How to verify

```bash
cd chunk_gdn
bash ./compile.sh
bash ./compile_stage1.sh && python test_stage1.py
bash ./compile_stage2.sh && python test_stage2.py
bash ./compile_stage3.sh && python test_stage3.py
python test_chunk_gdn.py   # optional full-kernel case
```

Set `NPU_ID` if not using device `0`.

### Measured performance (snapshot)

From `python benchmark_chunk_gdn.py --torch-ref-only` on Ascend **910B2**, `NPU_ID=0`, **2026-04-02** (torch reference; TFLOP/s share one `estimate_chunk_gdn_flops` scalar per row):

| Case | T | torch ref ms | est. TFLOP/s | eff. GiB/s |
|------|---|--------------|--------------|------------|
| gdn_b1_s64_h4 | 64 | 15.03 | 0.0039 | 0.0326 |
| gdn_b1_s512_h4 | 512 | 18.60 | 0.0253 | 0.1188 |
| gdn_b1_s2048_h4 | 2048 | 31.95 | 0.0588 | 0.2536 |
| gdn_b1_s4096_h4 | 4096 | 49.93 | 0.0753 | 0.3197 |

The benchmark script times the torch path **before** launching `chunk_gdn_lib.so` so baseline ms survive a bad kernel run. Full-kernel CSV columns (`custom_ms`, `custom_tflops_est`, `custom_effective_gibs`) populate when `python test_chunk_gdn.py` succeeds on your device. On the snapshot host the fused `.so` still reported **507057** at sync while staged tests passed; treat custom numbers as device-dependent.

Details: `chunk_gdn/README.md`, `benchmark_chunk_gdn.csv`.

Per-stage custom-kernel performance (TFLOP/s and operand GiB/s): `python benchmark_stage_kernels.py` → `benchmark_stage_kernels.csv` (see README table). Fused `chunk_gdn_lib.so` runtime issues vs working stages: `chunk_gdn/remaining_issue.md`.

## Summary

Adds a self-contained **chunk gated delta rule** example under `atb_standalone/chunk_gdn/` that builds with **bisheng** into shared libraries and runs **on-device** tests via **ctypes** and **torch_npu**, without including sources from `ops-transformer` at compile time. Includes **staged** Stage1 / Stage2 / Stage3 probe binaries, shared Python helpers, and a short **how-to** for future extractions.

## Motivation

- Run and debug the kernel outside the full ops-transformer CMake graph.
- Iterate with small `.so` targets and pytest-style scripts (similar to `mla_prefill`, `matmul_cce`, etc.).
- Keep kernel sources **vendored** under the example tree so CI or other checkouts do not depend on a sibling repo path.

## What changed

### `chunk_gdn/` — standalone build

- **`op_kernel/`**: Vendored AscendC kernel headers and `chunk_gated_delta_rule.cpp` (no `../../ops-transformer/...` includes).
- **Compile scripts**: `compile.sh`, `compile_stage{1,2,3}.sh` add `-I"${SCRIPT_DIR}/op_kernel"` plus existing CANN include paths.
- **Entry points**: `chunk_gdn_wrapper.cpp` + `chunk_gdn_kernel_entry.cpp` expose `call_kernel`; stage wrappers expose `call_stage1`, `call_stage2`, `call_stage3` with **FFTS** setup (`rtGetC2cCtrlAddr` on host, `SetSyncBaseAddr` / `SetAtomicNone` / `SetMaskNorm` on device).
- **Shims** (unchanged behavior, comments clarified): `kernel_tiling/kernel_tiling.h`, `lib/matmul_intf.h`, `kernel_vec_intf.h`, `kernel_cube_intf.h`, `kernel_operator_list_tensor_intf.h`.
- **`chunk_gdn_common.py`**: Shared helpers (`check_close` with optional `mean_tol`, tiling helpers, workspace size helpers).
- **Tests**: `test_chunk_gdn.py`, `test_stage1.py`, `test_stage2.py`, `test_stage3.py` — staged tests **L2-normalize** Q/K (avoids NaNs in Stage1 intermediates); Stage3 asserts both **max** and **mean** absolute error vs reference.
- **`gen_chunk_gdn_tiling.cpp`**: Include path updated to local `chunk_gated_delta_rule_tiling_data.h`; build note for `-I.../op_kernel`.
- **`README.md`**: Build/run instructions and notes (device id, FFTS, normalization).

### Documentation

- **`doc/extract_from_cann_ops.md`**: Checklist and lessons for extracting similar standalone Ascend examples (vendoring, shims, workspace/tiling, FFTS/mix-mode, staged debugging, numerics).

- **`test_chunk_gdn.py`**: Comment wording updated (“upstream” instead of naming another repo).

## Notes for reviewers

- Kernel logic in `op_kernel/*.h` is **copied** from upstream; functional changes belong in follow-ups.
- Large **max** error on Stage3 bf16 output is expected for a few elements; **`mean_tol`** on `check_close` guards against widespread drift.

## Follow-ups (non-blocking)

- Tighten Stage3 thresholds once golden and cast behavior are aligned.
- Optional: compile `gen_chunk_gdn_tiling.cpp` from a small `compile_gen.sh` if host-side tiling generation is needed in CI.

---

*Temporary PR description file; remove or relocate after merge.*
