# Remaining issue: end-to-end `chunk_gdn_lib.so` vs working staged kernels

## What works

- **`stage1_lib.so`**, **`stage2_lib.so`**, **`stage3_lib.so`**: each is launched from Python via ctypes, uses **`rtGetC2cCtrlAddr` + `SetSyncBaseAddr`** (FFTS) and **`KERNEL_TYPE_MIX_AIC_1_2`** where applicable. On-device tests **`test_stage1.py`**, **`test_stage2.py`**, **`test_stage3.py`** complete successfully with **custom AscendC kernels** (not PyTorch-only references).
- **`benchmark_stage_kernels.py`**: times each stage in isolation and reports TFLOP/s (heuristic) and operand GiB/s; see **`benchmark_stage_kernels.csv`**.

## What fails

- **`chunk_gdn_lib.so`** (fused `chunk_gated_delta_rule` in **`chunk_gdn_wrapper.cpp`** / **`chunk_gdn_kernel_entry.cpp`**) + **`python test_chunk_gdn.py`** and **`python benchmark_chunk_gdn.py`** (custom path): runtime failure **`507057`** (often surfaced at **`torch.npu.synchronize()`** after launch; ACL may report **`aclrtLaunchKernelWithHostArgs failed`** / **kernel launch from cache failed** depending on timing and driver).

So: **per-stage custom kernels are fine; the combined (e2e) fused entry is not stable on the same device/CANN stack** used for verification.

## Symptom details (for plog / vendor triage)

- Error class: **SUSPECT REMOTE ERROR**, code **507057** (`0x7B9C1` / variants in logs).
- Staged and fused builds use the same **`--npu-arch=dav-2201`** bisheng flags in **`compile*.sh`**.
- Fused kernel entry was aligned with stages (FFTS base, **`KERNEL_TYPE_MIX_AIC_1_2`**, **`SetAtomicNone` / `SetMaskNorm`**), and host tiling **`build_tiling_and_workspace`** was adjusted so matmul **`singleCoreM/N/K`** match **`default_matmul_tiling`** (splitting M across AI cores broke cube **`Init`**). **Despite that, e2e still faults** on at least one **910B2** runner.

## Likely directions (hypotheses for future work)

1. **Resource / code size**: The fused object links **Stage1 + Stage2 + Stage3 + full `CGDR` pipeline** in one binary. It may exceed limits for **kernel cache**, **launch args**, or **on-chip** resources where small stage `.so` files succeed. *Mitigation to try:* split fusion boundaries, reduce live ranges, or match upstream’s exact **blockDim** / **task type** split.
2. **Workspace / tiling parity with upstream op_host**: Standalone **`build_tiling_and_workspace`** mirrors comments from **`op_host`** but may still diverge in **`interWorkspaceSz`**, **`stageWorkspaceSz`**, or **mask** layout vs what the fused **`Init`** expects for **large `T`** and **`maxGroupLength` &lt; `T`**. *Mitigation:* diff against real **`GetTiling`** output from CANN for the same shapes.
3. **FFTS / cross-core lifecycle**: Stages set FFTS once per process; fused kernel sets it inside the entry. If the runtime expects a different **order** or **per-context** setup, mix-mode sync could fault. *Mitigation:* compare with upstream full-op launch sequence (host runtime).
4. **CANN / driver version**: **507057** is generic; collect **`plog`** as ACL suggests and test another **CANN** drop on the same hardware.

## Concrete next steps

1. Reproduce with **`ASCEND_LAUNCH_BLOCKING=1`** and **`ASCEND_GLOBAL_LOG_LEVEL=1`** (or your site’s ACL debug flags); capture **device plog** around the fault.
2. Confirm **`python test_chunk_gdn.py`** runs correctly; if only fused fails, bisect by **temporarily** linking a fused entry that calls **only `RunStage1`** then **`RunStage2`** then **`RunStage3`** in sequence (same as Python chain) to see whether **fusion in one kernel** vs **tiling** is at fault.
3. Compare **`.o` size** and **kernel metadata** between **`chunk_gdn_lib.so`** and **`stage1_lib.so`** (e.g. `nm`, vendor tools).
4. When e2e passes, re-run **`benchmark_chunk_gdn.py`** without **`--torch-ref-only`** and refresh **`README.md`** / **`pr.md`** custom-kernel rows.

## Files to read first

| Area | File |
|------|------|
| Fused entry | `chunk_gdn_kernel_entry.cpp`, `chunk_gdn_wrapper.cpp` |
| Full operator | `op_kernel/chunk_gated_delta_rule.h` |
| Host tiling | `test_chunk_gdn.py` → `build_tiling_and_workspace` |
| Stage references | `stage1_kernel.cpp`, `stage2_kernel.cpp`, `stage3_kernel.cpp` |
| Benchmarks | `benchmark_chunk_gdn.py`, `benchmark_stage_kernels.py` |
