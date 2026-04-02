# Chunk GDN (standalone)

Build a standalone shared library for the `chunk_gated_delta_rule` kernel and run on-device tests.

## Full kernel (`chunk_gdn_lib.so`)

```bash
bash ./compile.sh
python ./test_chunk_gdn.py
```

## Benchmark (timing + CSV)

Uses `torch.npu.Event` timing (same idea as `mla_prefill`), reports estimated TFLOP/s and **effective** GiB/s (user-visible inputs, tiling, and final outputs only; **excludes** GM workspace scratch), and writes `benchmark_chunk_gdn.csv`.

```bash
bash ./compile.sh
python ./benchmark_chunk_gdn.py
```

Use `python ./benchmark_chunk_gdn.py --torch-ref-only` to time **only** the PyTorch reference (`cgdr_benchmark_bf16`) if you need a baseline without launching the custom kernel (e.g. device issues after a bad kernel run).

### Measured performance (snapshot)

Hardware / run: Ascend **910B2**, `NPU_ID=0`, **2026-04-02**, command:

`python benchmark_chunk_gdn.py --torch-ref-only`

(warmup 5, timed iterations 20; TFLOP/s = `estimate_chunk_gdn_flops` / time; GiB/s = `effective_io_bytes / time`. Same FLOP scalar is used for both custom and torch columns so TFLOP/s are comparable.)

| Case | T | nk/nv | dk/dv | torch ref ms | est. TFLOP/s | eff. GiB/s |
|------|---|-------|-------|--------------|--------------|------------|
| gdn_b1_s64_h4 | 64 | 4 / 4 | 128 / 128 | 15.03 | 0.0039 | 0.0326 |
| gdn_b1_s512_h4 | 512 | 4 / 4 | 128 / 128 | 18.60 | 0.0253 | 0.1188 |
| gdn_b1_s2048_h4 | 2048 | 4 / 4 | 128 / 128 | 31.95 | 0.0588 | 0.2536 |
| gdn_b1_s4096_h4 | 4096 | 4 / 4 | 128 / 128 | 49.93 | 0.0753 | 0.3197 |

**Custom kernel (`chunk_gdn_lib.so`):** `call_kernel` now mirrors the staged entry points: `rtGetC2cCtrlAddr` + `SetSyncBaseAddr` / `SetAtomicNone` / `SetMaskNorm`, and `KERNEL_TYPE_MIX_AIC_1_2` (Stage1/2/3 rely on `CrossCoreWaitFlag`). Host tiling in `build_tiling_and_workspace` uses per-core matmul shapes consistent with `default_matmul_tiling` (splitting `singleCoreM` across cores broke cube `Init`).

Run `python test_chunk_gdn.py` after `bash ./compile.sh`. If that passes, run `python benchmark_chunk_gdn.py` (no flags): it times the **torch reference first**, then smoke-checks and times the custom kernel, and fills `custom_*` columns in `benchmark_chunk_gdn.csv`. On the **910B2** host used for this snapshot, the fused binary still hit **507057** at synchronize while `test_stage{1,2,3}.py` succeed, so custom-kernel ms are not listed above—use another CANN/card revision or collect plog if you need fused numbers.

## Staged probes (real Stage1 / Stage2 / Stage3; sources under `op_kernel/`)

Each stage is a small `.so` plus a Python test. Stage2 and Stage3 tests chain after Stage1.

```bash
bash ./compile_stage1.sh && python ./test_stage1.py
bash ./compile_stage2.sh && python ./test_stage2.py
bash ./compile_stage3.sh && python ./test_stage3.py
```

Shared helpers: `chunk_gdn_common.py`. Tiling ctypes layouts live in `test_chunk_gdn.py`.

**Staged vs fused:** `test_stage{1,2,3}.py` each launch a **custom** AscendC kernel (`stage*_lib.so`) and pass. The **fused** `chunk_gdn_lib.so` path used by `test_chunk_gdn.py` / `benchmark_chunk_gdn.py` can still fail at runtime (**507057**) on some hosts while stages succeed. See **`remaining_issue.md`** for symptoms, hypotheses, and next steps.

### Per-stage kernel benchmark (TFLOP/s and GiB/s)

Requires `stage{1,2,3}_lib.so` built first:

```bash
bash ./compile_stage1.sh && bash ./compile_stage2.sh && bash ./compile_stage3.sh
python ./benchmark_stage_kernels.py
```

Output: **`benchmark_stage_kernels.csv`**. Timing: `torch.npu.Event` (warmup 5, timed iters 20). **TFLOP/s** uses per-stage heuristics `estimate_stage1_flops` / `estimate_stage2_flops` / `estimate_stage3_flops` in the script. **GiB/s** divides total **operand footprint** (GM tensors + tiling passed to that stage’s `call_stage*`, excluding uint8 `workspace`) by kernel time—roofline-style, not HBM counter data.

Shapes in the script: **`nk=nv=1`**, **`dk=dv=64`**, **`chunk=64`**, **`T ∈ {64,512,2048,4096}`** (same family as `test_stage1.py`). Stage2 and Stage3 timings restore tensors from snapshots each iteration so repeated launches stay valid.

#### Measured snapshot (Ascend **910B2**, `NPU_ID=0`, **2026-04-02**)

| T | Stage1 ms | S1 est. TFLOP/s | S1 op GiB/s | Stage2 ms | S2 est. TFLOP/s | S2 op GiB/s | Stage3 ms | S3 est. TFLOP/s | S3 op GiB/s |
|---|-----------|-----------------|-------------|-----------|-----------------|-------------|-----------|-----------------|-------------|
| 64 | 0.085 | 0.019 | 4.05 | 0.312 | 0.013 | 0.30 | 0.279 | 0.008 | 1.07 |
| 512 | 0.083 | 0.152 | 12.56 | 0.310 | 0.108 | 2.03 | 0.272 | 0.062 | 2.47 |
| 2048 | 0.082 | 0.611 | 41.64 | 0.242 | 0.554 | 10.17 | 0.203 | 0.331 | 9.66 |
| 4096 | 0.105 | 0.957 | 62.87 | 0.273 | 0.985 | 18.02 | 0.205 | 0.653 | 17.90 |

## Notes

- Set `NPU_ID` if you need a specific device (default `0`). Example: `NPU_ID=7 python ./test_stage1.py`.
- `call_stage1` / `call_stage2` / `call_stage3` use `rtGetC2cCtrlAddr` + `SetSyncBaseAddr` for FFTS cross-core sync (same idea as the full kernel).
- The full-kernel test compares against the Python reference in `test_chunk_gdn.py`; staged tests use small torch references for their slice of the pipeline.
- Staged tests **L2-normalize** `query` and `key` on the last dimension (same as `test_chunk_gdn.py`). Without that, Stage1 can emit NaNs in `k_cum_decay` / `v_inner` for arbitrary random draws.
