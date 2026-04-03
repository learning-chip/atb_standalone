# Chunk GDN (standalone)

Standalone extraction of `chunk_gated_delta_rule`.

`compile.sh` now builds the three verified stage libraries:

```bash
bash ./compile.sh
python ./test_chunk_gdn.py
```

`test_chunk_gdn.py` runs an end-to-end custom path by launching `stage1_lib.so` -> `stage2_lib.so` -> `stage3_lib.so` on the same stream. This avoids the unstable monolithic `chunk_gdn_lib.so` launch path while still exercising the real AscendC stage kernels.

## End-to-end Benchmark

```bash
bash ./compile.sh
python ./benchmark_chunk_gdn.py
```

`benchmark_chunk_gdn.py` times:

- the staged custom kernel pipeline
- the PyTorch reference `cgdr_benchmark_bf16`

Timing uses `torch.npu.Event` with warmup `5` and timed iterations `20`.

`effective TFLOP/s` uses `estimate_chunk_gdn_flops` in the script.
`effective GiB/s` uses the summed Stage1/2/3 operand footprint passed through GM, excluding uint8 workspaces.

### Measured Snapshot

Hardware / run: Ascend **910B2**, `NPU_ID=7`, **2026-04-03**

| Case | T | nk/nv | dk/dv | custom ms | custom TFLOP/s | custom GiB/s | torch ref ms | torch ref TFLOP/s | torch ref GiB/s | speedup |
|------|---|-------|-------|-----------|----------------|--------------|--------------|-------------------|-----------------|---------|
| gdn_b1_s4096_h4 | 4096 | 4 / 4 | 64 / 64 | 0.555 | 2.17 | 108.74 | 47.87 | 0.0252 | 1.26 | 86.2x |
| gdn_b1_s16384_h4 | 16384 | 4 / 4 | 64 / 64 | 1.930 | 2.50 | 122.82 | 153.01 | 0.0316 | 1.55 | 79.3x |
| gdn_b1_s65536_h4 | 65536 | 4 / 4 | 64 / 64 | 8.208 | 2.35 | 114.95 | 860.39 | 0.0225 | 1.10 | 104.8x |

The custom stage pipeline is now about **79x-105x** faster than the PyTorch reference on these cases.

## Per-stage Benchmark

```bash
bash ./compile_stage1.sh
bash ./compile_stage2.sh
bash ./compile_stage3.sh
python ./benchmark_stage_kernels.py
```

Output: `benchmark_stage_kernels.csv`

Shapes used now match the stable end-to-end family: `nk=nv=4`, `dk=dv=64`, `chunk=64`, gamma enabled.

### Measured Snapshot

Hardware / run: Ascend **910B2**, `NPU_ID=7`, **2026-04-03**

| T | Stage1 ms | S1 TFLOP/s | S1 GiB/s | Stage2 ms | S2 TFLOP/s | S2 GiB/s | Stage3 ms | S3 TFLOP/s | S3 GiB/s |
|---|-----------|------------|----------|-----------|------------|----------|-----------|------------|----------|
| 4096 | 0.230 | 1.75 | 114.27 | 0.272 | 3.95 | 72.27 | 0.208 | 2.58 | 69.51 |
| 16384 | 0.914 | 1.76 | 112.58 | 0.889 | 4.83 | 88.26 | 0.249 | 8.61 | 223.28 |
| 65536 | 3.561 | 1.81 | 114.99 | 4.295 | 4.00 | 73.01 | 1.032 | 8.33 | 213.73 |

## Notes

- Set `NPU_ID` if you want a specific device, for example `NPU_ID=7 python ./benchmark_chunk_gdn.py`.
- `ai_core_num_from_device()` now uses the full `cube_core_num` reported by the device, rather than dividing it by `3`.
- `stage2_kernel.cpp` and `stage3_kernel.cpp` now propagate `tilingData.hasGamma`, which fixes the gamma-enabled all-stage benchmark path.
- `test_chunk_gdn.py` validates the staged custom path against the bf16 torch reference with relaxed max/mean-abs thresholds that match the observed cast error envelope of the standalone staged kernels.
- The legacy monolithic `chunk_gdn_lib.so` source is still present in the tree for debugging, but the supported benchmark/test path is the staged custom pipeline built by `compile.sh`.
