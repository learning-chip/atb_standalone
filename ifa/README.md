# IFA GQA benchmark (`bench_ifa_gpa.py`)

Benchmarks `torch_npu.npu_incre_flash_attention` on Qwen3-style Grouped-Query Attention shapes (decode `q_seq=1`). Metrics use the Event-based timer from `mla_prefill_cce/test_mla_prefill.py` (mean of 20 timed iterations after 5 warmup runs, `torch.float16`).

**Theory:** TFLOP/s = (QK+PV matmul FLOPs) / time; GiB/s = minimum tensor traffic (Q+K+V+O) / time. Values depend on device and software stack.

## Measured throughput (latest `python3 bench_ifa_gpa.py` run on this environment)

| Case | Time (ms) | TFLOP/s | Bandwidth (GiB/s) |
|------|-----------|---------|-------------------|
| Qwen3-0.6B GQA b1 h16/kv8 kv2048 | 0.1556 | 0.1078 | 50.27 |
| Qwen3-1.7B GQA b1 h16/kv8 kv4096 | 0.1493 | 0.2247 | 104.67 |
| Qwen3-4B GQA b1 h32/kv8 kv2048 | 0.1478 | 0.2270 | 52.95 |
| Qwen3-8B GQA b1 h32/kv8 kv4096 | 0.1416 | 0.4739 | 110.44 |
| Qwen3-8B GQA b1 h32/kv8 kv8192 | 0.1463 | 0.9177 | 213.77 |
| Qwen3-14B GQA b1 h40/kv8 kv2048 | 0.1518 | 0.2762 | 51.58 |
| Qwen3-32B GQA b1 h64/kv8 kv2048 | 0.1433 | 0.4685 | 54.75 |
| MHA synthetic b1 h32/kv32 kv2048 (not Qwen3-8B) | 0.1415 | 0.2372 | 220.99 |
| Qwen3-8B GQA b4 h32/kv8 kv2048 | 0.1424 | 0.9424 | 219.85 |
| Qwen3-8B GQA b8 h32/kv8 kv2048 | 0.1415 | 1.8969 | 442.52 |
| Qwen3-8B GQA b16 h32/kv8 kv2048 | 0.1584 | 3.3889 | 790.59 |
| Qwen3-8B GQA b32 h32/kv8 kv2048 | 0.2568 | 4.1812 | 975.41 |
| Qwen3-8B GQA b64 h32/kv8 kv2048 | 0.5107 | 4.2046 | 980.87 |

Run: `python3 bench_ifa_gpa.py` (optional: `--bf16`, `--warmup N`, `--iters N`).
