# Chunk GDN (standalone)

Build a standalone shared library for the `chunk_gated_delta_rule` kernel and run on-device tests.

## Full kernel (`chunk_gdn_lib.so`)

```bash
bash ./compile.sh
python ./test_chunk_gdn.py
```

## Staged probes (real Stage1 / Stage2 / Stage3; sources under `op_kernel/`)

Each stage is a small `.so` plus a Python test. Stage2 and Stage3 tests chain after Stage1.

```bash
bash ./compile_stage1.sh && python ./test_stage1.py
bash ./compile_stage2.sh && python ./test_stage2.py
bash ./compile_stage3.sh && python ./test_stage3.py
```

Shared helpers: `chunk_gdn_common.py`. Tiling ctypes layouts live in `test_chunk_gdn.py`.

## Notes

- Set `NPU_ID` if you need a specific device (default `0`). Example: `NPU_ID=7 python ./test_stage1.py`.
- `call_stage1` / `call_stage2` / `call_stage3` use `rtGetC2cCtrlAddr` + `SetSyncBaseAddr` for FFTS cross-core sync (same idea as the full kernel).
- The full-kernel test compares against the Python reference in `test_chunk_gdn.py`; staged tests use small torch references for their slice of the pipeline.
- Staged tests **L2-normalize** `query` and `key` on the last dimension (same as `test_chunk_gdn.py`). Without that, Stage1 can emit NaNs in `k_cum_decay` / `v_inner` for arbitrary random draws.
