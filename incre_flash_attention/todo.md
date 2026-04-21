# `incre_flash_attention` standalone — remaining work

This file tracks what is **done**, what **still fails**, and **concrete next steps** so another agent can continue toward a fully working custom kernel path that matches `torch_npu.npu_incre_flash_attention` and the reference benchmark.

## Context

- **Goal**: Standalone AscendC IFA kernel + host tiling + Python launch (`ifa_custom.py`), comparable to `cann_ops_standalone/mla_prefill` and `chunk_gdn`.
- **Reference API**: `torch_npu.npu_incre_flash_attention` (see `op-plugin/docs/context/torch_npu-npu_incre_flash_attention.md`).
- **Upstream tiling**: `ops-transformer/attention/incre_flash_attention/op_host/` (especially `incre_flash_attention_tiling_v2.cpp` / `IFATilingV2`).
- **Skill workflow**: `cann_ops_standalone/.skills/extract_kernel/skills.md` (bisheng build, NPU run, compare to reference).

---

## Completed (baseline for future work)

### Preprocessor / template selection (was blocking “empty kernel”)

- **`op_kernel/ifa_standalone_preprocess.h`**  
  - Defines numeric `ORIG_DTYPE_*` and `DT_*` macros so `#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && …` in `incre_flash_attention_obp.h` matches the **FP16 × FP16 × FP16** path (device enum names are invisible to the C preprocessor).
  - `#undef TILING_KEY_VAR` / `#define TILING_KEY_VAR 10000000000300001` so `#elif TILING_KEY_VAR == QF16_…` keeps the **BSH + paged KV + flash decode + C1V2** branch at compile time (`QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING` in `incre_flash_attention_tilingkey.h`).

- **`op_kernel/incre_flash_attention.cpp`**  
  Includes `ifa_standalone_preprocess.h` **after** `kernel_operator.h` and **before** `incre_flash_attention_obp.h`.

### Runtime tiling key (was suspected for NaNs / no dispatch)

- **`op_kernel/incre_flash_attention_obp.h`**  
  At the start of `incre_flash_attention_FIAS_OBP`, sets  
  `g_tilingKey = QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING`  
  so the device sees a consistent key with the pinned template (raw `<<<>>>` launch does not preload `g_tilingKey` like the full CANN graph).

### Build / launch plumbing

- **`compile.sh`**: bisheng shared object `incre_flash_attention_lib.so` with `incre_flash_attention_wrapper.cpp` → `call_incre_flash_attention`.
- **`compile_gen_tiling.sh`** / **`gen_incre_flash_tiling`**: host tool emitting hex tiling, workspace bytes, and `block_dim`.
- **`ifa_custom.py`**: ctypes launch, subprocess tiling generation, comparison hook vs `torch_npu`.

---

## Current failure mode (as of last verification)

The kernel is **no longer an empty preprocessed shell**, but execution is **not yet correct**:

1. **Small shapes** (e.g. `kv_seq=128`, `block_size=128`): sync can fail quickly with **AICore exception** (e.g. runtime 507015) — indicates **invalid tiling, workspace usage, or tensor layout** vs what the kernel expects, not merely a slow run.
2. **Large shapes** (e.g. `kv_seq=2048`): can appear as a **long hang** on `torch.npu.synchronize()`; treat as **fault/timeout** until tiling is proven correct.

So the **remaining blocker** is **host tiling fidelity**: `IncreFlashAttentionTilingDataV2` bytes (and related launch metadata) must match what the production `IFATilingV2` path produces for the same paged BSH GQA + flash-decode scenario.

---

## Remaining issues (detailed)

### 1. `gen_incre_flash_tiling` vs `IFATilingV2` (primary)

**Problem**: `gen_incre_flash_tiling.cpp` approximates IFA v2 tiling (matmul API tiling, softmax tiling, split/core heuristics). Field-level mismatches vs `incre_flash_attention_tiling_v2.cpp` will cause **illegal memory patterns** inside the kernel (UB/MTE errors) or wrong loop bounds.

**What to do**:

- Systematically map **every field** written into `IncreFlashAttentionTilingDataV2` / nested `TCubeTiling` / `SoftMaxTiling` / `IncreFlashAttentionBaseParams` / split-KV and core tables to the **same order and semantics** as the op host (`BEGIN_TILING_DATA_DEF` / `TILING_DATA_FIELD_DEF_*` chains in upstream `incre_flash_attention_tiling.h` and friends).
- Prefer **reusing or calling** the same helper logic as upstream where possible (linking against the same tiling libraries and platform APIs), rather than re-deriving formulas by hand.
- Add a **reference dump** path: for fixed CLI args, emit tiling from the **official** host path (or a minimal GE/acl op call that only runs tiling) and **diff** hex against `gen_incre_flash_tiling` until byte-identical for representative cases.

**Files**:

- Standalone: `gen_incre_flash_tiling.cpp`, `kernel_tiling/kernel_tiling.h`
- Upstream: `ops-transformer/attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp`, `incre_flash_attention_tiling.h`

### 2. Pinned compile-time tiling key vs future variants

**Problem**: `ifa_standalone_preprocess.h` pins **`TILING_KEY_VAR`** to `10000000000300001` (BSH paged flash-decode C1V2). The kernel entry also sets **`g_tilingKey`** to the same macro. Any scenario that needs a **different** key (BF16, CALL template, non–flash-decode, etc.) will require **another bisheng compile** or a **cleaner** multi-key story.

**What to do**:

- Document which **benchmark** / API cases use which key (cross-check `GenTilingKey` / `UpdateTilingKey*` in `IFATilingV2`).
- Longer term: either multiple standalone `.so` builds with `-DTILING_KEY_VAR=…` + matching `g_tilingKey` init, or upstream-style dynamic compilation (heavier).

### 3. `block_dim` and workspace

**Problem**: `gen_incre_flash_tiling` prints `block_dim` and workspace size. These must stay consistent with **`PlatformAscendC` / `CalcTschBlockDim`** (or equivalent) and with **workspace partitioning** inside the kernel (including any **system workspace** / FFTS base if the kernel expects it — compare `chunk_gdn` wrapper pattern with `rtGetC2cCtrlAddr`).

**What to do**:

- Compare `block_dim` and workspace with values logged or computed in upstream `IFATilingV2::CalcNumBlocks` / workspace sizing.
- Confirm whether IFA standalone wrapper must pass an **FFTS** or other **first argument** like `chunk_gdn` (current IFA wrapper does **not**; only change if kernel or arch32 path requires it).

### 4. Python ctypes ABI vs optional tensors

**Problem**: `call_incre_flash_attention` must match the **exact** argument list expected by the compiled kernel entry (`incre_flash_attention`). Optional inputs are often **null**; wrong count or order causes subtle failures.

**What to do**:

- Re-verify against `op_kernel/incre_flash_attention.cpp` and the torch API doc: `pseShift`, `attenMask`, quant/antiquant pointers, `kvPaddingSize`, etc.
- Keep a **table** in code comments mapping Python `None` → `ctypes.c_void_p(0)`.

### 5. Numerical and performance parity

**Problem**: Even after AICore runs clean, outputs must match **`torch_npu.npu_incre_flash_attention`** within tolerance, and timings should be in the same ballpark as `cann_ops_standalone/ifa/bench_ifa_gpa_paged.py`.

**What to do**:

- Add a **small** correctness test (few shapes, fp16 + optionally bf16) and a **benchmark** mode that mirrors `bench_ifa_gpa_paged.py` (same `scale_value`, `actual_seq_lengths`, `block_table`, `block_size`).
- Use `npu-smi info` to pick a free device; set `NPU_ID` / `torch.npu.set_device` consistently.

### 6. `actual_seq_lengths` dtype and layout

**Problem**: Reference path uses `actual_seq_lengths=act.cpu().tolist()` in Python; custom path passes an **NPU int64** tensor. The kernel expects a particular **GM layout and dtype** per IFA; mismatch causes bad sequence lengths in the kernel.

**What to do**:

- Confirm from op proto / tiling: **int64** vs **int32**, shape `(B,)` vs other, and whether values must live on CPU only for the official API while the custom kernel needs GM — align `ifa_custom.py` with the documented behavior.

---

## Suggested next steps (ordered)

1. **Golden tiling bytes**  
   For one fixed configuration (e.g. `B=1`, `H=32`, `KVH=8`, `D=128`, `kv_seq=2048`, `block_size=128`, fp16, BSH paged), obtain **reference** `IncreFlashAttentionTilingDataV2` bytes from upstream host tiling (or instrumented OPP) and **diff** against `gen_incre_flash_tiling` output; fix generator until **match**.

2. **Minimal NPU test**  
   After bytes match reference, run `ifa_custom.py` on a **tiny** case (`kv_seq == block_size`) and expand; confirm no 507015 / AICore fault.

3. **Scale_value and edge cases**  
   Align `scale_value` handling with `IFATilingV2` / `baseParams.scaleValue` if the kernel reads scale from tiling rather than only from user tensors.

4. **Document** in `README.md` (when stable): build order (`compile.sh`, `compile_gen_tiling.sh`), env vars (`ASCEND_TOOLKIT_HOME`, `NPU_ID`), pinned tiling key, and known limitations (single-key build).

5. **Optional**: add a **CI-style** script that runs `gen_incre_flash_tiling`, loads `.so`, runs one micro-case, and compares to `torch_npu` (same machine with NPU).

---

## Quick reference paths

| Item | Path |
|------|------|
| Standalone kernel + wrapper | `cann_ops_standalone/incre_flash_attention/` |
| Preprocessor shims | `op_kernel/ifa_standalone_preprocess.h`, `op_kernel/incre_flash_attention_obp.h` (g_tilingKey line) |
| Host tiling generator | `gen_incre_flash_tiling.cpp`, `compile_gen_tiling.sh` |
| Python launcher | `ifa_custom.py` |
| Reference benchmark | `cann_ops_standalone/ifa/bench_ifa_gpa_paged.py` |
| Upstream tiling v2 | `ops-transformer/attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp` |
| Tiling key constants | `op_kernel/incre_flash_attention_tilingkey.h` |
| Extract skill | `cann_ops_standalone/.skills/extract_kernel/skills.md` |

---

## Notes for agents

- Do **not** remove `ifa_standalone_preprocess.h` without replacing the **dtype** and **TILING_KEY_VAR** behavior; the kernel will compile to an **empty** body again.
- Treat **AICore exceptions** after launch as **tiling/layout bugs** until reference bytes match.
- Keep changes **scoped** to standalone files unless the user asks to modify `ops-transformer` upstream.
