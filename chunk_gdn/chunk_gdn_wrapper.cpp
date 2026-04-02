// Standalone wrapper to launch `chunk_gated_delta_rule` via ctypes.
//
// This file intentionally includes only the required kernel source
// (vendored under op_kernel/, plus local shims).

#include <stdint.h>

// AscendC types/keywords (bfloat16_t, __gm__, etc) are pulled in by the kernel include below.
#include "chunk_gdn_kernel_entry.cpp"

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *query,      // GM bfloat16
    uint8_t *key,        // GM bfloat16
    uint8_t *value,      // GM bfloat16
    uint8_t *beta,       // GM bfloat16
    uint8_t *initialState,  // GM bfloat16, shape (B, Nv, Dv, Dk)
    uint8_t *seqlens,    // GM int32, shape (B,)
    uint8_t *gOptional,  // GM float, shape (T, Nv); pass nullptr for hasGamma=0
    uint8_t *out,        // GM bfloat16, shape (T, Nv, Dv)
    uint8_t *finalState, // GM bfloat16, shape (B, Nv, Dv, Dk)
    uint8_t *workspaceGM,  // GM uint8 workspace (must include 16MB system workspace)
    uint8_t *tilingGM      // GM uint8 tiling struct bytes
) {
    // Upstream UT uses AIV/AIC kernel mode switching, but this standalone
    // bisheng build doesn't expose `AscendC::SetKernelMode(...)`. So we run
    // with the default kernel mode selected by CANN for this kernel type.

    // Kernel expects workspaceGM/tilingGM as GM_ADDR pointers.
    chunk_gated_delta_rule<<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t *)query,
        (__gm__ uint8_t *)key,
        (__gm__ uint8_t *)value,
        (__gm__ uint8_t *)beta,
        (__gm__ uint8_t *)initialState,
        (__gm__ uint8_t *)seqlens,
        (__gm__ uint8_t *)gOptional,
        (__gm__ uint8_t *)out,
        (__gm__ uint8_t *)finalState,
        (__gm__ uint8_t *)workspaceGM,
        (__gm__ uint8_t *)tilingGM
    );
}

