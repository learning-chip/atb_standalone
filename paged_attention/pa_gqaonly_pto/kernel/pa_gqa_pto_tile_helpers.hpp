/**
 * PTO-ISA bridge for the paged-attention cube path.
 *
 * The production kernel still uses AscendC intrinsics (copy_gm_to_cbuf, mad, …)
 * for full tiling coverage; this header pulls in PTO-ISA so new code and
 * incremental replacements can use pto::TLOAD / pto::TMATMUL / pto::TSTORE
 * in the same translation unit. See PORT_PROGRESS.md for the migration map.
 */
#pragma once

#if defined(__DAV_C220_CUBE__) && \
    ((defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || defined(__CHECK_FEATURE_AT_PRECOMPILE__))
#include <pto/pto-inst.hpp>

namespace pa_pto {

// Upper bounds for one QK / PV matmul tile (must cover runtime m,k,n from tiling).
constexpr int kPaPtoMadMaxM = 256;
constexpr int kPaPtoMadMaxK = 1024;
constexpr int kPaPtoMadMaxN = 4096;

/**
 * FP16/BF16 matmul with FP32 accumulator using PTO-ISA (`pto::TMatmul` → hardware `mad`).
 * When m==1, uses GEMV-style `isGemv=true` so behavior matches a raw `mad(..., m=1, ...)`
 * (the generic PTO matmul path would otherwise widen m to 16 on A2A3).
 */
template <typename InDtype>
__aicore__ inline void tmatmul_fp32acc(__cc__ float *c_ptr, __ca__ InDtype *a_ptr, __cb__ InDtype *b_ptr, uint16_t m,
    uint16_t k, uint16_t n)
{
    using namespace pto;
    using TL = TileLeft<InDtype, kPaPtoMadMaxM, kPaPtoMadMaxK, DYNAMIC, DYNAMIC>;
    using TR = TileRight<InDtype, kPaPtoMadMaxK, kPaPtoMadMaxN, DYNAMIC, DYNAMIC>;
    using TC = TileAcc<float, kPaPtoMadMaxM, kPaPtoMadMaxN, DYNAMIC, DYNAMIC>;

    TL a_tile(static_cast<size_t>(m), static_cast<size_t>(k));
    TR b_tile(static_cast<size_t>(k), static_cast<size_t>(n));
    TC c_tile(static_cast<size_t>(m), static_cast<size_t>(n));

    TASSIGN(a_tile, reinterpret_cast<uintptr_t>(a_ptr));
    TASSIGN(b_tile, reinterpret_cast<uintptr_t>(b_ptr));
    TASSIGN(c_tile, reinterpret_cast<uintptr_t>(c_ptr));

    const bool kDir = GetKDirectionAlign(a_tile, b_tile);
    if (m == 1) {
        TMatmul<AccPhase::Unspecified, TC, TL, TR, false, true, true>(
            c_tile.data(), a_tile.data(), b_tile.data(), 1, k, n, kDir);
    } else {
        TMatmul<AccPhase::Unspecified, TC, TL, TR, false, true, false>(
            c_tile.data(), a_tile.data(), b_tile.data(), m, k, n, kDir);
    }
}

} // namespace pa_pto

#elif defined(__DAV_C220_VEC__) && \
    ((defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || defined(__CHECK_FEATURE_AT_PRECOMPILE__))
// Vector core: pull PTO-ISA headers for incremental migration (TADD/TEXP/…) alongside AscendC.
#include <pto/pto-inst.hpp>
#endif
