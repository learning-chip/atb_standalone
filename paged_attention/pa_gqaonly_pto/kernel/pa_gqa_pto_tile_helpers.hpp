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
#endif
