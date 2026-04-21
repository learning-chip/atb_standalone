// Standalone IFA: extend AscendC::tiling with IncreFlashAttention tiling structs.
// Base types (TCubeTiling, SoftMaxTiling, ...) come from CANN highlevel kernel_tiling.h.
#pragma once

#include <cstddef>
#include <cstdint>

#include "highlevel_api/kernel_tiling/kernel_tiling.h"

namespace AscendC {
namespace tiling {

#pragma pack(push, 8)
struct IncreFlashAttentionInitOutputParams {
    uint32_t isPerChnOut = 0;
    uint32_t isOutQuantTypeBf16 = 0;
    uint32_t singleCoreSize = 0;
    uint32_t singleCoreLseSize = 0;
    int64_t totalOutputSize = 0;
    int64_t totalLseOutputSize = 0;
    uint32_t needInit = 0;
    uint32_t isBSNDOut = 0;
};

struct IncreFlashAttentionBaseParams {
    uint32_t batchSize = 0;
    uint32_t seqSize = 0;
    uint32_t qSeqSize = 0;
    uint32_t headSize = 0;
    uint32_t headSizeV = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t maxBlockNumPerSeq = 0;
    float scaleValue = 0.0f;
    uint32_t kvHeadNum = 0;
    uint32_t headNumRatio = 0;
    uint32_t qHeadNum = 0;
    uint32_t nNumOfQInOneGroup = 0;
    uint32_t batchContinuousFlag = 0;
    uint32_t pseShiftFlag = 0;
    uint32_t pseShiftB = 0;
    uint32_t pseShiftS = 0;
    uint32_t pseShiftS0 = 0;
    uint32_t selectWithByteMaskTmpMinSize = 0;
    uint32_t actualLenQDims = 0;
    uint32_t actualLenDims = 0;
    uint32_t qPaddingFlag = 0;
    uint32_t kvPaddingFlag = 0;
    uint32_t msdIterNum = 0;
    uint32_t l2CacheOffFlag = 0;
    uint32_t antiquantPerTensorFlag = 0;
    uint32_t antiquantPerHeadFlag = 0;
    uint32_t antiquantParamsInPagedAttentionFlag = 0;
    uint32_t attenMaskFlag = 0;
    uint32_t attenMaskBatch = 0;
    uint32_t attenMaskQSize = 0;
    uint32_t attenMaskSize = 0;
    uint32_t softmaxLseFlag = 0;
    uint32_t totalBlockNum = 0;
    uint32_t paKvShapeType = 0;
    uint32_t antiqSeqSize = 0;
    int32_t preToken = 0;
    int32_t nextToken = 0;
    uint32_t isRowInvalid = 0;
    uint32_t sparseMode = 0;
    uint32_t slidingFlag = 0;
    int64_t windowSize = 0;
};

struct IncreFlashAttentionCoreParams {
    uint32_t coreSidxEnd[50];
    uint32_t coreSidxEndRegbase[66];
    uint32_t coreSposStartRegbase[66];
};

struct IncreFlashAttentionSplitCoreParams {
    uint32_t headSplit = 0;
    uint32_t maskHeadStride = 0;
    uint32_t maskBatchStride = 0;
    uint32_t qTokens = 0;
    uint32_t isTriu = 0;
    uint32_t maxSeqlen = 0;
    uint32_t totalQBlockNum = 0;
    uint32_t seqStepQ = 0;
    uint32_t seqStepKv = 0;
    uint32_t startBlk[50];
    uint32_t endBlk[50];
    uint32_t startBatch[50];
    uint32_t endBatch[50];
};

struct IncreFlashAttentionSingleCoreParams {
    uint32_t sInnerLoopTimes = 0;
    uint32_t singleProcessSInnerSize = 0;
    uint32_t singleProcessSInnerSizeTail = 0;
    uint32_t usedCoreNum = 0;
    uint32_t formerCoreNum = 0;
    uint32_t blockSplitBn2Range = 0;
    uint32_t tailSplitedBatchRange = 0;
    uint32_t groupSplitSize = 0;
    uint32_t s1SplitSize = 0;
};

struct IncreFlashAttentionSingleCoreTensorSize {
    uint32_t mmResUbSize = 0;
    uint32_t bmm2ResUbSize = 0;
};

struct IncreFlashAttentionSplitKVParams {
    uint32_t s2 = 0;
    uint32_t sInnerLoopSize = 0;
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
};

struct IncreFlashAttentionTilingData {
    TCubeTiling bmm1TilingData;
    TCubeTiling bmm2TilingData;
    IncreFlashAttentionBaseParams baseParams;
    IncreFlashAttentionSplitKVParams splitKVParams;
    IncreFlashAttentionCoreParams increFlashAttentionCoreParams;
    IncreFlashAttentionSingleCoreParams increFlashAttentionSingleCoreParams;
    IncreFlashAttentionSingleCoreTensorSize increFlashAttentionSingleCoreTensorSize;
    SoftMaxTiling softmaxFlashTilingData;
    IncreFlashAttentionInitOutputParams outputParams;
};

struct IncreFlashAttentionTilingDataPrefix {
    IncreFlashAttentionTilingData base;
    uint64_t prefixAttenOutOffset = 0;
    uint64_t userPromptAttenOutOffset = 0;
    uint64_t tmpLseOffset = 0;
    uint64_t prefixLen = 0;
    uint32_t formerCoreNum = 0;
    uint32_t blockSplitBn2Range = 0;
    uint32_t tailSplitedBatchRange = 0;
    uint32_t usedCoreNum = 0;
    uint32_t batchSizeQ = 0;
};

struct IncreFlashAttentionTilingDataV2 {
    IncreFlashAttentionTilingData tilingBase;
    IncreFlashAttentionTilingDataPrefix tilingPrefix;
};

struct IncreFlashAttentionTilingAtbDataV2 {
    IncreFlashAttentionBaseParams tilingBase;
    IncreFlashAttentionSplitCoreParams tilingPerCore;
};

#pragma pack(pop)

} // namespace tiling
} // namespace AscendC

using AscendC::tiling::IncreFlashAttentionTilingData;
using AscendC::tiling::IncreFlashAttentionTilingDataV2;
using AscendC::tiling::IncreFlashAttentionTilingDataPrefix;
using AscendC::tiling::IncreFlashAttentionTilingAtbDataV2;

// -----------------------------------------------------------------------------
// Standalone tiling load from GM (same idea as cann_ops_standalone/chunk_gdn):
// copy bytes from the tiling pointer into local structs so the layout matches
// `optiling` serialization in ops-transformer/attention/incre_flash_attention/op_host/incre_flash_attention_tiling.h
// (field order above mirrors BEGIN_TILING_DATA_DEF / TILING_DATA_FIELD_DEF_*).
//
// Offsets use the null-pointer member address idiom: (size_t)&((T*)0)->member, including nested members
// such as tilingBase.bmm1TilingData or tilingPrefix.base.bmm2TilingData.
// -----------------------------------------------------------------------------

#ifdef GET_TILING_DATA_MEMBER
#undef GET_TILING_DATA_MEMBER
#endif
#ifdef GET_TILING_DATA_WITH_STRUCT
#undef GET_TILING_DATA_WITH_STRUCT
#endif

#define GET_TILING_DATA_MEMBER(tilingType, member, var, tiling)                                                                \
    decltype(((tilingType *)nullptr)->member) var;                                                                              \
    do {                                                                                                                       \
        __gm__ uint8_t const *__ifa_gm = reinterpret_cast<__gm__ uint8_t const *>(tiling);                                      \
        const size_t __ifa_off = static_cast<size_t>(reinterpret_cast<uintptr_t>(&(((tilingType *)0)->member)));               \
        uint8_t *__ifa_dst = reinterpret_cast<uint8_t *>(&var);                                                              \
        for (size_t __ifa_i = 0; __ifa_i < sizeof(var); ++__ifa_i) {                                                           \
            __ifa_dst[__ifa_i] = __ifa_gm[__ifa_off + __ifa_i];                                                                \
        }                                                                                                                      \
    } while (0)

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg)                                                    \
    tiling_struct tiling_data;                                                                                                 \
    do {                                                                                                                       \
        __gm__ uint8_t const *__ifa_gm = reinterpret_cast<__gm__ uint8_t const *>(tiling_arg);                                \
        uint8_t *__ifa_dst = reinterpret_cast<uint8_t *>(&tiling_data);                                                        \
        for (size_t __ifa_i = 0; __ifa_i < sizeof(tiling_struct); ++__ifa_i) {                                                  \
            __ifa_dst[__ifa_i] = __ifa_gm[__ifa_i];                                                                            \
        }                                                                                                                      \
    } while (0)
