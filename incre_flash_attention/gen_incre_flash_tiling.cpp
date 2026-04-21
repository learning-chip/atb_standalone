// Standalone host utility: emit IncreFlashAttentionTilingDataV2 bytes for paged BSH GQA decode,
// mirroring IFATilingV2 paths in ops-transformer/attention/incre_flash_attention/op_host.
//
// Build: see compile_gen_tiling.sh
//
// Usage:
//   gen_incre_flash_tiling <batch> <num_heads> <num_kv_heads> <head_dim> <kv_seq_len> <block_size> \
//       <block_table_width> <total_k_blocks> <dtype:fp16|bf16> [inner_precise:0|1]
//
// Prints:
//   line 1: hex encoding of IncreFlashAttentionTilingDataV2
//   line 2: workspace_bytes (uint64)
//   line 3: block_dim (uint32) for kernel launch

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "adv_api/activation/softmax_tiling.h"
#include "adv_api/matmul/matmul_tiling.h"
#include "graph/tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t NUM16 = 16;
constexpr uint32_t NUM32 = 32;
constexpr uint32_t NUM128 = 128;
constexpr uint32_t MAX_MATMUL_BASE = 512;
constexpr uint32_t MATMUL_BASE_N = 256;
constexpr uint32_t MAX_MATMUL_BASE_M = 128;
constexpr uint32_t L0B_SIZE = 64U * 1024U;
constexpr uint32_t L0C_SIZE = 128U * 1024U;
constexpr uint32_t NUM512 = 512;
constexpr uint32_t NUM256 = 256;
constexpr uint32_t NUM1024 = 1024;
constexpr uint32_t IFA_HIGH_PERFORMANCE = 1;
constexpr uint32_t IFA_HIGH_PRECISION = 0;
constexpr uint32_t KV_CACHE_BSH = 0;

template <typename T>
T AlignUp(T v, T a) {
  return (a == 0) ? v : ((v + a - 1) / a * a);
}

uint32_t IncreGcd(uint32_t a, uint32_t b) {
  if (a % b == 0U) {
    return b;
  }
  return IncreGcd(b, a % b);
}

void AdjustPABmm1Tiling(uint32_t blockSize, uint32_t& bmm1BaseN) {
  if (bmm1BaseN < blockSize) {
    while (blockSize % bmm1BaseN != 0) {
      bmm1BaseN /= 2U;
    }
  } else if (bmm1BaseN > blockSize) {
    uint32_t tmp = IncreGcd(bmm1BaseN, blockSize);
    bmm1BaseN = tmp;
  }
}

void AdjustPABmm2Tiling(uint32_t blockSize, AscendC::tiling::TCubeTiling& bmm2) {
  uint32_t targetBaseK = NUM128;
  const uint32_t baseN = static_cast<uint32_t>(bmm2.baseN);
  if (targetBaseK < blockSize) {
    while ((blockSize % targetBaseK != 0) || (targetBaseK * baseN * sizeof(float) > L0B_SIZE)) {
      targetBaseK /= 2U;
    }
  } else {
    uint32_t tmpBaseK = IncreGcd(targetBaseK, blockSize);
    while (tmpBaseK * baseN * sizeof(float) > L0B_SIZE) {
      tmpBaseK /= 2U;
    }
    targetBaseK = tmpBaseK;
  }
  bmm2.baseK = static_cast<int32_t>(targetBaseK);
}

bool CheckConstraints(uint32_t nNumOfQInOneGroup, uint32_t headDim) {
  if (nNumOfQInOneGroup > 16U) {
    return false;
  }
  if (headDim != 128U) {
    return false;
  }
  return true;
}

bool CheckDataCopyNd2Nz(uint32_t innerPrecise, uint32_t headDim, uint32_t numKvHeads) {
  if (innerPrecise == IFA_HIGH_PRECISION) {
    return false;
  }
  if (static_cast<uint64_t>(headDim) * numKvHeads >= 65536U) {
    return false;
  }
  return true;
}

bool EnableCubeViewMM(uint32_t batchSize, uint32_t numKvHeads, uint32_t maxActualseq, bool /*pageAttention*/,
                       uint32_t innerPrecise, uint32_t headDim, uint32_t nNumOfQInOneGroup,
                       uint32_t numKvHeadsForStride) {
  if (!CheckConstraints(nNumOfQInOneGroup, headDim)) {
    return false;
  }
  if (!CheckDataCopyNd2Nz(innerPrecise, headDim, numKvHeadsForStride)) {
    return false;
  }
  uint32_t s2Loop = (maxActualseq + 2048U - 1U) / 2048U;
  if (static_cast<uint64_t>(batchSize) * numKvHeads * s2Loop <= 128U) {
    return false;
  }
  return true;
}

bool EnableAllVec(bool pageAttention, uint32_t nNumOfQInOneGroup, uint32_t headDim, bool fp16_io) {
  if (pageAttention) {
    return false;
  }
  if (nNumOfQInOneGroup > 1U) {
    return false;
  }
  if (headDim > 512U) {
    return false;
  }
  return fp16_io;
}

// Mirror op_host/incre_flash_attention_tiling_struct.h IfaPerfMode numeric order.
enum class IfaPerfMode : uint32_t {
  NORMAL = 0,
  BMM_ALL_BY_VEC = 1,
  C1_V1 = 2,
  CUBE_VIEW_MM = 3,
  CUBE_VIEW_MM_FULL_LOAD = 4,
  CUBE_VIEW_MM_MLA = 5,
  CUBE_VIEW_MM_DD = 6,
};

IfaPerfMode SetupPerfMode910(bool pageAttention, uint32_t nNumOfQInOneGroup, uint32_t headDim, uint32_t numKvHeads,
                             uint32_t batchSize, uint32_t maxActualseq, uint32_t innerPrecise, bool fp16_io) {
  if (EnableAllVec(pageAttention, nNumOfQInOneGroup, headDim, fp16_io)) {
    return IfaPerfMode::BMM_ALL_BY_VEC;
  }
  if (EnableCubeViewMM(batchSize, numKvHeads, maxActualseq, pageAttention, innerPrecise, headDim, nNumOfQInOneGroup,
                       numKvHeads)) {
    return IfaPerfMode::CUBE_VIEW_MM;
  }
  return IfaPerfMode::NORMAL;
}

uint32_t SetCoreNum910(IfaPerfMode perfMode, uint32_t aicNum, uint32_t aivNum) {
  if (perfMode == IfaPerfMode::CUBE_VIEW_MM || perfMode == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD ||
      perfMode == IfaPerfMode::CUBE_VIEW_MM_DD) {
    return aicNum;
  }
  if (perfMode == IfaPerfMode::CUBE_VIEW_MM_MLA) {
    return aicNum;
  }
  return aivNum;
}

bool IsFlashDecodefaRun(uint32_t batchSize, uint32_t numKvHeads, uint32_t nNumOfQInOneGroup, uint32_t sOuterSize,
                        uint32_t aicNum, uint32_t sMax, uint32_t sInnerSize, bool enableKVPrefix,
                        uint32_t maxActualseq) {
  constexpr float flashDecodeBNRatio = 0.4F;
  if (enableKVPrefix) {
    return false;
  }
  if (sMax < sInnerSize * 2U) {
    return false;
  }
  uint64_t bng = static_cast<uint64_t>(batchSize) * numKvHeads *
                 ((static_cast<uint64_t>(nNumOfQInOneGroup) + sOuterSize - 1) / sOuterSize);
  if ((bng < static_cast<uint64_t>(flashDecodeBNRatio * static_cast<float>(aicNum))) && (nNumOfQInOneGroup == 1U)) {
    return true;
  }
  if ((bng < static_cast<uint64_t>(flashDecodeBNRatio * static_cast<float>(aicNum))) && (maxActualseq >= 2048U)) {
    return true;
  }
  return false;
}

void CalcInnerSize910(uint32_t seqSize, uint32_t nNumOfQInOneGroup, bool antiQuantFlag, uint32_t& sInnerSize,
                      uint32_t& sInnerLoopTimes, uint32_t& sInnerSizeTail, uint32_t& sInnerSizeAlign) {
  sInnerSize = (NUM32 * NUM1024 / sizeof(float) /
                 (((nNumOfQInOneGroup + 15U) / NUM16) * NUM16));
  sInnerSize = sInnerSize / BYTE_BLOCK * BYTE_BLOCK;
  if (antiQuantFlag && nNumOfQInOneGroup > 1U) {
    sInnerSize = NUM1024;
  }
  sInnerLoopTimes = (seqSize + sInnerSize - 1U) / sInnerSize;
  sInnerSizeTail = seqSize - (sInnerLoopTimes - 1U) * sInnerSize;
  if (sInnerSize > seqSize) {
    sInnerSize = seqSize;
  }
  sInnerSizeAlign = AlignUp(sInnerSize, BYTE_BLOCK);
}

uint64_t CalcWorkSpaceV2(bool splitKVFlag, bool pageAttention, IfaPerfMode perfMode, uint32_t coreNum,
                         uint32_t /*aicNum*/, uint32_t libapiSize, uint32_t mmResUbSize, uint32_t bmm2ResUbSize,
                         uint32_t blockTypeSize, uint32_t accumOutElems, uint32_t logSumExpElems) {
  uint32_t cubeL1UbSize = (NUM512 / 2) * NUM1024;
  uint32_t cubeL0CUbSize = (NUM256 / 2) * NUM1024;
  uint32_t mmPACallBackDataSize = 64U;

  uint64_t workspaceSize = libapiSize;
  if (perfMode != IfaPerfMode::BMM_ALL_BY_VEC) {
    workspaceSize += static_cast<uint64_t>(mmResUbSize) * coreNum * 4ULL;
    workspaceSize += static_cast<uint64_t>(mmResUbSize) * coreNum * 2ULL;
    workspaceSize += static_cast<uint64_t>(bmm2ResUbSize) * coreNum * 4ULL;
    workspaceSize += static_cast<uint64_t>(bmm2ResUbSize) * coreNum * 4ULL;
    workspaceSize += static_cast<uint64_t>(bmm2ResUbSize) * coreNum * 1ULL;
  }
  workspaceSize += static_cast<uint64_t>(cubeL1UbSize) * coreNum;
  workspaceSize += static_cast<uint64_t>(cubeL0CUbSize) * coreNum;
  if (splitKVFlag) {
    workspaceSize += (static_cast<uint64_t>(accumOutElems) + static_cast<uint64_t>(logSumExpElems) * 2ULL) *
                       static_cast<uint64_t>(blockTypeSize);
  }
  if (pageAttention) {
    workspaceSize += static_cast<uint64_t>(coreNum) * mmPACallBackDataSize * 2ULL;
  }
  return workspaceSize;
}

matmul_tiling::DataType TorchDtypeToMm(bool bf16) {
  return bf16 ? matmul_tiling::DataType::DT_BF16 : matmul_tiling::DataType::DT_FLOAT16;
}

void AppendHex(const uint8_t* data, size_t n, std::ostream& os) {
  static const char* kHex = "0123456789abcdef";
  for (size_t i = 0; i < n; ++i) {
    uint8_t b = data[i];
    os << kHex[b >> 4] << kHex[b & 0x0F];
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 10 || argc > 11) {
    std::cerr << "Usage: " << argv[0]
              << " batch num_heads num_kv_heads head_dim kv_seq_len block_size block_table_width "
                 "total_k_blocks dtype(fp16|bf16) [inner_precise]\n";
    return 2;
  }

  const uint32_t batchSize = static_cast<uint32_t>(std::stoul(argv[1]));
  const uint32_t numHeads = static_cast<uint32_t>(std::stoul(argv[2]));
  const uint32_t numKvHeads = static_cast<uint32_t>(std::stoul(argv[3]));
  const uint32_t headDim = static_cast<uint32_t>(std::stoul(argv[4]));
  const uint32_t kvSeqLen = static_cast<uint32_t>(std::stoul(argv[5]));
  const uint32_t blockSize = static_cast<uint32_t>(std::stoul(argv[6]));
  const uint32_t blockTableWidth = static_cast<uint32_t>(std::stoul(argv[7]));
  const uint32_t totalBlockNum = static_cast<uint32_t>(std::stoul(argv[8]));
  std::string dt = argv[9];
  for (auto& c : dt) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  const bool bf16 = (dt == "bf16");
  const uint32_t innerPrecise =
      (argc >= 11) ? static_cast<uint32_t>(std::stoul(argv[10])) : IFA_HIGH_PERFORMANCE;

  if (batchSize == 0 || numHeads == 0 || numKvHeads == 0 || headDim == 0 || blockSize == 0) {
    std::cerr << "Invalid shape args.\n";
    return 2;
  }
  if (numHeads % numKvHeads != 0) {
    std::cerr << "num_heads must be divisible by num_kv_heads.\n";
    return 2;
  }

  auto* plat = platform_ascendc::PlatformAscendCManager::GetInstance();
  if (plat == nullptr) {
    std::cerr << "PlatformAscendCManager::GetInstance() failed.\n";
    return 1;
  }

  const uint32_t aicNum = plat->GetCoreNumAic();
  const uint32_t aivNum = plat->GetCoreNumAiv();
  const uint32_t libapiSize = plat->GetLibApiWorkSpaceSize();

  const uint32_t nNumOfQInOneGroup = numHeads / numKvHeads;
  const uint32_t sOfQuery = 1;
  const uint32_t maxActualseq = kvSeqLen;
  const uint32_t sMax = blockTableWidth * blockSize;
  const uint32_t seqSize = sMax;
  const uint32_t headDimAlign = AlignUp(headDim, BYTE_BLOCK);
  const uint32_t msdIterNum = 1;
  const bool pageAttention = true;
  const bool antiQuant = false;
  const uint32_t blockTypeSize = sizeof(float);

  uint32_t sInnerSize = 0;
  uint32_t sInnerLoopTimes = 0;
  uint32_t sInnerSizeTail = 0;
  uint32_t sInnerSizeAlign = 0;
  CalcInnerSize910(seqSize, nNumOfQInOneGroup, antiQuant, sInnerSize, sInnerLoopTimes, sInnerSizeTail, sInnerSizeAlign);

  const uint32_t sOuterSize = (sOfQuery > 1U && nNumOfQInOneGroup > 1U) ? NUM32 : NUM16;

  const bool splitKVFlag =
      IsFlashDecodefaRun(batchSize, numKvHeads, nNumOfQInOneGroup, sOuterSize, aicNum, sMax, sInnerSize, false,
                         maxActualseq);

  uint32_t kvSplitPart = 1;
  if (splitKVFlag) {
    const uint64_t bng = static_cast<uint64_t>(batchSize) * numKvHeads *
                         ((static_cast<uint64_t>(nNumOfQInOneGroup) + sOuterSize - 1) / sOuterSize);
    kvSplitPart = aicNum / static_cast<uint32_t>(std::max<uint64_t>(bng, 1ULL));
    if (kvSplitPart == 0U) {
      kvSplitPart = 1U;
    }
    uint32_t kvSplitLimit = sInnerSize <= 256U ? 256U : sInnerSize;
    while (((maxActualseq / kvSplitPart) < kvSplitLimit) && (kvSplitPart > 1U)) {
      kvSplitPart--;
    }
  }

  const bool fp16_io = !bf16;
  IfaPerfMode perfMode = SetupPerfMode910(pageAttention, nNumOfQInOneGroup, headDim, numKvHeads, batchSize,
                                          maxActualseq, innerPrecise, fp16_io);
  uint32_t coreNum = SetCoreNum910(perfMode, aicNum, aivNum);

  const uint32_t mmResUbSize = msdIterNum * nNumOfQInOneGroup * sInnerSizeAlign;
  const uint32_t bmm2ResUbSize = msdIterNum * nNumOfQInOneGroup * headDimAlign;

  ge::Shape tmpShape({nNumOfQInOneGroup, AlignUp(sInnerSize, BYTE_BLOCK / blockTypeSize)});
  uint32_t softmaxFlashTmpSize =
      AscendC::GetSoftMaxFlashV2MinTmpSize(tmpShape, blockTypeSize, blockTypeSize, true, false);

  IncreFlashAttentionTilingDataV2 tiling{};
  std::memset(&tiling, 0, sizeof(tiling));

  // Softmax tiling (same as IFATilingV2::FillTilingSoftmax)
  {
    ge::Shape softmaxShape({1, AlignUp(sInnerSize, BYTE_BLOCK / blockTypeSize)});
    AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, blockTypeSize, blockTypeSize, softmaxFlashTmpSize,
                                      tiling.tilingBase.softmaxFlashTilingData, true, false);
  }

  matmul_tiling::DataType kvType = TorchDtypeToMm(bf16);
  matmul_tiling::MatmulApiTiling bmm1(*plat);
  matmul_tiling::MatmulApiTiling bmm2(*plat);

  const uint32_t singleM = msdIterNum * nNumOfQInOneGroup;

  bmm1.SetShape(singleM, sInnerSize, headDim);
  bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
  bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, true);
  bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
  bmm1.SetOrgShape(singleM, seqSize, headDim, headDim * numKvHeads);
  bmm1.SetBias(false);

  uint32_t bmm1BaseN = std::min(AlignUp(sInnerSize, NUM16), MATMUL_BASE_N);
  AdjustPABmm1Tiling(blockSize, bmm1BaseN);
  uint32_t bmm1MaxBaseM = AlignUp(static_cast<uint32_t>(L0C_SIZE / sizeof(float) / bmm1BaseN) - NUM16, NUM16);
  if (bmm1.SetFixSplit(std::min(AlignUp(singleM, NUM16), bmm1MaxBaseM), AlignUp(bmm1BaseN, NUM16)) != 0) {
    std::cerr << "bmm1 SetFixSplit failed.\n";
    return 1;
  }
  if (bmm1.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN) != 0) {
    std::cerr << "bmm1 SetTraverse failed.\n";
    return 1;
  }
  if (bmm1.GetTiling(tiling.tilingBase.bmm1TilingData) != 0) {
    std::cerr << "bmm1 GetTiling failed.\n";
    return 1;
  }

  bmm2.SetShape(singleM, headDim, sInnerSize);
  bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
  bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
  bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, matmul_tiling::DataType::DT_FLOAT);
  bmm2.SetOrgShape(singleM, headDim * numKvHeads, sInnerSizeAlign, seqSize);
  bmm2.SetBias(false);
  if (bmm2.SetFixSplit(std::min(AlignUp(singleM, NUM16), MAX_MATMUL_BASE_M)) != 0) {
    std::cerr << "bmm2 SetFixSplit failed.\n";
    return 1;
  }
  if (bmm2.GetTiling(tiling.tilingBase.bmm2TilingData) != 0) {
    std::cerr << "bmm2 GetTiling failed.\n";
    return 1;
  }
  AdjustPABmm2Tiling(blockSize, tiling.tilingBase.bmm2TilingData);

  // Base params
  auto& bp = tiling.tilingBase.baseParams;
  bp.batchSize = batchSize;
  bp.seqSize = sMax;
  bp.qSeqSize = sOfQuery;
  bp.headSize = headDim;
  bp.headSizeV = headDim;
  bp.blockSize = blockSize;
  bp.maxBlockNumPerBatch = blockTableWidth;
  bp.maxBlockNumPerSeq = blockTableWidth;
  bp.scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));
  bp.kvHeadNum = numKvHeads;
  bp.qHeadNum = numHeads;
  bp.headNumRatio = 1;
  bp.nNumOfQInOneGroup = nNumOfQInOneGroup;
  bp.batchContinuousFlag = 1;
  bp.pseShiftFlag = 0;
  bp.actualLenDims = batchSize;
  bp.actualLenQDims = 0;
  bp.msdIterNum = msdIterNum;
  bp.qPaddingFlag = 0;
  bp.kvPaddingFlag = 0;
  bp.antiquantPerTensorFlag = 0;
  bp.antiquantPerHeadFlag = 0;
  bp.attenMaskFlag = 0;
  bp.attenMaskBatch = 0;
  bp.attenMaskQSize = 0;
  bp.attenMaskSize = 0;
  bp.sparseMode = 0;
  bp.preToken = 0;
  bp.nextToken = 0;
  bp.isRowInvalid = 0;
  bp.l2CacheOffFlag = 0;
  bp.softmaxLseFlag = 0;
  bp.totalBlockNum = totalBlockNum;
  bp.paKvShapeType = KV_CACHE_BSH;
  bp.antiqSeqSize = 0;

  // Split KV
  auto& sk = tiling.tilingBase.splitKVParams;
  sk.s2 = kvSplitPart;
  sk.sInnerLoopSize = (maxActualseq + kvSplitPart - 1) / kvSplitPart;
  sk.accumOutSize = batchSize * numHeads * kvSplitPart * headDimAlign;
  sk.logSumExpSize = batchSize * numHeads * kvSplitPart * (BYTE_BLOCK / blockTypeSize);

  // Single-core params (IFATilingV2 flash-decode path often leaves core split arrays at 0)
  auto& sc = tiling.tilingBase.increFlashAttentionSingleCoreParams;
  sc.sInnerLoopTimes = sInnerLoopTimes;
  sc.singleProcessSInnerSize = sInnerSize;
  sc.singleProcessSInnerSizeTail = sInnerSizeTail;
  sc.usedCoreNum = 0;
  sc.formerCoreNum = 0;
  sc.blockSplitBn2Range = 0;
  sc.tailSplitedBatchRange = 0;

  tiling.tilingBase.increFlashAttentionSingleCoreTensorSize.mmResUbSize = mmResUbSize;
  tiling.tilingBase.increFlashAttentionSingleCoreTensorSize.bmm2ResUbSize = bmm2ResUbSize;

  tiling.tilingBase.outputParams.needInit = 0;
  tiling.tilingBase.outputParams.isBSNDOut = 0;

  const uint32_t accumOut = splitKVFlag ? sk.accumOutSize : 0;
  const uint32_t logSumExp = splitKVFlag ? sk.logSumExpSize : 0;
  uint64_t workspaceBytes =
      CalcWorkSpaceV2(splitKVFlag, pageAttention, perfMode, coreNum, aicNum, libapiSize, mmResUbSize, bmm2ResUbSize,
                        blockTypeSize, accumOut, logSumExp);

  uint32_t blockDim = plat->CalcTschBlockDim(aivNum, aicNum, aivNum);

  AppendHex(reinterpret_cast<const uint8_t*>(&tiling), sizeof(tiling), std::cout);
  std::cout << "\n";
  std::cout << workspaceBytes << "\n";
  std::cout << blockDim << "\n";
  return 0;
}
