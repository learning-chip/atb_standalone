"""
ctypes layouts for IFA tiling (IncreFlashAttentionTilingDataV2 and nested types).

Field order matches:
- `kernel_tiling/kernel_tiling.h` (device structs, #pragma pack 8)
- `ops-transformer/attention/incre_flash_attention/op_host/incre_flash_attention_tiling.h` (optiling defs)

Use `tiling_to_device` to build a uint8 device tensor for `call_incre_flash_attention` (same pattern as
`cann_ops_standalone/chunk_gdn/chunk_gdn_common.py`).

Filling valid values still requires CANN op-host tiling logic (or a ported subset); this module only
provides the packed layout and helpers.
"""

from __future__ import annotations

import ctypes

import numpy as np
import torch


class TCubeTiling(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("usedCoreNum", ctypes.c_int32),
        ("M", ctypes.c_int32),
        ("N", ctypes.c_int32),
        ("Ka", ctypes.c_int32),
        ("Kb", ctypes.c_int32),
        ("singleCoreM", ctypes.c_int32),
        ("singleCoreN", ctypes.c_int32),
        ("singleCoreK", ctypes.c_int32),
        ("baseM", ctypes.c_int32),
        ("baseN", ctypes.c_int32),
        ("baseK", ctypes.c_int32),
        ("depthA1", ctypes.c_int32),
        ("depthB1", ctypes.c_int32),
        ("stepM", ctypes.c_int32),
        ("stepN", ctypes.c_int32),
        ("isBias", ctypes.c_int32),
        ("transLength", ctypes.c_int32),
        ("iterateOrder", ctypes.c_int32),
        ("shareMode", ctypes.c_int32),
        ("shareL1Size", ctypes.c_int32),
        ("shareL0CSize", ctypes.c_int32),
        ("shareUbSize", ctypes.c_int32),
        ("batchM", ctypes.c_int32),
        ("batchN", ctypes.c_int32),
        ("singleBatchM", ctypes.c_int32),
        ("singleBatchN", ctypes.c_int32),
        ("stepKa", ctypes.c_int32),
        ("stepKb", ctypes.c_int32),
        ("depthAL1CacheUB", ctypes.c_int32),
        ("depthBL1CacheUB", ctypes.c_int32),
        ("dbL0A", ctypes.c_int32),
        ("dbL0B", ctypes.c_int32),
        ("dbL0C", ctypes.c_int32),
        ("ALayoutInfoB", ctypes.c_int32),
        ("ALayoutInfoS", ctypes.c_int32),
        ("ALayoutInfoN", ctypes.c_int32),
        ("ALayoutInfoG", ctypes.c_int32),
        ("ALayoutInfoD", ctypes.c_int32),
        ("BLayoutInfoB", ctypes.c_int32),
        ("BLayoutInfoS", ctypes.c_int32),
        ("BLayoutInfoN", ctypes.c_int32),
        ("BLayoutInfoG", ctypes.c_int32),
        ("BLayoutInfoD", ctypes.c_int32),
        ("CLayoutInfoB", ctypes.c_int32),
        ("CLayoutInfoS1", ctypes.c_int32),
        ("CLayoutInfoN", ctypes.c_int32),
        ("CLayoutInfoG", ctypes.c_int32),
        ("CLayoutInfoS2", ctypes.c_int32),
        ("BatchNum", ctypes.c_int32),
        ("mxTypePara", ctypes.c_int32),
    ]


class SoftMaxTiling(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("srcM", ctypes.c_uint32),
        ("srcK", ctypes.c_uint32),
        ("srcSize", ctypes.c_uint32),
        ("outMaxM", ctypes.c_uint32),
        ("outMaxK", ctypes.c_uint32),
        ("outMaxSize", ctypes.c_uint32),
        ("splitM", ctypes.c_uint32),
        ("splitK", ctypes.c_uint32),
        ("splitSize", ctypes.c_uint32),
        ("reduceM", ctypes.c_uint32),
        ("reduceK", ctypes.c_uint32),
        ("reduceSize", ctypes.c_uint32),
        ("rangeM", ctypes.c_uint32),
        ("tailM", ctypes.c_uint32),
        ("tailSplitSize", ctypes.c_uint32),
        ("tailReduceSize", ctypes.c_uint32),
    ]


class IncreFlashAttentionInitOutputParams(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("isPerChnOut", ctypes.c_uint32),
        ("isOutQuantTypeBf16", ctypes.c_uint32),
        ("singleCoreSize", ctypes.c_uint32),
        ("singleCoreLseSize", ctypes.c_uint32),
        ("totalOutputSize", ctypes.c_int64),
        ("totalLseOutputSize", ctypes.c_int64),
        ("needInit", ctypes.c_uint32),
        ("isBSNDOut", ctypes.c_uint32),
    ]


class IncreFlashAttentionBaseParams(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("batchSize", ctypes.c_uint32),
        ("seqSize", ctypes.c_uint32),
        ("qSeqSize", ctypes.c_uint32),
        ("headSize", ctypes.c_uint32),
        ("headSizeV", ctypes.c_uint32),
        ("blockSize", ctypes.c_uint32),
        ("maxBlockNumPerBatch", ctypes.c_uint32),
        ("maxBlockNumPerSeq", ctypes.c_uint32),
        ("scaleValue", ctypes.c_float),
        ("kvHeadNum", ctypes.c_uint32),
        ("headNumRatio", ctypes.c_uint32),
        ("qHeadNum", ctypes.c_uint32),
        ("nNumOfQInOneGroup", ctypes.c_uint32),
        ("batchContinuousFlag", ctypes.c_uint32),
        ("pseShiftFlag", ctypes.c_uint32),
        ("pseShiftB", ctypes.c_uint32),
        ("pseShiftS", ctypes.c_uint32),
        ("pseShiftS0", ctypes.c_uint32),
        ("selectWithByteMaskTmpMinSize", ctypes.c_uint32),
        ("actualLenQDims", ctypes.c_uint32),
        ("actualLenDims", ctypes.c_uint32),
        ("qPaddingFlag", ctypes.c_uint32),
        ("kvPaddingFlag", ctypes.c_uint32),
        ("msdIterNum", ctypes.c_uint32),
        ("l2CacheOffFlag", ctypes.c_uint32),
        ("antiquantPerTensorFlag", ctypes.c_uint32),
        ("antiquantPerHeadFlag", ctypes.c_uint32),
        ("antiquantParamsInPagedAttentionFlag", ctypes.c_uint32),
        ("attenMaskFlag", ctypes.c_uint32),
        ("attenMaskBatch", ctypes.c_uint32),
        ("attenMaskQSize", ctypes.c_uint32),
        ("attenMaskSize", ctypes.c_uint32),
        ("softmaxLseFlag", ctypes.c_uint32),
        ("totalBlockNum", ctypes.c_uint32),
        ("paKvShapeType", ctypes.c_uint32),
        ("antiqSeqSize", ctypes.c_uint32),
        ("preToken", ctypes.c_int32),
        ("nextToken", ctypes.c_int32),
        ("isRowInvalid", ctypes.c_uint32),
        ("sparseMode", ctypes.c_uint32),
        ("slidingFlag", ctypes.c_uint32),
        ("windowSize", ctypes.c_int64),
    ]


class IncreFlashAttentionCoreParams(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("coreSidxEnd", ctypes.c_uint32 * 50),
        ("coreSidxEndRegbase", ctypes.c_uint32 * 66),
        ("coreSposStartRegbase", ctypes.c_uint32 * 66),
    ]


class IncreFlashAttentionSplitCoreParams(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("headSplit", ctypes.c_uint32),
        ("maskHeadStride", ctypes.c_uint32),
        ("maskBatchStride", ctypes.c_uint32),
        ("qTokens", ctypes.c_uint32),
        ("isTriu", ctypes.c_uint32),
        ("maxSeqlen", ctypes.c_uint32),
        ("totalQBlockNum", ctypes.c_uint32),
        ("seqStepQ", ctypes.c_uint32),
        ("seqStepKv", ctypes.c_uint32),
        ("startBlk", ctypes.c_uint32 * 50),
        ("endBlk", ctypes.c_uint32 * 50),
        ("startBatch", ctypes.c_uint32 * 50),
        ("endBatch", ctypes.c_uint32 * 50),
    ]


class IncreFlashAttentionSingleCoreParams(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("sInnerLoopTimes", ctypes.c_uint32),
        ("singleProcessSInnerSize", ctypes.c_uint32),
        ("singleProcessSInnerSizeTail", ctypes.c_uint32),
        ("usedCoreNum", ctypes.c_uint32),
        ("formerCoreNum", ctypes.c_uint32),
        ("blockSplitBn2Range", ctypes.c_uint32),
        ("tailSplitedBatchRange", ctypes.c_uint32),
        ("groupSplitSize", ctypes.c_uint32),
        ("s1SplitSize", ctypes.c_uint32),
    ]


class IncreFlashAttentionSingleCoreTensorSize(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("mmResUbSize", ctypes.c_uint32),
        ("bmm2ResUbSize", ctypes.c_uint32),
    ]


class IncreFlashAttentionSplitKVParams(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("s2", ctypes.c_uint32),
        ("sInnerLoopSize", ctypes.c_uint32),
        ("accumOutSize", ctypes.c_uint32),
        ("logSumExpSize", ctypes.c_uint32),
    ]


class IncreFlashAttentionTilingData(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("bmm1TilingData", TCubeTiling),
        ("bmm2TilingData", TCubeTiling),
        ("baseParams", IncreFlashAttentionBaseParams),
        ("splitKVParams", IncreFlashAttentionSplitKVParams),
        ("increFlashAttentionCoreParams", IncreFlashAttentionCoreParams),
        ("increFlashAttentionSingleCoreParams", IncreFlashAttentionSingleCoreParams),
        ("increFlashAttentionSingleCoreTensorSize", IncreFlashAttentionSingleCoreTensorSize),
        ("softmaxFlashTilingData", SoftMaxTiling),
        ("outputParams", IncreFlashAttentionInitOutputParams),
    ]


class IncreFlashAttentionTilingDataPrefix(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("base", IncreFlashAttentionTilingData),
        ("prefixAttenOutOffset", ctypes.c_uint64),
        ("userPromptAttenOutOffset", ctypes.c_uint64),
        ("tmpLseOffset", ctypes.c_uint64),
        ("prefixLen", ctypes.c_uint64),
        ("formerCoreNum", ctypes.c_uint32),
        ("blockSplitBn2Range", ctypes.c_uint32),
        ("tailSplitedBatchRange", ctypes.c_uint32),
        ("usedCoreNum", ctypes.c_uint32),
        ("batchSizeQ", ctypes.c_uint32),
    ]


class IncreFlashAttentionTilingDataV2(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("tilingBase", IncreFlashAttentionTilingData),
        ("tilingPrefix", IncreFlashAttentionTilingDataPrefix),
    ]


class IncreFlashAttentionTilingAtbDataV2(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("tilingBase", IncreFlashAttentionBaseParams),
        ("tilingPerCore", IncreFlashAttentionSplitCoreParams),
    ]


def tiling_to_device(tiling: ctypes.Structure, device: str) -> torch.Tensor:
    raw = ctypes.string_at(ctypes.addressof(tiling), ctypes.sizeof(tiling))
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).to(device=device)


if __name__ == "__main__":
    print("sizeof IncreFlashAttentionTilingData:", ctypes.sizeof(IncreFlashAttentionTilingData))
    print("sizeof IncreFlashAttentionTilingDataPrefix:", ctypes.sizeof(IncreFlashAttentionTilingDataPrefix))
    print("sizeof IncreFlashAttentionTilingDataV2:", ctypes.sizeof(IncreFlashAttentionTilingDataV2))
    print("sizeof IncreFlashAttentionTilingAtbDataV2:", ctypes.sizeof(IncreFlashAttentionTilingAtbDataV2))
