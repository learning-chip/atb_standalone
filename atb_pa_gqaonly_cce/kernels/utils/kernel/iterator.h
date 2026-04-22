/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INCLUDE_ITERTOR_H
#define INCLUDE_ITERTOR_H

#include "common_func.h"
#include "hardware.h"
#include "mem.h"

template <ArchType ArchTag, typename DataType, DataFormat FormatInGM, DataFormat FormatInL1>
struct gm_to_l1 {
    __aicore__ gm_to_l1(LocalTensorView<DataType> l1Tensor,
                        RawAddrTensorView<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal) {};
};

template <ArchType ArchTag, typename DataType, bool IsTransPose, DataFormat DFmtIn, DataFormat DFmtOut>
struct l1_to_l0_a {
    __aicore__ l1_to_l0_a(LocalTensorView<DataType> l0Tensor,
                          LocalTensorView<DataType> l1Tensor,
                          uint32_t mTileCeil,
                          uint32_t kPartCeil,
                          uint32_t mSrcStride,
                          uint32_t kSrcStride,
                          uint32_t mDstStride,
                          uint32_t kDstStride) {};
};

template <ArchType ArchTag, typename DataType, bool IsTransPose, DataFormat DFmtIn, DataFormat DFmtOut>
struct l1_to_l0_b {
    __aicore__ l1_to_l0_b(LocalTensorView<DataType> l0Tensor,
                          LocalTensorView<DataType> l1Tensor,
                          uint32_t nTileCeil,
                          uint32_t kPartCeil,
                          uint32_t nSrcStride,
                          uint32_t kSrcStride,
                          uint32_t nDstStride,
                          uint32_t kDstStride) {};
};

template <ArchType ArchTag, DataFormat OutFormatType, typename OutDataType, typename L0CDataType>
struct l0c_to_gm {
    __aicore__ l0c_to_gm(RawAddrTensorView<OutDataType> gmTensor,
                         LocalTensorView<L0CDataType> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride,
                         uint8_t  unitFlag = 0) {};
};

#include "iterators/gm_to_l1_iterator.inc"
#include "iterators/gm_to_ub_iterator.inc"
#include "iterators/l0c_to_gm_iterator.inc"
#include "iterators/l1_to_l0_iterator.inc"
#endif
