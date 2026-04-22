/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../iterator.h"

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, float, float> {
    __aicore__ l0c_to_gm(RawAddrTensorView<float> gmTensor,
                         LocalTensorView<float> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride,
                         uint8_t  unitFlag = 0)
    {
        set_nd_para((uint64_t)1);
        pipe_barrier(PIPE_FIX);
        copy_matrix_cc_to_gm((__gm__ float*)gmTensor.GetPhyAddr(),
                             (__cc__ float*)l0cTensor.GetPhyAddr(),
                             0, nTileActual, mTileActual, dstStride, srcStride, unitFlag,
                             QuantMode_t::NoQuant, 0, false, true);
    };
};
