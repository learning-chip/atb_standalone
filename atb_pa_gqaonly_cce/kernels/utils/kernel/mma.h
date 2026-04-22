/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INCLUDE_MMA_H
#define INCLUDE_MMA_H

#include "hardware.h"
#include "mem.h"
#include "kernel_operator.h"

// Primary template (no-op / unsupported configuration).
template <ArchType ArchTag, typename ElementA, typename ElementB, typename AccDTypeC, bool IsTransposeA = false>
struct mmad {
    __aicore__ mmad(LocalTensorView<AccDTypeC> l0cTensor,
                    LocalTensorView<ElementA>  l0aTensor,
                    LocalTensorView<ElementB>  l0bTensor,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool     initC,
                    uint8_t  unitFlag = 0) {}

    __aicore__ mmad(LocalTensorView<AccDTypeC> l0cTensor,
                    LocalTensorView<ElementA>  l0aTensor,
                    LocalTensorView<ElementB>  l0bTensor,
                    uint64_t biasBt,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool     initC,
                    uint8_t  unitFlag = 0) {}
};

// Partial specialization for IsTransposeA = false.
template <ArchType ArchTag, typename AccDTypeC, typename ElementA, typename ElementB>
struct mmad<ArchTag, ElementA, ElementB, AccDTypeC, false> {

    // Constructor without bias.
    __aicore__ mmad(LocalTensorView<AccDTypeC> l0cTensor,
                    LocalTensorView<ElementA>  l0aTensor,
                    LocalTensorView<ElementB>  l0bTensor,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool     initC,
                    uint8_t  unitFlag = 0)
    {
        mad((__cc__ AccDTypeC*)l0cTensor.GetPhyAddr(),
            (__ca__ ElementA*)l0aTensor.GetPhyAddr(),
            (__cb__ ElementB*)l0bTensor.GetPhyAddr(),
            mTileActual,   // m
            kPartActual,   // k
            nTileActual,   // n
            unitFlag,
            false,         // kDirectionAlign
            false,         // cmatrixSource
            initC);        // cmatrixInitVal
    }

    // Constructor with bias address.
    __aicore__ mmad(LocalTensorView<AccDTypeC> l0cTensor,
                    LocalTensorView<ElementA>  l0aTensor,
                    LocalTensorView<ElementB>  l0bTensor,
                    uint64_t biasBt,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool     initC,
                    uint8_t  unitFlag = 0)
    {
        mad((__cc__ AccDTypeC*)l0cTensor.GetPhyAddr(),
            (__ca__ ElementA*)l0aTensor.GetPhyAddr(),
            (__cb__ ElementB*)l0bTensor.GetPhyAddr(),
            biasBt,
            mTileActual,   // m
            kPartActual,   // k
            nTileActual,   // n
            unitFlag,
            false,         // kDirectionAlign
            true,          // cmatrixSource = true (bias present)
            false);        // cmatrixInitVal = false (bias path)
    }
};

#endif
