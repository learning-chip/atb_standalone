/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INCLUDE_SIMD_H
#define INCLUDE_SIMD_H

#include <type_traits>

#include "hardware.h"
#include "kernel_operator.h"

constexpr Order_t ORDER_ONLY_VALUE = ONLY_VALUE;

template <ArchType ArchTag, typename DTypeIn, typename DTypeOut>
__aicore__ inline void conv_v(__ubuf__ DTypeOut* dst,
                              __ubuf__ DTypeIn* src,
                              uint8_t repeat,
                              uint16_t dstBlockStride,
                              uint16_t srcBlockStride,
                              uint16_t dstRepeatStride,
                              uint16_t srcRepeatStride)
{
    if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, __bf16>::value) {
        vconv_f322bf16r((__ubuf__ __bf16*)dst, (__ubuf__ float*)src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, half>::value) {
        vconv_f322f16((__ubuf__ half*)dst, (__ubuf__ float*)src, repeat, dstBlockStride, srcBlockStride,
                      dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, half>::value && std::is_same<DTypeOut, float>::value) {
        vconv_f162f32((__ubuf__ float*)dst, (__ubuf__ half*)src, repeat, dstBlockStride, srcBlockStride,
                      dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, __bf16>::value && std::is_same<DTypeOut, float>::value) {
        vconv_bf162f32((__ubuf__ float*)dst, (__ubuf__ __bf16*)src, repeat, dstBlockStride, srcBlockStride,
                       dstRepeatStride, srcRepeatStride);
    } else {
        static_assert(!std::is_same<DTypeIn, DTypeIn>::value, "Unsupported conv_v dtype combination.");
    }
}

#endif
