/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INCLUDE_MEM_H
#define INCLUDE_MEM_H

#include "hardware.h"

enum class BufferType { ASCEND_UB, ASCEND_CB, ASCEND_L0A, ASCEND_L0B, ASCEND_L0C, ASCEND_MAX };

template <typename T>
struct RawTensorView {
    uint64_t addr{0};
    __aicore__ RawTensorView() = default;
    __aicore__ explicit RawTensorView(uint64_t in) : addr(in) {}
    __aicore__ inline uint64_t GetPhyAddr() const { return addr; }
    __aicore__ inline RawTensorView<T> operator[](uint64_t offset) const
    {
        return RawTensorView<T>(addr + offset * sizeof(T));
    }
    template <typename U>
    __aicore__ inline RawTensorView<U> ReinterpretCast() const
    {
        return RawTensorView<U>(addr);
    }
};

template <BufferType BufferType_, typename DstDataType>
__aicore__ inline RawTensorView<DstDataType> MakeRawBuffer(const uint32_t offset)
{
    if constexpr (BufferType_ == BufferType::ASCEND_UB) {
        return RawTensorView<DstDataType>((uint64_t)reinterpret_cast<__ubuf__ uint8_t *>((uintptr_t)offset));
    } else if constexpr (BufferType_ == BufferType::ASCEND_CB) {
        return RawTensorView<DstDataType>((uint64_t)reinterpret_cast<__cbuf__ uint8_t *>((uintptr_t)offset));
    } else if constexpr (BufferType_ == BufferType::ASCEND_L0A) {
        return RawTensorView<DstDataType>((uint64_t)reinterpret_cast<__ca__ uint8_t *>((uintptr_t)offset));
    } else if constexpr (BufferType_ == BufferType::ASCEND_L0B) {
        return RawTensorView<DstDataType>((uint64_t)reinterpret_cast<__cb__ uint8_t *>((uintptr_t)offset));
    } else if constexpr (BufferType_ == BufferType::ASCEND_L0C) {
        return RawTensorView<DstDataType>((uint64_t)reinterpret_cast<__cc__ uint8_t *>((uintptr_t)offset));
    }
    return RawTensorView<DstDataType>(0);
}

template <ArchType ArchTag>
struct AsdopsBuffer {
public:
    __aicore__ AsdopsBuffer() {};

    template <BufferType BufferType_, typename DstDataType = half>
    __aicore__ RawTensorView<DstDataType> GetBuffer(const uint32_t offset) const
    {
        return MakeRawBuffer<BufferType_, DstDataType>(offset);
    }
};

#endif