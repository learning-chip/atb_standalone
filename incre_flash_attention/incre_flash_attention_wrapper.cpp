// Host wrapper compiled with bisheng (same pattern as cann_ops_standalone/chunk_gdn).
#include <cstdint>

#include "op_kernel/incre_flash_attention.cpp"

extern "C" void call_incre_flash_attention(
    uint32_t blockDim,
    void *stream,
    uint8_t *query,
    uint8_t *key,
    uint8_t *value,
    uint8_t *pseShift,
    uint8_t *attenMask,
    uint8_t *actualSeqLengths,
    uint8_t *deqScale1,
    uint8_t *quantScale1,
    uint8_t *deqScale2,
    uint8_t *quantScale2,
    uint8_t *quantOffset2,
    uint8_t *antiquantScale,
    uint8_t *antiquantOffset,
    uint8_t *blocktable,
    uint8_t *kvPaddingSize,
    uint8_t *attentionOut,
    uint8_t *workspace,
    uint8_t *tilingGM) {
    incre_flash_attention<<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t *)query,
        (__gm__ uint8_t *)key,
        (__gm__ uint8_t *)value,
        (__gm__ uint8_t *)pseShift,
        (__gm__ uint8_t *)attenMask,
        (__gm__ uint8_t *)actualSeqLengths,
        (__gm__ uint8_t *)deqScale1,
        (__gm__ uint8_t *)quantScale1,
        (__gm__ uint8_t *)deqScale2,
        (__gm__ uint8_t *)quantScale2,
        (__gm__ uint8_t *)quantOffset2,
        (__gm__ uint8_t *)antiquantScale,
        (__gm__ uint8_t *)antiquantOffset,
        (__gm__ uint8_t *)blocktable,
        (__gm__ uint8_t *)kvPaddingSize,
        (__gm__ uint8_t *)attentionOut,
        (__gm__ uint8_t *)workspace,
        (__gm__ uint8_t *)tilingGM);
}
