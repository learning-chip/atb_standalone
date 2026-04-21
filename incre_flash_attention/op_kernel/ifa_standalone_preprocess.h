/**
 * Standalone bisheng build does not run the AscendC OPP template preprocessor that sets
 * ORIG_DTYPE_* / DT_* for incre_flash_attention_obp.h. On device, AscendC::DataType uses
 * enumerator names, which are not visible to #if, so every (ORIG_DTYPE_QUERY == DT_*) guard
 * becomes false and the kernel body is empty.
 *
 * Include this file after kernel_operator.h (so kernel_type.h enums are parsed first) and
 * before incre_flash_attention_obp.h. These #defines only affect preprocessor; they must
 * match the numeric order in basic_api/kernel_type.h (non-host branch).
 */
#ifndef IFA_STANDALONE_PREPROCESS_H
#define IFA_STANDALONE_PREPROCESS_H

// kernel_utils_macros.h defaults TILING_KEY_VAR to g_tilingKey; #if TILING_KEY_VAR == K in
// incre_flash_attention_obp.h then never matches at preprocess time. Pin one template for
// standalone fp16 BSH + paged KV + flash decoding (C1V2), matching incre_flash_attention_tilingkey.h.
#ifdef TILING_KEY_VAR
#undef TILING_KEY_VAR
#endif
#define TILING_KEY_VAR 10000000000300001

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY 1
#endif
#ifndef ORIG_DTYPE_KEY
#define ORIG_DTYPE_KEY 1
#endif
#ifndef ORIG_DTYPE_ATTENTION_OUT
#define ORIG_DTYPE_ATTENTION_OUT 1
#endif

#define DT_FLOAT 0
#define DT_FLOAT16 1
#define DT_INT8 2
#define DT_INT32 3
#define DT_UINT8 4
#define DT_INT16 6
#define DT_UINT16 7
#define DT_UINT32 8
#define DT_INT64 9
#define DT_UINT64 10
#define DT_DOUBLE 11
#define DT_BOOL 12
#define DT_STRING 13
#define DT_DUAL_SUB_INT8 14
#define DT_DUAL_SUB_UINT8 15
#define DT_COMPLEX64 16
#define DT_COMPLEX128 17
#define DT_QINT8 18
#define DT_QINT16 19
#define DT_QINT32 20
#define DT_QUINT8 21
#define DT_QUINT16 22
#define DT_RESOURCE 23
#define DT_STRING_REF 24
#define DT_DUAL 25
#define DT_VARIANT 26
#define DT_BF16 27
#define DT_UNDEFINED 28
#define DT_INT4 29
#define DT_UINT1 30
#define DT_INT2 31
#define DT_UINT2 32
#define DT_COMPLEX32 33
#define DT_HIFLOAT8 34
#define DT_FLOAT8_E5M2 35
#define DT_FLOAT8_E4M3FN 36
#define DT_FLOAT8_E8M0 37
#define DT_FLOAT6_E3M2 38
#define DT_FLOAT6_E2M3 39
#define DT_FLOAT4_E2M1 40
#define DT_FLOAT4_E1M2 41
#define DT_MAX 42

#endif // IFA_STANDALONE_PREPROCESS_H
