#!/usr/bin/env python3
"""Regenerate paged_attention_decoder_nd_common.cce from the AscendC reference kernel."""
from __future__ import annotations

import sys
from pathlib import Path

STRUCT = """struct LoadData2dTransposeParams {
    uint16_t startIndex{0};
    uint16_t repeatTimes{0};
    uint16_t srcStride{0};
    uint16_t dstGap{0};
    uint16_t dstFracGap{0};
};

"""

LOAD1 = """                        AscendC::LoadDataWithTranspose(
                                l0b_buf_tensor[mad_l0b_offset + l0b_load_idx * RoundUp<16>(__k) * T_BLOCK_SIZE],
                                l1kv_buf_addr_tensor[move_l1b_offset +
                                    headdim_idx * round_k * qk_round_n / group_num +
                                    l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE],
                                loadDataParams);"""

LOAD1_CCE = """                        load_cbuf_to_cb_transpose(
                                (__cb__ IN_DTYPE*)(l0b_buf_tensor[mad_l0b_offset + l0b_load_idx * RoundUp<16>(__k) * T_BLOCK_SIZE]).GetPhyAddr(),
                                (__cbuf__ IN_DTYPE*)(l1kv_buf_addr_tensor[move_l1b_offset +
                                    headdim_idx * round_k * qk_round_n / group_num +
                                    l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE]).GetPhyAddr(),
                                loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride,
                                loadDataParams.dstGap, (addr_cal_mode_t)0, loadDataParams.dstFracGap);"""

LOAD2 = """                        AscendC::LoadDataWithTranspose(
                                l0b_buf_tensor[mad_l0b_offset + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE],
                                l1kv_buf_addr_tensor[move_l1b_offset +
                                    headdim_idx * round_k * qk_round_n / group_num +
                                    l0b_load_idx * qk_round_n * T_BLOCK_SIZE],
                                loadDataParams);"""

LOAD2_CCE = """                        load_cbuf_to_cb_transpose(
                                (__cb__ IN_DTYPE*)(l0b_buf_tensor[mad_l0b_offset + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE]).GetPhyAddr(),
                                (__cbuf__ IN_DTYPE*)(l1kv_buf_addr_tensor[move_l1b_offset +
                                    headdim_idx * round_k * qk_round_n / group_num +
                                    l0b_load_idx * qk_round_n * T_BLOCK_SIZE]).GetPhyAddr(),
                                loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride,
                                loadDataParams.dstGap, (addr_cal_mode_t)0, loadDataParams.dstFracGap);"""

LOAD3 = """                            AscendC::LoadDataWithTranspose(
                                    l0b_buf_tensor[l0b_offset + l0b_load_idx * RoundUp<16>(__k) * T_BLOCK_SIZE],
                                    l1kv_buf_addr_tensor[l1kv_offset + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE],
                                    loadDataParams);"""

LOAD3_CCE = """                            load_cbuf_to_cb_transpose(
                                    (__cb__ IN_DTYPE*)(l0b_buf_tensor[l0b_offset + l0b_load_idx * RoundUp<16>(__k) * T_BLOCK_SIZE]).GetPhyAddr(),
                                    (__cbuf__ IN_DTYPE*)(l1kv_buf_addr_tensor[l1kv_offset + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE]).GetPhyAddr(),
                                    loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride,
                                    loadDataParams.dstGap, (addr_cal_mode_t)0, loadDataParams.dstFracGap);"""

LOAD4 = """                        AscendC::LoadDataWithTranspose(
                                l0b_buf_tensor[l0b_offset + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE],
                                l1kv_buf_addr_tensor[l1kv_offset + l0b_load_idx * qk_round_n * T_BLOCK_SIZE],
                                loadDataParams);"""

LOAD4_CCE = """                        load_cbuf_to_cb_transpose(
                                (__cb__ IN_DTYPE*)(l0b_buf_tensor[l0b_offset + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE]).GetPhyAddr(),
                                (__cbuf__ IN_DTYPE*)(l1kv_buf_addr_tensor[l1kv_offset + l0b_load_idx * qk_round_n * T_BLOCK_SIZE]).GetPhyAddr(),
                                loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride,
                                loadDataParams.dstGap, (addr_cal_mode_t)0, loadDataParams.dstFracGap);"""


def port(ascend_text: str) -> str:
    t = ascend_text
    t = t.replace(
        "#include \"mixkernels/pagedattention/tiling/paged_attention_tiling_dependency.h\"\n\n// define common const value",
        "#include \"mixkernels/pagedattention/tiling/paged_attention_tiling_dependency.h\"\n\n"
        + STRUCT
        + "// define common const value",
        1,
    )
    t = t.replace(
        "template <typename T>\nusing GlobalT = AscendC::GlobalTensor<T>;",
        "template <typename T>\nusing GlobalT = RawAddrTensorView<T>;",
    )
    t = t.replace(
        "template <typename T>\nusing LocalT = AscendC::LocalTensor<T>;",
        "template <typename T>\nusing LocalT = LocalTensorView<T>;",
    )
    t = t.replace("AscendC::GlobalTensor", "RawAddrTensorView")
    t = t.replace("AscendC::LocalTensor", "LocalTensorView")
    t = t.replace("AscendC::LoadData2dTransposeParams", "LoadData2dTransposeParams")
    t = t.replace(
        "cmax_v<ArchType::ASCEND_V220, float, AscendC::ReduceOrder::ORDER_ONLY_VALUE>",
        "cmax_v<ArchType::ASCEND_V220, float, ORDER_ONLY_VALUE>",
    )
    for a, b in ((LOAD1, LOAD1_CCE), (LOAD2, LOAD2_CCE), (LOAD3, LOAD3_CCE), (LOAD4, LOAD4_CCE)):
        if a not in t:
            raise RuntimeError("Expected LoadDataWithTranspose block not found — AscendC kernel changed?")
        t = t.replace(a, b, 1)
    return t


def main() -> None:
    ascend = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.write_text(port(ascend.read_text()))


if __name__ == "__main__":
    main()
