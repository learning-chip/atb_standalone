#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ASCEND_TOOLKIT_HOME:?Set ASCEND_TOOLKIT_HOME to your CANN toolkit root}"

g++ -std=c++17 -O2 \
  -I"${SCRIPT_DIR}" \
  -I"${ASCEND_TOOLKIT_HOME}/include" \
  -I"${ASCEND_TOOLKIT_HOME}/aarch64-linux/asc/include" \
  -I"${ASCEND_TOOLKIT_HOME}/aarch64-linux/asc/include/utils" \
  -I"${ASCEND_TOOLKIT_HOME}/aarch64-linux/ascendc/include" \
  "${SCRIPT_DIR}/gen_incre_flash_tiling.cpp" \
  "${ASCEND_TOOLKIT_HOME}/lib64/libtiling_api.a" \
  -L"${ASCEND_TOOLKIT_HOME}/lib64" \
  -lgraph -lgraph_base -lregister -lplatform -lascendcl -lc_sec -lpthread -ldl \
  -o "${SCRIPT_DIR}/gen_incre_flash_tiling"
