#include "../../ops-transformer/attention/chunk_gated_delta_rule/op_kernel/chunk_gated_delta_rule.h"
#include "../../ops-transformer/attention/chunk_gated_delta_rule/op_kernel/chunk_gated_delta_rule_tiling_data.h"

using namespace AscendC;
using namespace matmul;
using namespace ChunkGatedDeltaRule;

extern "C" __global__ __aicore__ void chunk_gated_delta_rule(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR initialState, GM_ADDR seqlens, GM_ADDR gOptional,
    GM_ADDR out, GM_ADDR finalState, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    // Standalone debug path: force vector-core execution only.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    CGDR<bfloat16_t, float> op(&pipe, &tilingData);
    CGDRInitParams initParams{
        query, key, value, beta, initialState, seqlens, gOptional,
        out, finalState};
    op.Init(initParams, user);
    op.Process();
}

