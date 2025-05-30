#include "maga_transformer/cpp/devices/DeviceBase.h"
#include "maga_transformer/cpp/devices/utils/DebugUtils.h"
#include "maga_transformer/cpp/core/BufferHelper.h"

namespace rtp_llm {

BufferPtr DeviceBase::mhaQKVGemm(const AttentionLayerParams& params) {
    const auto& input      = params.input;
    const auto& qkv_weight = params.weights.qkv_weight;
    

#if defined(__aarch64__)
    // Arm attention op only support fp32 data type
    auto qkv_gemm_params = GemmParams(input, *(qkv_weight->kernel), std::nullopt, nullptr, DataType::TYPE_FP32);
#else
    auto qkv_gemm_params = GemmParams(input, *(qkv_weight->kernel));
#endif

    auto lora_linear_params = LoraLinearParams(qkv_gemm_params, params.common.lora_input.qkv_lora_input);
    BufferPtr qkv;
    if (!params.configs.fuse_qkv_add_bias && params.weights.qkv_weight) {
        ActivationParams act_params(ActivationType::Identity,
                                    nullptr,
                                    mayGetRef(params.weights.qkv_weight->bias),
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt);
        qkv = loraLinearWithActivation(LoraLinearWithActivationParams(lora_linear_params, act_params));
    } else {
        qkv = loraLinear(LoraLinearParams(qkv_gemm_params, params.common.lora_input.qkv_lora_input)).output;
    }
    printBufferData(*qkv, "qkv");

    if (params.weights.q_norm_weight) {
        auto after_q_norm = layernormWithStride(LayernormWithStrideParams({qkv, 
                                                      *params.weights.q_norm_weight, 
                                                      params.ln_params.eps, 
                                                      params.ln_params.norm_type, 
                                                      0, 
                                                      params.configs.size_per_head * params.configs.head_num}));

        qkv = std::move(after_q_norm.output);
        printBufferData(*qkv, "qkv_after_q_norm");
    }

    if (params.weights.k_norm_weight) {
        auto after_k_norm = layernormWithStride(LayernormWithStrideParams({qkv,
                                                      *params.weights.k_norm_weight,
                                                      params.ln_params.eps,
                                                      params.ln_params.norm_type,
                                                      params.configs.size_per_head * params.configs.head_num,
                                                      params.configs.size_per_head * params.configs.kv_head_num}));

        qkv = std::move(after_k_norm.output);
        printBufferData(*qkv, "qkv_after_k_norm");
    }

    return qkv;
}
}  // namespace rtp_llm