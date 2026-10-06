// Exercise the patched policy with real ggml graphs without requiring a GPU.
// Qualification additionally checks actual Vulkan generation and ACE-Step.
#include "engine/community_models/minimax_music3/vulkan_precision.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

ggml_prec precision(const ggml_tensor * tensor) {
    ggml_prec value;
    static_assert(sizeof(value) == sizeof(tensor->op_params[0]));
    std::memcpy(&value, tensor->op_params, sizeof(value));
    return value;
}

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_policy(engine::core::BackendType backend, ggml_type weight_type) {
    ggml_init_params params{1 << 20, nullptr, true};
    ggml_context * context = ggml_init(params);
    require(context != nullptr, "graph context allocation failed");
    ggml_tensor * weight = ggml_new_tensor_2d(context, weight_type, 64, 32);
    ggml_tensor * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 64, 3);
    ggml_tensor * first = ggml_mul_mat(context, weight, input);
    ggml_tensor * second = ggml_mul_mat(context, weight, input);
    ggml_tensor * preserved = ggml_mul_mat(context, weight, input);
    ggml_mul_mat_set_prec(preserved, GGML_PREC_F32);
    ggml_tensor * output = ggml_add(context, ggml_add(context, first, second), preserved);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);

    int32_t original_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    std::memcpy(original_params, output->op_params, sizeof(original_params));
    const bool vulkan = backend == engine::core::BackendType::Vulkan;
    const ggml_prec expected = vulkan ? GGML_PREC_F32 : GGML_PREC_DEFAULT;
    require(precision(first) == GGML_PREC_DEFAULT, "unexpected initial matmul precision");
    engine::models::minimax_music3::apply_vulkan_matmul_precision(backend, graph);
    require(precision(first) == expected, "first matmul selected wrong precision");
    require(precision(second) == expected, "second matmul selected wrong precision");
    require(precision(preserved) == GGML_PREC_F32, "existing explicit precision changed");
    require(std::memcmp(original_params, output->op_params, sizeof(original_params)) == 0,
            "non-matmul operation parameters changed");
    require(engine::models::minimax_music3::vulkan_matmul_precision(backend) == expected,
            "global LM precision policy disagrees with graph policy");
    engine::models::minimax_music3::apply_vulkan_matmul_precision(backend, graph);
    require(precision(first) == expected, "reapplying policy changed precision");
    ggml_free(context);
}

}  // namespace

int main() {
    for (const auto backend : {
            engine::core::BackendType::Cpu, engine::core::BackendType::Metal,
            engine::core::BackendType::Cuda, engine::core::BackendType::Hip,
            engine::core::BackendType::Vulkan}) {
        for (const auto weight_type : {GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
            check_policy(backend, weight_type);
        }
    }
    std::cout << "MiniMax Vulkan precision policy: 15 graph cases passed\n";
}
