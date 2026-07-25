/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch_npu/csrc/aten/CustomFunctions.h>

#include <string>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "npu_ops_api.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#endif

namespace xllm::kernel::npu {

// ACL FRACTAL_NZ format enum value. The fused MatmulAllReduce requires the
// weight in FRACTAL_NZ on some SoCs (e.g. Ascend 910_93): the ND-weight opType
// is not registered there, so an ND weight triggers
// "socVersion does not support opType [MatmulAllReduce]".
constexpr int64_t kAclFormatFractalNz = 29;

torch::Tensor to_fractal_nz(const torch::Tensor& tensor) {
  auto contiguous = tensor.contiguous();
#ifdef TORCH_HIGHER_THAN_PTA6
  return at_npu::native::npu_format_cast(contiguous, kAclFormatFractalNz);
#else
  return at_npu::native::NPUNativeFunctions::npu_format_cast(contiguous,
                                                             kAclFormatFractalNz);
#endif
}

bool has_mm_all_reduce() {
  // torch_npu's npu_mm_all_reduce_base has dispatched to a few different aclnn
  // symbols across CANN versions; probe the known workspace entry points and
  // treat the op as available if any of them resolves.
  static const bool available = [] {
    for (const char* symbol : {"aclnnMatmulAllReduceV3GetWorkspaceSize",
                               "aclnnMatmulAllReduceV2GetWorkspaceSize",
                               "aclnnMatmulAllReduceGetWorkspaceSize",
                               "aclnnInplaceMatmulAllReduceGetWorkspaceSize"}) {
      if (aclnn::detail::get_op_api_func_addr(symbol) != nullptr) {
        return true;
      }
    }
    return false;
  }();
  return available;
}

torch::Tensor mm_all_reduce(const torch::Tensor& x1,
                            const torch::Tensor& x2,
                            const std::string& hcom,
                            const std::string& reduce_op,
                            const c10::optional<torch::Tensor>& bias,
                            const c10::optional<torch::Tensor>& dequant_scale,
                            const c10::optional<torch::Tensor>& pertoken_scale,
                            int64_t comm_turn) {
  const c10::optional<at::Tensor> none = c10::nullopt;
  return at_npu::native::custom_ops::npu_mm_all_reduce_base(
      x1,
      x2,
      hcom,
      reduce_op,
      bias,
      /*antiquant_scale=*/none,
      /*antiquant_offset=*/none,
      /*x3=*/none,
      dequant_scale,
      pertoken_scale,
      /*comm_quant_scale_1=*/none,
      /*comm_quant_scale_2=*/none,
      /*antiquant_group_size=*/0,
      comm_turn);
}

}  // namespace xllm::kernel::npu
