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

#pragma once

#include "parallel_args.h"
#include "process_group.h"

namespace xllm {

// Forward declaration
namespace runtime {
struct Options;
}

namespace parallel_state {

struct GatherAsyncCtx {
  torch::Tensor input;
  torch::Tensor stacked;
  c10::intrusive_ptr<c10d::Work> work;
  std::vector<int32_t> token_num_list;
};

struct ReduceAsyncCtx {
  torch::Tensor tensor;
  c10::intrusive_ptr<c10d::Work> work;
};

std::optional<ParallelArgs> get_dp_attn_parallel_args(
    const ParallelArgs& parallel_args);

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     int32_t dim = -1);

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     const std::vector<int32_t>& token_num_list);

GatherAsyncCtx launch_gather(const torch::Tensor& input,
                             ProcessGroup* process_group,
                             const std::vector<int32_t>& token_num_list);

torch::Tensor finish_gather(GatherAsyncCtx ctx);

ReduceAsyncCtx launch_reduce(torch::Tensor input, ProcessGroup* process_group);

torch::Tensor finish_reduce(ReduceAsyncCtx ctx);

torch::Tensor all_gather_interleaved(const torch::Tensor& input,
                                     ProcessGroup* process_group);

torch::Tensor reduce(torch::Tensor& input, ProcessGroup* process_group);

torch::Tensor reduce_scatter(const torch::Tensor& input,
                             ProcessGroup* process_group);

// FlashComm1 primitive: reduce-scatter along dim0 that KEEPS the per-rank chunk
// padded to a uniform size (ceil(N / world_size)). Unlike reduce_scatter()
// above, it does NOT trim the trailing padding on boundary ranks, so every rank
// holds an identically-shaped token shard. The padding is removed later by
// all_gather_dim0_unpad() at the sequence-parallel boundary.
torch::Tensor reduce_scatter_padded_dim0(const torch::Tensor& input,
                                         ProcessGroup* process_group);

// FlashComm1 primitive: slice this rank's dim0 shard of a replicated tensor,
// padding dim0 up to a multiple of world_size first. Used to token-shard the
// (replicated) embedding output at the start of a FlashComm1 forward. No
// communication is performed.
torch::Tensor shard_dim0_padded(const torch::Tensor& input,
                                int32_t rank,
                                int32_t world_size);

// Launch/finish form of the lossless dim0 all-gather. The split API lets the
// caller overlap independent local work (for example MoE routing) with HCCL.
GatherAsyncCtx launch_all_gather_dim0(const torch::Tensor& input,
                                      ProcessGroup* process_group);

torch::Tensor finish_all_gather_dim0_unpad(GatherAsyncCtx ctx,
                                           int64_t original_num_tokens);

// FlashComm1 primitive: all-gather dim0-sharded tensors back to the full token
// dimension and slice off the padding introduced by reduce_scatter_padded_dim0.
// `original_num_tokens < 0` keeps the full padded length (no unpad).
torch::Tensor all_gather_dim0_unpad(const torch::Tensor& input,
                                    ProcessGroup* process_group,
                                    int64_t original_num_tokens);

// FlashComm1 primitive: quantized variant of all_gather_dim0_unpad. When the
// input is a floating tensor, per-token symmetric int8 quantization is applied
// before the all-gather so only int8 payload (plus small fp32 per-token scales)
// crosses the wire, roughly halving the gather volume versus bf16/fp16. The
// gathered shards are dequantized back to the input dtype. This trades ~int8
// activation rounding for communication bandwidth and is only reached when
// --enable_flashcomm1_quant_allgather is set. Falls back to the lossless
// all_gather_dim0_unpad for non-floating inputs or a single-rank group.
torch::Tensor all_gather_dim0_unpad_quant(const torch::Tensor& input,
                                          ProcessGroup* process_group,
                                          int64_t original_num_tokens);

torch::Tensor scatter(torch::Tensor input,
                      ProcessGroup* process_group,
                      int dim = -1);

std::function<torch::Tensor()> all_to_all_4D(const torch::Tensor& input_,
                                             int32_t scatter_idx,
                                             int32_t gather_idx,
                                             bool is_sync,
                                             ProcessGroup* pg);

// Create a process group where each process has a single device
// devices: list of devices to create process groups on.
std::vector<std::unique_ptr<ProcessGroup>> create_npu_process_groups(
    const std::vector<torch::Device>& devices);

// Create process groups for local (single-node) scenarios
// Supports GPU (CUDA/MLU) and NPU, including single-device case
// Parse port from options.master_node_addr() to support multiple instances
std::vector<std::unique_ptr<ProcessGroup>> create_local_process_groups(
    const std::vector<torch::Device>& devices,
    const runtime::Options& options);

}  // namespace parallel_state
}  // namespace xllm
