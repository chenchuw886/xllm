/* Copyright 2026 The xLLM Authors.

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

#include <cstdint>

namespace xllm {

class ProcessGroup;

namespace parallel_state {

// Per-forward FlashComm1 (sequence-parallel) state.
//
// FlashComm1 turns the RowParallelLinear tail all-reduce into a padded
// reduce-scatter(dim0), so the residual stream stays token-sharded across the
// TP group and the norm / gate work runs on 1/tp_world_size tokens. The
// full token dimension is restored with an all-gather at the sequence-parallel
// boundaries (attention QKV input, final norm before logits).
//
// Layers such as RowParallelLinearImpl::forward() do not receive
// ModelInputParams, so the active context is published via a thread_local set
// by FlashComm1Guard at the top of each model forward().
struct FlashComm1Context {
  bool enabled = false;
  // Original number of tokens in this forward.
  int64_t num_tokens = 0;
  int32_t tp_rank = 0;
  int32_t tp_world_size = 1;

  // Independent performance capabilities. MMRS is lossless relative to the
  // unfused matmul + reduce-scatter path. Quantized all-gather is lossy and is
  // only applied after MoE routing has been computed on the local shard.
  bool mmrs_enabled = false;
  bool quant_allgather_enabled = false;
  bool router_sp_enabled = false;

  // TP process group used for the sequence-parallel all-gather boundaries.
  // Populated by FlashComm1Guard so layers that only see the context (e.g. the
  // DSV4 decoder layer) can reach the group without extra plumbing.
  ProcessGroup* tp_group = nullptr;

};

// Returns the FlashComm1 context active on the current thread. When no guard is
// in scope the returned context has enabled == false.
const FlashComm1Context& current_flash_comm1_context();

// Convenience helper: true when a FlashComm1 context is active AND enabled.
bool flash_comm1_active();

// RAII guard that publishes a FlashComm1 context for the duration of a model
// forward() on the calling thread and restores the previous context on scope
// exit. `enabled` should already fold in the token-count threshold and the
// tp_world_size > 1 / backend checks decided by the caller.
class FlashComm1Guard final {
 public:
  FlashComm1Guard(bool enabled,
                  int64_t num_tokens,
                  int32_t tp_rank,
                  int32_t tp_world_size,
                  ProcessGroup* tp_group,
                  bool mmrs_enabled = false,
                  bool quant_allgather_enabled = false,
                  bool router_sp_enabled = false);
  ~FlashComm1Guard();

  FlashComm1Guard(const FlashComm1Guard&) = delete;
  FlashComm1Guard& operator=(const FlashComm1Guard&) = delete;

 private:
  FlashComm1Context previous_;
};

}  // namespace parallel_state
}  // namespace xllm
