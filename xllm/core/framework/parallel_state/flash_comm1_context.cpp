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

#include "core/framework/parallel_state/flash_comm1_context.h"

#include <algorithm>

namespace xllm {
namespace parallel_state {

namespace {
// Active FlashComm1 context for the calling thread. Each worker thread runs its
// own model forward(), so thread_local isolation prevents one rank/stream's
// context from leaking into another.
thread_local FlashComm1Context t_flash_comm1_context{};
}  // namespace

const FlashComm1Context& current_flash_comm1_context() {
  return t_flash_comm1_context;
}

bool flash_comm1_active() {
  return t_flash_comm1_context.enabled;
}

FlashComm1Guard::FlashComm1Guard(bool enabled,
                                 int64_t num_tokens,
                                 int32_t tp_rank,
                                 int32_t tp_world_size,
                                 ProcessGroup* tp_group,
                                 bool mmrs_enabled,
                                 bool quant_allgather_enabled,
                                 bool router_sp_enabled)
    : previous_(t_flash_comm1_context) {
  FlashComm1Context ctx;
  ctx.enabled = enabled && tp_world_size > 1 && num_tokens > 0;
  ctx.num_tokens = num_tokens;
  ctx.tp_rank = tp_rank;
  ctx.tp_world_size = std::max<int32_t>(1, tp_world_size);
  ctx.tp_group = tp_group;
  ctx.mmrs_enabled = ctx.enabled && mmrs_enabled;
  ctx.quant_allgather_enabled = ctx.enabled && quant_allgather_enabled;
  ctx.router_sp_enabled = ctx.enabled && router_sp_enabled;
  t_flash_comm1_context = ctx;
}

FlashComm1Guard::~FlashComm1Guard() {
  t_flash_comm1_context = previous_;
}

}  // namespace parallel_state
}  // namespace xllm
