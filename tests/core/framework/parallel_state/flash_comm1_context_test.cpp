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

#include <gtest/gtest.h>

#include "core/framework/parallel_state/flash_comm1_context.h"
#include "core/framework/parallel_state/parallel_state.h"

namespace xllm {
namespace parallel_state {
namespace test {

TEST(FlashComm1ContextTest, KeepsPerformanceCapabilitiesIndependent) {
  EXPECT_FALSE(flash_comm1_active());

  {
    FlashComm1Guard guard(/*enabled=*/true,
                          /*num_tokens=*/7,
                          /*tp_rank=*/1,
                          /*tp_world_size=*/4,
                          /*tp_group=*/nullptr,
                          /*mmrs_enabled=*/true,
                          /*quant_allgather_enabled=*/false,
                          /*router_sp_enabled=*/true);
    const FlashComm1Context& context = current_flash_comm1_context();
    EXPECT_TRUE(context.enabled);
    EXPECT_TRUE(context.mmrs_enabled);
    EXPECT_FALSE(context.quant_allgather_enabled);
    EXPECT_TRUE(context.router_sp_enabled);
    EXPECT_EQ(context.num_tokens, 7);
    EXPECT_EQ(context.tp_rank, 1);
    EXPECT_EQ(context.tp_world_size, 4);
  }

  EXPECT_FALSE(flash_comm1_active());
}

TEST(FlashComm1ContextTest, RestoresNestedContext) {
  FlashComm1Guard outer(/*enabled=*/true,
                        /*num_tokens=*/8,
                        /*tp_rank=*/0,
                        /*tp_world_size=*/2,
                        /*tp_group=*/nullptr,
                        /*mmrs_enabled=*/false,
                        /*quant_allgather_enabled=*/true,
                        /*router_sp_enabled=*/false);
  EXPECT_TRUE(current_flash_comm1_context().quant_allgather_enabled);

  {
    FlashComm1Guard inner(/*enabled=*/false,
                          /*num_tokens=*/8,
                          /*tp_rank=*/0,
                          /*tp_world_size=*/2,
                          /*tp_group=*/nullptr,
                          /*mmrs_enabled=*/true,
                          /*quant_allgather_enabled=*/true);
    EXPECT_FALSE(flash_comm1_active());
    EXPECT_FALSE(current_flash_comm1_context().mmrs_enabled);
    EXPECT_FALSE(current_flash_comm1_context().quant_allgather_enabled);
  }

  EXPECT_TRUE(flash_comm1_active());
  EXPECT_FALSE(current_flash_comm1_context().mmrs_enabled);
  EXPECT_TRUE(current_flash_comm1_context().quant_allgather_enabled);
}

TEST(FlashComm1ContextTest, ShardsRestoredLocalDpTokensAcrossTpRanks) {
  const torch::Tensor local_dp_tokens =
      torch::arange(10, torch::TensorOptions().dtype(torch::kFloat32))
          .reshape({10, 1});

  const torch::Tensor middle_shard =
      shard_dim0_padded(local_dp_tokens, /*rank=*/1, /*world_size=*/4);
  const torch::Tensor expected_middle =
      torch::tensor({3.0F, 4.0F, 5.0F}).reshape({3, 1});
  EXPECT_TRUE(torch::equal(middle_shard, expected_middle));

  const torch::Tensor boundary_shard =
      shard_dim0_padded(local_dp_tokens, /*rank=*/3, /*world_size=*/4);
  const torch::Tensor expected_boundary =
      torch::tensor({9.0F, 0.0F, 0.0F}).reshape({3, 1});
  EXPECT_TRUE(torch::equal(boundary_shard, expected_boundary));
}

}  // namespace test
}  // namespace parallel_state
}  // namespace xllm
