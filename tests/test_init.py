# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class TestTask:
    def test_cartpole(self):
        from evojax.task.cartpole import CartPoleSwingUp
        _ = CartPoleSwingUp()
        assert True

    def test_seq2seq(self):
        from evojax.task.seq2seq import Seq2seqTask
        _ = Seq2seqTask()
        assert True

    def test_waterworld(self):
        from evojax.task.waterworld import WaterWorld
        _ = WaterWorld()
        assert True

    def test_waterworld_ma(self):
        from evojax.task.ma_waterworld import MultiAgentWaterWorld
        _ = MultiAgentWaterWorld()
        assert True

    def test_flocing(self):
        from evojax.task.flocking import FlockingTask
        _ = FlockingTask()
        assert True


class TestPolicy:
    def test_seq2seq(self):
        from evojax.policy import Seq2seqPolicy
        _ = Seq2seqPolicy()
        assert True

    def test_mlp(self):
        from evojax.policy import MLPPolicy
        _ = MLPPolicy(input_dim=16, hidden_dims=(16, 16), output_dim=16)
        assert True

    def test_mlp_pi(self):
        from evojax.policy import PermutationInvariantPolicy
        _ = PermutationInvariantPolicy(act_dim=16, hidden_dim=16)
        assert True

    def test_convnet(self):
        from evojax.policy import ConvNetPolicy
        _ = ConvNetPolicy()
        assert True


class TestAlgo:
    def test_pgpe(self):
        from evojax.algo import PGPE
        _ = PGPE(pop_size=16, param_size=16)
        assert True

    def test_cma_es_jax(self):
        from evojax.algo import CMA_ES_JAX
        _ = CMA_ES_JAX(pop_size=16, param_size=16)
        assert True
