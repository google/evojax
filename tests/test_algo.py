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

class TestAlgo:
    def test_cma_es_jax_save_and_load_state(self):
        from evojax.algo import CMA_ES_JAX
        from jax import numpy as jnp
        solver = CMA_ES_JAX(pop_size=16, param_size=16)
        # one step
        _ = solver.ask()
        solver.tell(jnp.arange(16, dtype=jnp.float32))
        # one step
        state = solver.save_state()
        internal_state_0 = solver.state._asdict()
        # one step
        _ = solver.ask()
        solver.tell(-jnp.arange(16, dtype=jnp.float32))
        internal_state_1 = solver.state._asdict()
        solver.load_state(state)
        internal_state_2 = solver.state._asdict()

        keys = list(internal_state_0.keys())

        assert not all([jnp.all(internal_state_0[key] == internal_state_1[key]) for key in keys])
        assert all([jnp.all(internal_state_0[key] == internal_state_2[key]) for key in keys])
