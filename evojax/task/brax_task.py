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

import sys
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

try:
    from brax.envs import create
    from brax.envs import State as BraxState
except ModuleNotFoundError:
    print('You need to install brax for Brax tasks:')
    print('  pip install git+https://github.com/google/brax.git@main')
    sys.exit(1)


@dataclass
class State(TaskState):
    state: BraxState
    obs: jnp.ndarray


class BraxTask(VectorizedTask):
    """Tasks from the Brax simulator."""

    def __init__(self,
                 env_name: str,
                 max_steps: int = 1000,
                 legacy_spring: bool = True,
                 test: bool = False):
        self.max_steps = max_steps
        self.test = test
        brax_env = create(
            env_name=env_name,
            episode_length=max_steps,
            legacy_spring=legacy_spring,
        )
        self.obs_shape = tuple([brax_env.observation_size, ])
        self.act_shape = tuple([brax_env.action_size, ])

        def reset_fn(key):
            state = brax_env.reset(key)
            return State(state=state, obs=state.obs)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            state = brax_env.step(state.state, action)
            return State(state=state, obs=state.obs), state.reward, state.done

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
