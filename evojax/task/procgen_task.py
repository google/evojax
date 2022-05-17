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
import multiprocessing as mp
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState


@jax.jit
def to_jnp_array(data: np.ndarray):
    return jnp.array(data, dtype=jnp.float32)


@dataclass
class State(TaskState):
    obs: jnp.ndarray


class ProcgenTask(VectorizedTask):
    """Wrapper of the procgen env."""

    def __init__(self,
                 env_name: str,
                 num_envs: int = 1024,
                 n_repeats: int = 1,
                 num_levels: int = 200,
                 distribution_mode: str = 'easy',
                 max_steps: int = 1000,
                 num_threads: int = -1,
                 test: bool = False):
        self.max_steps = max_steps
        self.test = test

        try:
            from procgen import ProcgenEnv
        except ModuleNotFoundError:
            print('You need to install procgen for this task:')
            print('  pip install procgen')
            sys.exit(1)

        procgen_venv = ProcgenEnv(
            num_envs=4,
            env_name=env_name,
            num_levels=0 if test else num_levels,
            start_level=num_levels if test else 0,
            distribution_mode=distribution_mode,
            num_threads=num_threads if num_threads > 0 else mp.cpu_count(),
        )
        obs_space = procgen_venv.observation_space['rgb']
        act_space = procgen_venv.action_space
        self.procgen_venv = procgen_venv
        self.obs_shape = obs_space.shape
        self.act_shape = act_space.shape
        self.num_acts = act_space.n
        assert num_envs % n_repeats == 0
        pop_size = num_envs // n_repeats

        def reset_fn(key):
            rand_seed = int(jax.random.randint(
                key[0], shape=(), minval=0, maxval=2 ** 31 - 1))
            self.procgen_venv = ProcgenEnv(
                num_envs=num_envs,
                env_name=env_name,
                num_levels=0 if test else num_levels,
                start_level=num_levels if test else 0,
                distribution_mode=distribution_mode,
                num_threads=num_threads if num_threads > 0 else mp.cpu_count(),
                rand_seed=rand_seed,
                use_backgrounds=True,
            )
            if not self.test:
                state = self.procgen_venv.env.get_state()
                assert len(state) == num_envs, '{} vs {}'.format(
                    len(state), num_envs)
                ss = [x for _ in range(pop_size) for x in state[:n_repeats]]
                self.procgen_venv.env.set_state(ss)

            obs, _, _, _ = self.procgen_venv.step(np.zeros(num_envs))
            return State(obs=to_jnp_array(obs['rgb']))
        self._reset_fn = reset_fn

        def step_fn(state, action):
            action_np = np.array(action)
            obs, reward, done, info = self.procgen_venv.step(action_np)
            return (State(obs=to_jnp_array(obs['rgb'])),
                    to_jnp_array(reward), to_jnp_array(done))
        self._step_fn = step_fn

    def reset(self, key: jnp.ndarray) -> TaskState:
        return self._reset_fn(key)

    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
