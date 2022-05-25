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
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState
from evojax.task.base import BDExtractor

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
    feet_contact: jnp.ndarray


class BraxTask(VectorizedTask):
    """Tasks from the Brax simulator."""

    def __init__(self,
                 env_name: str,
                 max_steps: int = 1000,
                 legacy_spring: bool = True,
                 bd_extractor: Optional[BDExtractor] = None,
                 test: bool = False):
        self.max_steps = max_steps
        self.bd_extractor = bd_extractor
        self.test = test
        brax_env = create(
            env_name=env_name,
            episode_length=max_steps,
            legacy_spring=legacy_spring,
        )
        self.obs_shape = tuple([brax_env.observation_size, ])
        self.act_shape = tuple([brax_env.action_size, ])

        def detect_feet_contact(state, action):
            _, info = brax_env.sys.step(state.qp, action)
            has_contacts = jnp.abs(
                jnp.clip(info.contact.vel, a_min=-1., a_max=1.)
            ).sum(axis=-1) > 0
            return has_contacts[2::2].astype(jnp.int32)

        def reset_fn(key):
            state = brax_env.reset(key)
            state = State(state=state, obs=state.obs,
                          feet_contact=jnp.zeros(4, dtype=jnp.int32))
            if bd_extractor is not None:
                state = bd_extractor.init_extended_state(state)
            return state
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            feet_contact = detect_feet_contact(state.state, action)
            brax_state = brax_env.step(state.state, action)
            state = state.replace(
                state=brax_state, obs=brax_state.obs, feet_contact=feet_contact)
            if bd_extractor is not None:
                state = bd_extractor.update(
                    state, action, brax_state.reward, brax_state.done)
            return state, brax_state.reward, brax_state.done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)


class AntBDExtractor(BDExtractor):
    """Behavior descriptor extractor for the Ant locomotion task."""

    def __init__(self, logger):
        # Each BD represents the quantized foot-ground contact time ratio.
        bd_spec = [
            ('feet1_contact', 10),
            ('feet2_contact', 10),
            ('feet3_contact', 10),
            ('feet4_contact', 10),
        ]
        bd_state_spec = [
            ('feet_contact_times', jnp.ndarray),
            ('step_cnt', jnp.int32),
            ('valid_mask', jnp.int32),
        ]
        super(AntBDExtractor, self).__init__(bd_spec, bd_state_spec, State)
        self._logger = logger

    def init_state(self, extended_task_state):
        return {
            'feet_contact_times': jnp.zeros_like(
                extended_task_state.feet_contact, dtype=jnp.int32),
            'step_cnt': jnp.zeros((), dtype=jnp.int32),
            'valid_mask': jnp.ones((), dtype=jnp.int32),
        }

    def update(self, extended_task_state, action, reward, done):
        valid_mask = (
                extended_task_state.valid_mask * (1 - done)).astype(jnp.int32)
        return extended_task_state.replace(
            feet_contact_times=(
                extended_task_state.feet_contact_times +
                extended_task_state.feet_contact * valid_mask),
            step_cnt=extended_task_state.step_cnt + valid_mask,
            valid_mask=valid_mask)

    def summarize(self, extended_task_state):
        feet_contact_ratio = (
            extended_task_state.feet_contact_times /
            extended_task_state.step_cnt[..., None]).mean(axis=1)
        bds = {bd[0]: (feet_contact_ratio[:, i] * bd[1]).astype(jnp.int32)
               for i, bd in enumerate(self.bd_spec)}
        return extended_task_state.replace(**bds)
