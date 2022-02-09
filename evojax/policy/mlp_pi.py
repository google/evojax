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

"""AttentionNeuron augmented MLP.

Ref: https://attentionneuron.github.io/
"""

import logging
from typing import Tuple
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.util import create_logger
from evojax.util import get_params_format_fn


@dataclass
class State(PolicyState):
    prev_actions: jnp.ndarray
    lstm_states: Tuple[jnp.ndarray, jnp.ndarray]


def pos_table(n, dim):
    """Create a table of positional encodings."""

    def get_angle(x, h):
        return x / np.power(10000, 2 * (h // 2) / dim)

    def get_angle_vec(x):
        return [get_angle(x, j) for j in range(dim)]

    tab = np.array([get_angle_vec(i) for i in range(n)]).astype(float)
    tab[:, 0::2] = np.sin(tab[:, 0::2])
    tab[:, 1::2] = np.cos(tab[:, 1::2])
    return tab


class AttentionNeuronMLP(nn.Module):

    act_dim: int
    msg_dim: int
    hidden_dim: int
    pos_em_dim: int
    pos_em: np.ndarray

    @nn.compact
    def __call__(self, obs, prev_act, lstm_h):
        obs_dim = obs.shape[0]
        # obs.shape: (obs_dim,) to (obs_dim, 1)
        obs = jnp.expand_dims(obs, axis=-1)
        # prev_act.shape: (act_dim,) to (obs_dim, act_dim)
        prev_act = jnp.repeat(
            jnp.expand_dims(prev_act, axis=0), repeats=obs_dim, axis=0)
        # x_aug.shape: (obs_dim, act_dim + 1)
        x_aug = jnp.concatenate([obs, prev_act], axis=-1)

        # q.shape: (hidden_dim, msg_dim)
        q = nn.Dense(self.msg_dim)(self.pos_em)

        # x_key.shape: (obs_dim, pos_em_dim)
        new_lstm_h, x_key = nn.LSTMCell()(lstm_h, x_aug)
        # k.shape: (obs_dim, msg_dim)
        k = nn.Dense(self.msg_dim)(x_key)

        # att.shape: (hidden_dim, obs_dim)
        att = nn.tanh(jnp.matmul(q, k.T))
        # x.shape: (hidden_dim,)
        x = nn.tanh(jnp.matmul(att, obs).squeeze(-1))
        # act.shape: (act_dim,)
        act = nn.tanh(nn.Dense(self.act_dim)(x))

        return act, new_lstm_h


class PermutationInvariantPolicy(PolicyNetwork):
    """A permutation invariant model."""

    def __init__(self,
                 act_dim: int,
                 hidden_dim: int,
                 msg_dim: int = 32,
                 pos_em_dim: int = 8,
                 logger: logging.Logger = None):
        if logger is None:
            self._logger = create_logger(name='PermutationInvariantPolicy')
        else:
            self._logger = logger

        self.act_dim = act_dim
        self.pos_em_dim = pos_em_dim
        model = AttentionNeuronMLP(
            act_dim=act_dim,
            msg_dim=msg_dim,
            pos_em_dim=pos_em_dim,
            hidden_dim=hidden_dim,
            pos_em=pos_table(hidden_dim, pos_em_dim),
        )
        obs_dim = 5
        params = model.init(
            random.PRNGKey(0),
            obs=jnp.ones([obs_dim, ]),
            prev_act=jnp.zeros([act_dim, ]),
            lstm_h=(jnp.zeros([obs_dim, pos_em_dim]),
                    jnp.zeros([obs_dim, pos_em_dim])),
        )
        self.num_params, format_params_fn = get_params_format_fn(params)
        self._logger.info(
            'PermutationInvariantPolicy.num_params = {}'.format(self.num_params)
        )
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def reset(self, states: TaskState) -> PolicyState:
        keys = random.split(random.PRNGKey(0), states.obs.shape[0])
        b_size, obs_dim = states.obs.shape
        prev_act = jnp.zeros([b_size, self.act_dim])
        lstm_h = (jnp.zeros([b_size, obs_dim, self.pos_em_dim]),
                  jnp.zeros([b_size, obs_dim, self.pos_em_dim]))
        return State(keys=keys, prev_actions=prev_act, lstm_states=lstm_h)

    def get_actions(self,
                    t_tasks: TaskState,
                    params: jnp.ndarray,
                    p_states: State) -> Tuple[jnp.ndarray, State]:
        params = self._format_params_fn(params)
        act, lstm_h = self._forward_fn(
            params,
            obs=t_tasks.obs,
            prev_act=p_states.prev_actions,
            lstm_h=p_states.lstm_states)
        return act, State(p_states.keys, act, lstm_h)
