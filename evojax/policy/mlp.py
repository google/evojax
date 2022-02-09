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

import logging
from typing import Sequence
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn


class MLP(nn.Module):
    feat_dims: Sequence[int]
    out_dim: int
    out_fn: str

    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.feat_dims:
            x = nn.tanh(nn.Dense(hidden_dim)(x))
        x = nn.Dense(self.out_dim)(x)
        if self.out_fn == 'tanh':
            x = nn.tanh(x)
        elif self.out_fn == 'softmax':
            x = nn.softmax(x, axis=-1)
        else:
            raise ValueError(
                'Unsupported output activation: {}'.format(self.out_fn))
        return x


class MLPPolicy(PolicyNetwork):
    """A general purpose multi-layer perceptron model."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: int,
                 output_act_fn: str = 'tanh',
                 logger: logging.Logger = None):
        if logger is None:
            self._logger = create_logger(name='MLPPolicy')
        else:
            self._logger = logger

        model = MLP(
            feat_dims=hidden_dims, out_dim=output_dim, out_fn=output_act_fn)
        params = model.init(random.PRNGKey(0), jnp.ones([1, input_dim]))
        self.num_params, format_params_fn = get_params_format_fn(params)
        self._logger.info('MLPPolicy.num_params = {}'.format(self.num_params))
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states
