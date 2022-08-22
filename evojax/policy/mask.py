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
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from flax.core import freeze, unfreeze

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn

final_layer_name = "DENSE-FINAL"

class Mask(nn.Module):
    """Mask network for MNIST."""
    mask_size: int
    dataset_number: int = 3
    round_output: bool = True
    test_no_mask: bool = False

    @nn.compact
    def __call__(self, x):
        if self.test_no_mask:
            x = jnp.ones((x.shape[0], self.mask_size))

        x = nn.one_hot(x, self.dataset_number)
        x = nn.Dense(features=10, name="DENSE1")(x)
        x = nn.relu(x)
        x = nn.Dense(features=100, name="DENSE2")(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.mask_size, name=final_layer_name)(x)
        x = nn.sigmoid(x)
        if self.round_output:
            x = jnp.round(x)
        return x


class MaskPolicy(PolicyNetwork):
    """A dense neural network for masking the MNIST classification task."""

    def __init__(self, logger: logging.Logger = None, mask_size: int = None,
                 batch_size: int = None, test_no_mask=False):
        if logger is None:
            self._logger = create_logger('MaskNetPolicy')
        else:
            self._logger = logger

        model = Mask(mask_size=mask_size, test_no_mask=test_no_mask)
        params = model.init(random.PRNGKey(0), jnp.ones([batch_size, ]))

        # I want to start with no masking, then move away from this
        # Try having all weights in the final layer be zero with a bias of ones
        params = unfreeze(params)
        final_mask_weights = params["params"][final_layer_name]["kernel"]
        final_mask_bias = params["params"][final_layer_name]["bias"]
        params["params"][final_layer_name]["kernel"] = jnp.zeros_like(final_mask_weights)
        params["params"][final_layer_name]["bias"] = jnp.ones_like(final_mask_bias)
        params = freeze(params)

        self.num_params, format_params_fn = get_params_format_fn(params)
        self._logger.info(f'Mask.num_params = {self.num_params}')
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states
