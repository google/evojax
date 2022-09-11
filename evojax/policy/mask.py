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
from typing import Tuple, Optional

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

from evojax.datasets import DATASET_LABELS
mask_final_layer_name = 'DENSE'


class Mask(nn.Module):
    """Mask network for MNIST."""
    mask_size: int
    dataset_number: int = 4
    round_output: bool = True
    test_no_mask: bool = False
    pixel_input: bool = False

    @nn.compact
    def __call__(self, x):
        if self.pixel_input:
            x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME", name="CONV1")(x)
            x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME", name="CONV2")(x)
            x = x.reshape((x.shape[0], -1))
        else:
            x = nn.one_hot(x, self.dataset_number)
        # x = nn.Dense(features=10, name="DENSE1")(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=100, name="DENSE2")(x)
        # x = nn.relu(x)
        x = nn.Dense(features=self.mask_size, name=mask_final_layer_name)(x)
        x = nn.sigmoid(x)
        if self.round_output:
            x = jnp.round(x)

        if self.test_no_mask:
            return jnp.ones((x.shape[0], self.mask_size))
        else:
            return x


def set_bias_and_weights(params):
    """ Sets the bias and weights such that initial outputs of the Mask will all be one. """
    params = unfreeze(params)
    final_mask_weights = params["params"][mask_final_layer_name]["kernel"]
    final_mask_bias = params["params"][mask_final_layer_name]["bias"]
    params["params"][mask_final_layer_name]["kernel"] = jnp.zeros_like(final_mask_weights)
    params["params"][mask_final_layer_name]["bias"] = jnp.ones_like(final_mask_bias) * 0.51
    return freeze(params)


class MaskPolicy(PolicyNetwork):
    """A dense neural network for masking the MNIST classification task."""

    def __init__(self, logger: logging.Logger = None, mask_size: int = None,
                 batch_size: int = None, test_no_mask=False, dataset_number=4, pixel_input=False):
        if logger is None:
            self._logger = create_logger('MaskNetPolicy')
        else:
            self._logger = logger

        model = Mask(mask_size=mask_size, test_no_mask=test_no_mask,
                     dataset_number=dataset_number, pixel_input=pixel_input)
        params = model.init(random.PRNGKey(0), jnp.ones([batch_size, ]))

        # I want to start with no masking, then move away from this
        # Try having all weights in the final layer be zero with a bias of ones
        params = set_bias_and_weights(params)

        self.dummy_data = jnp.array(list(DATASET_LABELS.values()))
        dummy_output = model.apply({"params": params["params"]}, self.dummy_data)
        assert jnp.array_equal(jnp.ones((len(DATASET_LABELS), mask_size)), dummy_output)

        self.initial_params = params

        self.num_params, format_params_fn = get_params_format_fn(params)
        self._logger.info(f'Mask.num_params = {self.num_params}')
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)
        self._format_params_fn_no_vmap = format_params_fn
        self._forward_fn_no_vmap = model.apply

    def get_actions(self,
                    t_states: Optional[TaskState],
                    params: jnp.ndarray,
                    p_states: Optional[PolicyState]) -> Tuple[jnp.ndarray, Optional[PolicyState]]:
        # Use this to test what masks are generated for each dataset
        if t_states is None:
            params = self._format_params_fn_no_vmap(params)
            return self._forward_fn_no_vmap(params, self.dummy_data), None
        else:
            params = self._format_params_fn(params)
            return self._forward_fn(params, t_states.obs), p_states
