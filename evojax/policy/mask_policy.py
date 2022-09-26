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
from flax.training import train_state
from flax.struct import dataclass

from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.task.masking_task import MaskTaskState
from evojax.util import create_logger, get_params_format_fn
from evojax.models import Mask, PixelMask, cnn_final_layer_name, create_train_state


@dataclass
class MaskPolicyState(PolicyState):
    keys: jnp.ndarray
    cnn_params: jnp.ndarray


class MaskPolicy(PolicyNetwork):
    """A dense neural network for masking the MNIST classification task."""

    def __init__(self, logger: logging.Logger = None, learning_rate: float = 1e-3,
                 mask_threshold: float = 0.5, pixel_input: bool = False, image_mask: bool = False,
                 pretrained_cnn_state: train_state.TrainState = None):
        if logger is None:
            self._logger = create_logger('MaskNetPolicy')
        else:
            self._logger = logger

        self.mask_threshold = mask_threshold
        self.image_mask = image_mask
        self.lr = learning_rate

        if pretrained_cnn_state:
            self.cnn_state = pretrained_cnn_state
        else:
            self.cnn_state = create_train_state(random.PRNGKey(0), learning_rate)

        if image_mask:
            self.mask_size = 28*28
        else:
            self.mask_size = self.cnn_state.params[cnn_final_layer_name]["kernel"].shape[0]

        self._train_fn_cnn = jax.vmap(self.cnn_state.apply_fn, in_axes=(None, 0, 0))

        self.cnn_num_params, cnn_format_params_fn = get_params_format_fn(self.cnn_state.params)
        self._cnn_format_params_fn = jax.vmap(cnn_format_params_fn)

        if pixel_input:
            mask_model = PixelMask(mask_size=self.mask_size)
            params = mask_model.init(random.PRNGKey(0), jnp.ones([1, 28, 28, 1]))
        else:
            mask_model = Mask(mask_size=self.mask_size)
            params = mask_model.init(random.PRNGKey(0), jnp.ones([1, ]))

        self.num_params, format_params_fn = get_params_format_fn(params)
        self.external_format_params_fn = format_params_fn

        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(mask_model.apply)
        self._no_vmap_forward_fn = mask_model.apply

    def get_task_label_masks(self, params):
        masks = self._no_vmap_forward_fn(params, jnp.arange(4))
        if self.mask_threshold is not None:
            masks = jnp.where(masks > self.mask_threshold, 1, 0)
        return masks

    def get_actions(self,
                    t_states: MaskTaskState,
                    params: jnp.ndarray,
                    p_states: MaskPolicyState) -> Tuple[jnp.ndarray, MaskPolicyState]:

        params = self._format_params_fn(params)
        masks = self._forward_fn(params, t_states.obs)
        if self.mask_threshold is not None:
            masks = jnp.where(masks > self.mask_threshold, 1, 0)

        return masks, p_states
