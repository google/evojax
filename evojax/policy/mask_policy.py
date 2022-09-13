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
import optax

from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.task.masking_task import State, CNNData
from evojax.util import create_logger, get_params_format_fn
from evojax.models import Mask, CNN, cnn_final_layer_name


def create_train_state(rng, learning_rate):
    """Creates initial `TrainState`."""
    params = CNN().init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=CNN().apply, params=params, tx=tx)


def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).sum()


@jax.jit
def cnn_train_step(cnn_state: train_state.TrainState, cnn_data: CNNData, masks: jnp.ndarray):
    """Train for a single step."""
    def loss_fn(params):
        output_logits = CNN().apply({'params': params}, cnn_data.obs, masks)
        loss = cross_entropy_loss(logits=output_logits, labels=cnn_data.labels)
        return loss, output_logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(cnn_state.params)
    cnn_state = cnn_state.apply_gradients(grads=grads)
    return cnn_state, logits


class MaskPolicy(PolicyNetwork):
    """A dense neural network for masking the MNIST classification task."""

    def __init__(self, logger: logging.Logger = None, learning_rate: float = 1e-3):
        if logger is None:
            self._logger = create_logger('MaskNetPolicy')
        else:
            self._logger = logger

        self.cnn_state = create_train_state(random.PRNGKey(0), learning_rate)
        self.mask_size = self.cnn_state.params[cnn_final_layer_name]["kernel"].shape[0]
        self.apply_cnn = jax.vmap(cnn_train_step)

        mask_model = Mask(mask_size=self.mask_size)
        params = mask_model.init(random.PRNGKey(0), jnp.ones([1, ]))

        self.num_params, format_params_fn = get_params_format_fn(params)
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(mask_model.apply)

    def get_actions(self,
                    t_states: State,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        masks = self._forward_fn(params, t_states.obs)
        # mask_input = masks.reshape((8, 1024, self.mask_size))

        self._logger.info(f'Masks of shape: {masks.shape}')
        # self._logger.info(f'Mask input of shape: {mask_input.shape}')

        cnn_data = t_states.cnn_data
        self.cnn_state, output_logits = self.apply_cnn(self.cnn_state, cnn_data, masks)

        return output_logits, p_states
