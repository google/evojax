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
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
import optax

from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.task.masking_task import MaskTaskState
from evojax.util import create_logger, get_params_format_fn
from evojax.models import Mask, CNN, cnn_final_layer_name


@dataclass
class MaskPolicyState(PolicyState):
    keys: jnp.ndarray
    cnn_params: jnp.ndarray


def create_train_state(rng, learning_rate):
    """Creates initial `TrainState`."""
    params = CNN().init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=CNN().apply, params=params, tx=tx)


def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).sum()


def cnn_train_step(cnn_params: FrozenDict, images: jnp.ndarray, labels: jnp.ndarray, masks: jnp.ndarray):
    """Train for a single step."""
    def loss_fn(params):
        output_logits = CNN().apply({'params': params}, images, masks)
        loss = cross_entropy_loss(logits=output_logits, labels=labels)
        return loss, output_logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(cnn_params)
    return grads, logits


class MaskPolicy(PolicyNetwork):
    """A dense neural network for masking the MNIST classification task."""

    def __init__(self, logger: logging.Logger = None, learning_rate: float = 1e-3, mask_threshold: float = 0.5,
                 pretrained_cnn_state: train_state.TrainState = None):
        if logger is None:
            self._logger = create_logger('MaskNetPolicy')
        else:
            self._logger = logger

        self.mask_threshold = mask_threshold
        self.lr = learning_rate

        if pretrained_cnn_state:
            self.cnn_state = pretrained_cnn_state
        else:
            self.cnn_state = create_train_state(random.PRNGKey(0), learning_rate)

        self.mask_size = self.cnn_state.params[cnn_final_layer_name]["kernel"].shape[0]

        self._train_fn_cnn = jax.vmap(self.cnn_state.apply_fn)
        # self._train_fn_cnn = jax.vmap(cnn_train_step)

        self.cnn_num_params, cnn_format_params_fn = get_params_format_fn(self.cnn_state.params)
        self._cnn_format_params_fn = jax.vmap(cnn_format_params_fn)
        # self._cnn_format_params_fn = cnn_format_params_fn

        mask_model = Mask(mask_size=self.mask_size)
        params = mask_model.init(random.PRNGKey(0), jnp.ones([1, ]))

        self.num_params, format_params_fn = get_params_format_fn(params)
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(mask_model.apply)

    @staticmethod
    def flatten_params(dict_params: FrozenDict):
        flat, _ = jax.tree_util.tree_flatten(dict_params)
        return jnp.concatenate([i.ravel() for i in flat])

    def update_from_state(self, policy_state: MaskPolicyState):
        """ Func to update the cnn from the pmapped policy state. """
        mean_params = jnp.mean(policy_state.cnn_params, axis=0)
        param_dict = self._cnn_format_params_fn(mean_params)
        self.cnn_state = self.cnn_state.replace(params=param_dict)

    # def reset(self, states: MaskTaskState) -> MaskPolicyState:
    #     """Reset the policy.
    #
    #     Args:
    #         State - Initial observations.
    #     Returns:
    #         PolicyState. Policy internal states.
    #     """
    #
    #     split_size = states.obs.shape[0]
    #     keys = jax.random.split(jax.random.PRNGKey(0), split_size)
    #
    #     flat_params = self.flatten_params(self.cnn_state.params)
    #     # flat_params = jnp.tile(flat_params, jax.local_device_count())
    #     flat_params = jnp.stack([flat_params]*split_size, axis=0)
    #
    #     return MaskPolicyState(keys=keys,
    #                            cnn_params=flat_params)

    def get_actions(self,
                    t_states: MaskTaskState,
                    params: jnp.ndarray,
                    p_states: MaskPolicyState) -> Tuple[jnp.ndarray, MaskPolicyState]:

        # import ipdb
        # ipdb.set_trace()

        params = self._format_params_fn(params)
        masking_output = self._forward_fn(params, t_states.obs)
        masks = jnp.where(masking_output > self.mask_threshold, 1, 0)

        cnn_data = t_states.cnn_data
        output_logits = self._train_fn_cnn({"params": self.cnn_state.params}, cnn_data.obs, masks)
        #
        #
        # # cnn_params = self._cnn_format_params_fn(p_states.cnn_params)
        # # cnn_params = self._cnn_format_params_fn(jnp.mean(p_states.cnn_params, axis=0))
        # # Note that there should only be one set of cnn_params so this func shouldn't be vmapped
        # # mean_cnn_params = jnp.mean(p_states.cnn_params, axis=0)
        # cnn_params = self._cnn_format_params_fn(p_states.cnn_params)
        #
        # # self.cnn_state, output_logits = self.apply_cnn(self.cnn_state, cnn_data, masks)
        # # self.cnn_state, output_logits = cnn_train_step(self.cnn_state, cnn_data.obs, cnn_data.labels, masks)
        # # output_logits = self._forward_fn_cnn({"params": self.cnn_state.params}, cnn_data.obs, masks)
        # # output_logits = self._forward_fn_cnn({"params": self.cnn_state.params}, cnn_data.obs, masks)
        # grads, output_logits = self._train_fn_cnn(cnn_params, cnn_data.obs, cnn_data.labels, masks)
        #
        # # Need to average the grads over the vmap
        # # mean_grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)
        # # self.cnn_state = self.cnn_state.apply_gradients(grads=mean_grads)
        #
        # # # TODO see if these can be applied using the opt in the cnn_state
        # # mean_grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)
        # # # new_cnn_state = cnn_state.apply_gradients(grads=mean_grads)
        # updated_params = jax.tree_map(
        #     lambda p, g: p - self.lr * g, cnn_params, grads
        # )
        #
        # flat, _ = jax.tree_util.tree_flatten(updated_params)
        # flat_params = jnp.concatenate([i.reshape((i.shape[0], -1)) for i in flat], axis=1)
        # # mean_flat_params = jax.lax.pmean(flat_params, axis_name='num_devices')
        #
        # assert flat_params.shape == p_states.cnn_params.shape
        #
        # # new_p_state_params = jnp.stack([flat_params] * jax.local_device_count(), axis=0)
        # # TODO check how these are recombined
        # new_p_states = MaskPolicyState(keys=p_states.keys,
        #                                cnn_params=flat_params)
        #
        # return output_logits, new_p_states
        return output_logits, p_states
