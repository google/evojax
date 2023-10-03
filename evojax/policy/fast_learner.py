
import logging
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
from flax import linen as nn

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn


class CNN(nn.Module):
    """CNN for MNIST."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=8, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((*x.shape[:-3], -1))  # flatten
        x = nn.Dense(features=32)(x)
        x = nn.Dense(features=5)(x)
        x = nn.log_softmax(x)
        return x

def get_update_params_fn(apply_fn):
    # this function could be something much more complicated other than vanilla sgd...
    # params has shape [pop_size, ...]
    # x has shape [pop_size, batch_size, ways * shots, 28, 28 ,1]
    # y has shape [pop_size, batch_size, ways * shots, 1]
    lr = 0.4
    # loss for one param and one set
    def loss_fn(params, x, y):
        pred = apply_fn(params, x) # [ways * shots, n_classes]
        return -jnp.take_along_axis(pred, y, axis=-1).mean()
    
    def update_one_set(params, x, y):
        grad = jax.grad(loss_fn)(params, x, y)
        new_params = tree_map(lambda p,g : p - lr * g, params, grad)
        return new_params
    
    update_multi_set = jax.vmap(update_one_set, in_axes=[0,0,0], out_axes=0) # maps over pop_size
    update_multi_params_multi_set = jax.vmap(update_multi_set,in_axes=[None,1,1], out_axes=1) # maps over batch_size
    # return update_multi_params_multi_set(params, x, y)
    subsequent_update_multi_params_multi_set = jax.vmap(update_multi_set,in_axes=[1,1,1], out_axes=1)
    return update_multi_params_multi_set, subsequent_update_multi_params_multi_set
    

class FastLearner(PolicyNetwork):
    """A convolutional neural network for the MNIST classification task."""

    def __init__(self,
                 n_classes: int,
                 num_grad_steps: int,
                 logger: logging.Logger = None,
                 ):
        if logger is None:
            self._logger = create_logger('FastLearner')
        else:
            self._logger = logger
        assert num_grad_steps > 0
        self.num_grad_steps = num_grad_steps
        model = CNN()
        params = model.init(random.PRNGKey(0), jnp.zeros([1, 28, 28, 1]))
        self.num_params, format_params_fn = get_params_format_fn(params)
        self._logger.info(
            'FastLearner.num_params = {}'.format(self.num_params))
        self._format_params_fn = jax.vmap(format_params_fn) # this maps over members
        self._model_apply = model.apply
        # self._forward_fn = jax.vmap(model.apply) # this maps over members
        self._update_fn, self._subsequent_update_fn = get_update_params_fn(self._model_apply)
        # lr = 1e-3
        # _loss_fn = partial(loss_fn, self._forward_fn, n_classes)
        # def update_params_one_set(params, x, y):
        #     grads = jax.grad(_loss_fn)(params, x, y)
        #     new_params = params - lr * grads
        #     return new_params
        
        # update_params_multi_set = jax.vmap(update_params_one_set, in_axes=[None,0,0], out_axes=[0])
        # update_params_pop_multi_set = jax.vmap(update_params_multi_set, in_axes=[0,None,None], out_axes=[0])

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        x, y = t_states.obs, t_states.labels
        # x shape: [pop_size, batch_size, ways x shots, 28, 28, 1]
        # y shape: [pop_size, batch_size, ways x shots]
        # params shape: [pop_size, ...]
        # updated_params = update_params(self._model_apply, params, x, y)
        updated_params = self._update_fn(params, x, y)
        # updated_params shape: [pop_size, batch_size, ...]
        for _ in range(self.num_grad_steps-1):
            updated_params = self._subsequent_update_fn(updated_params, x, y)

        apply_batched_params = jax.vmap(jax.vmap(self._model_apply))
        # the output prediction should have shape: [pop_size, batch_size, ways * shots, n_classes]
        return apply_batched_params(updated_params, t_states.test_obs), p_states
