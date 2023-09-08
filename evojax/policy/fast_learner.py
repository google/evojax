
import logging
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


class CNN(nn.Module):
    """CNN for MNIST."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=8, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[:-3], -1))  # flatten
        x = nn.Dense(features=n_classes)(x)
        x = nn.log_softmax(x)
        return x

def update_params(apply_fn, params, x, y):
    # this function could be something much more complicated other than vanilla sgd...
    # params has shape [pop_size, ...]
    # x has shape [batch_size, ways * shots, 28, 28 ,1]
    # y has shape [batch_size, ways * shots, 1]

    # loss for one param and one set
    def loss_fn(params, x, y):
        pred = apply_fn(params, x) # [ways * shots, n_classes]
        return -jnp.take_along_axis(pred, labels, axis=-1).mean()
    
    def update_one_set(params, x, y):
        grad = jax.grad(loss_fn)(params, x, y)
        new_params = tree_map(lambda p,g : p - lr * g, params, grad)
        return new_params
    
    update_multi_set = jax.vmap(update_one_set, in_axes=[None,0,0], out_axes=0)
    update_multi_params_multi_set = jax.vmap(update_multi_set, in_axes=[0,None,None], out_axes=0)
    return update_multi_params_multi_set(params, x, y)
    

class FastLearner(PolicyNetwork):
    """A convolutional neural network for the MNIST classification task."""

    def __init__(self,
                 n_classes: int,
                 logger: logging.Logger = None,
                 ):
        if logger is None:
            self._logger = create_logger('FastLearner')
        else:
            self._logger = logger

        model = CNN()
        params = model.init(random.PRNGKey(0), jnp.zeros([1, 28, 28, 1]))
        self.num_params, format_params_fn = get_params_format_fn(params)
        self._logger.info(
            'FastLearner.num_params = {}'.format(self.num_params))
        self._format_params_fn = jax.vmap(format_params_fn) # this maps over members
        self._forward_fn = jax.vmap(model.apply) # this maps over members
        
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
        x, y = t_state.obs
        # x shape: [pop_size, batch_size, ways x shots, 28, 28, 1]
        # y shape: [pop_size, batch_size, ways x shots]
        # params shape: [pop_size, ...]
        updated_params = update_params(params, x, y)
        # updated_params shape: [pop_size, batch_size, ...]

        apply_batched_params = jax.vmap(model.apply, in_axes=[0,None])
        # the output prediction should have shape: [pop_size, batch_size, ways * shots, n_classes]
        return apply_batched_params(updated_params, t_states.test_inputs), p_states