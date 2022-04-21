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
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray


def sample_batch(key: jnp.ndarray,
                 data: jnp.ndarray,
                 labels: jnp.ndarray,
                 batch_size: int) -> Tuple:
    ix = random.choice(
        key=key, a=data.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(data, indices=ix, axis=0),
            jnp.take(labels, indices=ix, axis=0))


def loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    target = jax.nn.one_hot(target, 10)
    return -jnp.mean(jnp.sum(prediction * target, axis=1))


def accuracy(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    predicted_class = jnp.argmax(prediction, axis=1)
    return jnp.mean(predicted_class == target)


class MNIST(VectorizedTask):
    """MNIST classification task."""

    def __init__(self,
                 batch_size: int = 1024,
                 test: bool = False):

        self.max_steps = 1
        self.obs_shape = tuple([28, 28, 1])
        self.act_shape = tuple([10, ])

        # Delayed importing of torchvision

        try:
            from torchvision import datasets
        except ModuleNotFoundError:
            print('You need to install torchvision for this task.')
            print('  pip install torchvision')
            sys.exit(1)

        dataset = datasets.MNIST('./data', train=not test, download=True)
        data = np.expand_dims(dataset.data.numpy() / 255., axis=-1)
        labels = dataset.targets.numpy()

        def reset_fn(key):
            if test:
                batch_data, batch_labels = data, labels
            else:
                batch_data, batch_labels = sample_batch(
                    key, data, labels, batch_size)
            return State(obs=batch_data, labels=batch_labels)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            if test:
                reward = accuracy(action, state.labels)
            else:
                reward = -loss(action, state.labels)
            return state, reward, jnp.ones(())
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
