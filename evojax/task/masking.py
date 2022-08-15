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

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

from evojax.datasets import read_data_files, digit, fashion, kuzushiji


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    class_labels: jnp.ndarray
    dataset_labels: jnp.ndarray


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


class Masking(VectorizedTask):
    """Masking task for MNIST."""

    def __init__(self,
                 batch_size: int = 1024,
                 test: bool = False,
                 mnist_model=None):

        self.mnist_model = mnist_model

        self.max_steps = 1
        self.obs_shape = tuple([1, ])
        self.act_shape = tuple([10, ])

        train_dataset = {}
        test_dataset = {}
        x_array_train, y_array_train = [], []
        x_array_test, y_array_test = [], []

        for dataset_name in [digit, fashion, kuzushiji]:
            x_train, y_train = read_data_files(dataset_name, 'train')
            x_array_train.append(x_train)
            y_array_train.append(y_train)

            x_test, y_test = read_data_files(dataset_name, 'test')
            x_array_test.append(x_test)
            y_array_test.append(y_test)

        train_dataset['image'] = jnp.float32(np.concatenate(x_array_train)) / 255.
        test_dataset['image'] = jnp.float32(np.concatenate(x_array_test)) / 255.

        # Only use the class label, not dataset label here
        train_dataset['label'] = jnp.int16(np.concatenate(y_array_train)[:, 0])
        test_dataset['label'] = jnp.int16(np.concatenate(y_array_test)[:, 0])


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
                params = unfreeze(params)
                params['bert'] = bert_params
                params = freeze(params)

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
