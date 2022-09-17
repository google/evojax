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
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState
from evojax.datasets import get_train_val_split


# This will allow training the CNN on the train data and mask on the validation split
@dataclass
class CNNData(object):
    obs: jnp.ndarray  # This will be the mnist image etc
    labels: jnp.ndarray  # This is the class label
    task_labels: jnp.ndarray  # This will be the label associated with each dataset


@dataclass
class MaskTaskState(TaskState):
    obs: jnp.ndarray  # This will be the mnist image etc
    labels: jnp.ndarray  # This is the class label
    task_labels: jnp.ndarray  # This will be the label associated with each dataset
    cnn_data: CNNData
    key: jnp.ndarray  # Random key for JAX
    steps: jnp.int32


def sample_batch(key: jnp.ndarray,
                 image_data: jnp.ndarray,
                 class_labels: jnp.ndarray,
                 task_labels: jnp.ndarray,
                 batch_size: int) -> Tuple:
    ix = random.choice(
        key=key, a=image_data.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(image_data, indices=ix, axis=0),
            jnp.take(class_labels, indices=ix, axis=0),
            jnp.take(task_labels, indices=ix, axis=0))


def step_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    target = jax.nn.one_hot(target, 10)
    return -jnp.mean(jnp.sum(prediction * target, axis=1))


def step_accuracy(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    predicted_class = jnp.argmax(prediction, axis=1)
    return jnp.mean(predicted_class == target)


def setup_task_data(test: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    image_data, labels = get_train_val_split(test)

    class_labels = labels[:, 0]
    task_labels = labels[:, 1]

    return image_data, class_labels, task_labels


class Masking(VectorizedTask):
    """Masking task for MNIST."""

    def __init__(self,
                 batch_size: int = 1024,
                 max_steps: int = 100,
                 test: bool = False):

        self.max_steps = max_steps
        self.obs_shape = tuple([1, ])
        self.act_shape = tuple([10, ])

        image_data, class_labels, task_labels = setup_task_data(test)

        def reset_fn(key):
            next_key, key = random.split(key)

            batch_images, batch_class_labels, batch_task_labels = sample_batch(
                key, image_data, class_labels, task_labels, batch_size)

            cnn_data = CNNData(obs=batch_images,
                               labels=batch_class_labels,
                               task_labels=batch_task_labels,)

            return MaskTaskState(obs=batch_task_labels,
                                 labels=batch_class_labels,
                                 task_labels=batch_task_labels,
                                 cnn_data=cnn_data,
                                 key=next_key,
                                 steps=jnp.zeros((), dtype=int))

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state: MaskTaskState, action: jnp.ndarray):

            if test:
                reward = step_accuracy(action, state.labels)
            else:
                reward = step_accuracy(action, state.labels)

            # next_key, key = random.split(state.key)
            # steps = state.steps + 1
            # done = jnp.where(steps >= self.max_steps, 1, 0)
            # steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            #
            # batch_images, batch_class_labels, batch_task_labels = sample_batch(
            #     key, image_data, class_labels, task_labels, batch_size)
            #
            # cnn_data = CNNData(obs=batch_images,
            #                    labels=batch_class_labels,
            #                    task_labels=batch_task_labels, )
            #
            # new_state = MaskTaskState(obs=batch_task_labels,
            #                           labels=batch_class_labels,
            #                           task_labels=batch_task_labels,
            #                           cnn_data=cnn_data,
            #                           key=next_key,
            #                           steps=steps)
            #
            # return new_state, reward, done
            return state, reward, jnp.ones(())
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> MaskTaskState:
        return self._reset_fn(key)

    def step(self,
             state: MaskTaskState,
             action: jnp.ndarray) -> Tuple[MaskTaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
