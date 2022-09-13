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

from typing import Tuple, Optional
import numpy as np

import optax
import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass
from flax.training import train_state
from flax.core.frozen_dict import FrozenDict

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

from evojax.datasets import read_data_files, DATASET_LABELS
from evojax.models import CNN, cnn_final_layer_name


def create_train_state(rng, learning_rate):
    """Creates initial `TrainState`."""
    params = CNN().init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=CNN().apply, params=params, tx=tx)


@dataclass
class CNNData(object):
    obs: jnp.ndarray  # This will be the mnist image etc
    labels: jnp.ndarray  # This is the class label
    task_labels: jnp.ndarray  # This will be the label associated with each dataset


@dataclass
class State(TaskState):
    obs: jnp.ndarray  # This will be the mnist image etc
    labels: jnp.ndarray  # This is the class label
    task_labels: jnp.ndarray  # This will be the label associated with each dataset
    cnn_state: train_state.TrainState  # The parameters for the CNN
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


def step_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    target = jax.nn.one_hot(target, 10)
    return -jnp.mean(jnp.sum(prediction * target, axis=1))


def step_accuracy(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    predicted_class = jnp.argmax(prediction, axis=1)
    return jnp.mean(predicted_class == target)


def setup_task_data(test: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_array, y_array = [], []
    for dataset_name in list(DATASET_LABELS.keys()):
        x, y = read_data_files(dataset_name, 'test' if test else 'train')
        x_array.append(x)
        y_array.append(y)

    # # A validation set will be split out from the train set
    # if not test:
    #     full_train_images = jnp.float32(np.concatenate(x_array)) / 255.
    #     full_train_labels = jnp.int16(np.concatenate(y_array))
    #
    #     number_of_points = full_train_images.shape[0]
    #     number_for_validation = number_of_points // 5
    #
    #     # Use these indices every time for consistency
    #     ix = random.permutation(key=random.PRNGKey(0), x=number_of_points)
    #     validation_ix = ix[:number_for_validation]
    #     train_ix = ix[number_for_validation:]
    #
    #     # Just select the section of the data corresponding to train/val indices
    #     train_image_data = jnp.take(full_train_images, indices=train_ix, axis=0)
    #     train_labels = jnp.take(full_train_labels, indices=train_ix, axis=0)
    #     val_image_data = jnp.take(full_train_images, indices=validation_ix, axis=0)
    #     val_labels = jnp.take(full_train_labels, indices=validation_ix, axis=0)
    # else:
    image_data = jnp.float32(np.concatenate(x_array)) / 255.
    labels = jnp.int16(np.concatenate(y_array))

    class_labels = labels[:, 0]
    task_labels = labels[:, 1]

    return image_data, class_labels, task_labels


class Masking(VectorizedTask):
    """Masking task for MNIST."""

    def __init__(self,
                 batch_size: int = 1024,
                 max_steps: int = 100,
                 learning_rate: float = 1e-3,
                 test: bool = False,
                 mask_size: int = None):

        self.max_steps = max_steps
        self.obs_shape = tuple([1, ])
        self.act_shape = tuple([mask_size, ])

        image_data, class_labels, task_labels = setup_task_data(test)

        def reset_fn(key):
            next_key, key = random.split(key)

            cnn_state = create_train_state(key, learning_rate)

            batch_images, batch_class_labels, batch_task_labels = sample_batch(
                key, image_data, class_labels, task_labels, batch_size)

            cnn_data = CNNData(obs=batch_images,
                               labels=batch_class_labels,
                               task_labels=batch_task_labels,)

            return State(obs=batch_task_labels,
                         labels=batch_class_labels,
                         task_labels=batch_task_labels,
                         cnn_state=cnn_state,
                         cnn_data=cnn_data,
                         key=next_key,
                         steps=jnp.zeros((), dtype=int))
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            next_key, key = random.split(state.key)

            cnn_state, output_logits = cnn_train_step(state.cnn_state, state.cnn_data, action)

            if test:
                reward = step_accuracy(output_logits, state.labels)
            else:
                reward = -step_loss(output_logits, state.labels)

            steps = state.steps + 1
            done = jnp.where(steps >= self.max_steps, 1, 0)
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)

            batch_images, batch_class_labels, batch_task_labels = sample_batch(
                key, image_data, class_labels, task_labels, batch_size)

            cnn_data = CNNData(obs=batch_images,
                               labels=batch_class_labels,
                               task_labels=batch_task_labels, )

            new_state = State(obs=batch_task_labels,
                              labels=batch_class_labels,
                              task_labels=batch_task_labels,
                              cnn_state=cnn_state,
                              cnn_data=cnn_data,
                              key=next_key,
                              steps=steps)

            return new_state, reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
