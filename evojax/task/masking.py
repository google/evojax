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

import jax
import jax.numpy as jnp
from jax import random
from flax.core import freeze, unfreeze
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

from evojax.datasets import read_data_files, DATASET_LABELS
from evojax.train_mnist_cnn import CNN, linear_layer_name
from flaxmodels.flaxmodels.resnet import ResNet18


@dataclass
class State(TaskState):
    obs: Optional[jnp.ndarray]  # This will be the dataset label for now as this is the input for the masker
    labels: Optional[jnp.ndarray]  # This is the class label
    image_data: Optional[jnp.ndarray]


def sample_batch(key: jnp.ndarray,
                 data: jnp.ndarray,
                 class_labels: jnp.ndarray,
                 dataset_labels: jnp.ndarray,
                 batch_size: int) -> Tuple:
    ix = random.choice(
        key=key, a=data.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(data, indices=ix, axis=0),
            jnp.take(class_labels, indices=ix, axis=0),
            jnp.take(dataset_labels, indices=ix, axis=0))


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
                 validation: bool = False,
                 mnist_params=None,
                 mask_size: int = None):

        self.mnist_params = mnist_params
        self.linear_weights_orig = self.mnist_params[linear_layer_name]["kernel"]

        self.max_steps = 1
        self.obs_shape = tuple([1, ])
        self.act_shape = tuple([mask_size, ])

        x_array, y_array = [], []
        for dataset_name in list(DATASET_LABELS.keys()):
            x, y = read_data_files(dataset_name, 'test' if test else 'train')
            x_array.append(x)
            y_array.append(y)

        # A validation set will be split out from the train set
        if not test:
            full_train_images = jnp.float32(np.concatenate(x_array)) / 255.
            full_train_labels = jnp.int16(np.concatenate(y_array)[:, 0])

            number_of_points = full_train_images.shape[0]
            number_for_validation = number_of_points // 5

            # Use these indices every time for consistency
            ix = random.permutation(key=random.PRNGKey(0), x=number_of_points)
            validation_ix = ix[:number_for_validation]
            train_ix = ix[number_for_validation:]

            # Just select the section of the data corresponding to train/val indices
            if not validation:
                image_data = jnp.take(full_train_images, indices=train_ix, axis=0)
                labels = jnp.take(full_train_labels, indices=train_ix, axis=0)
            else:
                image_data = jnp.take(full_train_images, indices=validation_ix, axis=0)
                labels = jnp.take(full_train_labels, indices=validation_ix, axis=0)

        else:
            image_data = jnp.float32(np.concatenate(x_array)) / 255.
            labels = jnp.int16(np.concatenate(y_array))

        class_labels = labels[:, 0]
        dataset_labels = labels[:, 1]

        def reset_fn(key):
            batch_data, batch_class_labels, batch_dataset_labels = sample_batch(
                key, image_data, class_labels, dataset_labels, batch_size)
            return State(obs=batch_dataset_labels,
                         labels=batch_class_labels,
                         image_data=batch_data)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):

            # Action should be the mask which will be applied to the linear weights
            # TODO Should the weights be masked or just the input to the linear layer
            # params = unfreeze(self.mnist_params)
            # masked_weights = self.linear_weights_orig * action.reshape(self.linear_weights_orig.shape)
            # params[linear_layer_name]["kernel"] = masked_weights
            # params = freeze(params)

            output_logits = CNN().apply({'params': self.mnist_params}, state.image_data, action)
            # output_logits = ResNet18(num_classes=10,
            #                          pretrained='').apply({'params': self.mnist_params}, state.image_data, action)

            if test:
                reward = accuracy(output_logits, state.labels)
            else:
                reward = accuracy(output_logits, state.labels)
                # reward = -loss(output_logits, state.labels)
            return state, reward, jnp.ones(())
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
