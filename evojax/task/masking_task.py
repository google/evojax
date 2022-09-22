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
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState
from evojax.datasets import get_train_val_split, DatasetUtilClass
from evojax.models import CNN, create_train_state
from evojax.util import cross_entropy_loss


# This will allow training the CNN on the train data and mask on the validation split
@dataclass
class CNNData(object):
    obs: jnp.ndarray  # This will be the mnist image etc
    labels: jnp.ndarray  # This is the class label
    task_labels: jnp.ndarray  # This will be the label associated with each dataset
    # cnn_params: FrozenDict
    cnn_state: TrainState


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
                 max_steps: int = 1,
                 validation: bool = False,
                 test: bool = False,
                 pixel_input: bool = False,
                 datasets_tuple: Tuple[DatasetUtilClass, DatasetUtilClass, DatasetUtilClass] = None):

        self.max_steps = max_steps
        if pixel_input:
            self.obs_shape = tuple([1, 28, 28, 1])
        else:
            self.obs_shape = tuple([1, ])
        self.act_shape = tuple([10, ])

        if not test:
            dataset_class = datasets_tuple[int(validation)]
            image_data, class_labels, task_labels = dataset_class.return_data_arrays()
        else:
            dataset_class = datasets_tuple[-1]
            image_data, class_labels, task_labels = dataset_class.return_data_arrays()

        def reset_fn(key):
            next_key, key = random.split(key)

            batch_images, batch_class_labels, batch_task_labels = sample_batch(
                key, image_data, class_labels, task_labels, batch_size)

            cnn_state = create_train_state(key)

            cnn_data = CNNData(obs=batch_images,
                               labels=batch_class_labels,
                               task_labels=batch_task_labels,
                               cnn_state=cnn_state)

            if pixel_input:
                mask_obs = batch_images
            else:
                mask_obs = batch_task_labels

            return MaskTaskState(obs=mask_obs,
                                 labels=batch_class_labels,
                                 task_labels=batch_task_labels,
                                 cnn_data=cnn_data,
                                 key=next_key,
                                 steps=jnp.zeros((), dtype=int))

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state: MaskTaskState, action: jnp.ndarray):

            # if test:
            #     reward = step_accuracy(action, state.labels)
            # else:
            #     reward = -step_loss(action, state.labels)
            def train_step(cnn_state: TrainState, state_images, state_masks, state_labels):

                def loss_fn(params):
                    output_logits = CNN().apply({'params': params}, state_images, state_masks)
                    loss = cross_entropy_loss(logits=output_logits, labels=state_labels)
                    return loss, output_logits

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (_, logits), grads = grad_fn(cnn_state.params)
                cnn_state = cnn_state.apply_gradients(grads=grads)
                return cnn_state, logits

            cnn_data = TrainState.cnn_data

            new_cnn_state, step_logits = train_step(cnn_data.cnn_state, cnn_data.obs, action, cnn_data.labels)

            next_key, key = random.split(state.key)
            steps = state.steps + 1
            reward = jnp.where(steps >= self.max_steps, step_accuracy(step_logits, state.labels), 0)
            done = jnp.where(steps >= self.max_steps, 1, 0)
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)

            batch_images, batch_class_labels, batch_task_labels = sample_batch(
                key, image_data, class_labels, task_labels, batch_size)

            cnn_data = CNNData(obs=batch_images,
                               labels=batch_class_labels,
                               task_labels=batch_task_labels,
                               cnn_state=new_cnn_state)

            new_state = MaskTaskState(obs=batch_task_labels,
                                      labels=batch_class_labels,
                                      task_labels=batch_task_labels,
                                      cnn_data=cnn_data,
                                      key=next_key,
                                      steps=steps)

            return new_state, reward, done
            # return state, reward, jnp.ones(())
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> MaskTaskState:
        return self._reset_fn(key)

    def step(self,
             state: MaskTaskState,
             action: jnp.ndarray) -> Tuple[MaskTaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
