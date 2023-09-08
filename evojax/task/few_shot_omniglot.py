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



### step_fn and reset_fn defined here are meant for a ___single___ policy network
### which will then be mapped to all the members using vmap
### such that all members take as input the same observation
@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray

def loss(prediction: jnp.ndarray, labels: jnp.ndarray) -> jnp.float32:
    # target = jax.nn.one_hot(target, ways)
    # return -jnp.mean(jnp.sum(prediction * target, axis=1))
    return -jnp.take_along_axis(prediction, labels, axis=-1).mean()

def accuracy(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    predicted_class = jnp.argmax(prediction, axis=1)
    return jnp.mean(predicted_class == target)


class Omniglot(VectorizedTask):
    """Omniglot few-show learning classification task."""

    def __init__(self,
                 batch_size: int = 16,
                 test: bool = False):
        num_workers = 4
        test_shots = 15
        shots = 5
        self.ways = 5
        self.max_steps = 1
        self.obs_shape = tuple([28, 28, 1])
        self.act_shape = tuple([self.ways, ])


        # Delayed importing of torchmeta
        try:
            # TODO: torchmeta seems to be unmaintained and does not support newer versions of torch
            # maybe replace this with processing code from https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch
            from torchmeta.datasets.helpers import omniglot
            from torchmeta.utils.data import BatchMetaDataLoader
        except ModuleNotFoundError:
            print('You need to install torchmeta for this task.')
            print('  pip install torchmeta')
            sys.exit(1)
        
        dataset = omniglot("data", ways=self.ways, shots=shots, test_shots=test_shots,
                           shuffle=not test, meta_train=not test, meta_test=test, download=True)
        dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        iterator = iter(dataloader)

        def reset_fn(key):
            try:
                batch = iterator.next()
            except StopIteration:
                iterator = iter(dataloader)
                batch = iterator.next()
            # train_inputs shape: (batch_size, ways x shots, 1, 28, 28)
            # train_labels shape: (batch_size, ways x shots)
            train_inputs, train_labels = batch['train']
            train_inputs = jnp.transpose(jnp.array(train_inputs), (0,1,3,4,2))
            train_labels = jnp.array(train_labels.unsqueeze(-1))
            test_inputs, test_labels = batch['test']
            test_inputs = jnp.transpose(jnp.array(test_inputs), (0,1,3,4,2))
            test_labels = jnp.array(test_labels.unsqueeze(-1))
            # the shape of this State is meant for ___one___ of the members of the population
            return State(obs=(train_inputs, train_labels), test_inputs=test_inputs, test_labels=test_labels)
        
        # this vmap is for pop_size (so all members see the same state)
        self._reset_fn = jax.jit(jax.vmap(reset_fn)) 

        def step_fn(state, action):
            # state: state returned by reset_fn
            # action: predictions from ___one___ of the members in the population
            # should have the shape of [batch_size, ways x shots, ways (n_classes)]
            if test:
                reward = accuracy(action, state.test_labels)
            else:
                reward = -loss(action, state.test_labels)
            return state, reward, jnp.ones(())
        
        # this vmap is for pop_size (so all members output actions given the same state)
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
