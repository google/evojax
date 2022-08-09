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
from flax import linen as nn

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn


class CNN(nn.Module):
    """CNN for MNIST."""

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME', name='conv1')
        self.bn1 = nn.BatchNorm(32, momentum=0.1, use_running_average=True, name='bn1')

        self.conv2 = nn.Conv(features=48, kernel_size=(3, 3), padding='SAME', name='conv2')
        self.bn2 = nn.BatchNorm(48, momentum=0.1, use_running_average=True, name='bn2')

        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', name='conv3')
        self.bn3 = nn.BatchNorm(64, momentum=0.1, use_running_average=True, name='bn3')

        self.conv4 = nn.Conv(features=80, kernel_size=(3, 3), padding='SAME', name='conv4')
        self.bn4 = nn.BatchNorm(80, momentum=0.1, use_running_average=True, name='bn4')

        self.conv5 = nn.Conv(features=96, kernel_size=(3, 3), padding='SAME', name='conv5')
        self.bn5 = nn.BatchNorm(96, momentum=0.1, use_running_average=True, name='bn5')

        self.conv6 = nn.Conv(features=112, kernel_size=(3, 3), padding='SAME', name='conv6')
        self.bn6 = nn.BatchNorm(112, momentum=0.1, use_running_average=True, name='bn6')

        self.conv7 = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME', name='conv7')
        self.bn7 = nn.BatchNorm(128, momentum=0.1, use_running_average=True, name='bn7')

        self.conv8 = nn.Conv(features=144, kernel_size=(3, 3), padding='SAME', name='conv8')
        self.bn8 = nn.BatchNorm(144, momentum=0.1, use_running_average=True, name='bn8')

        self.conv9 = nn.Conv(features=160, kernel_size=(3, 3), padding='SAME', name='conv9')
        self.bn9 = nn.BatchNorm(160, momentum=0.1, use_running_average=True, name='bn9')

        self.conv10 = nn.Conv(features=176, kernel_size=(3, 3), padding='SAME', name='conv10')
        self.bn10 = nn.BatchNorm(176, momentum=0.1, use_running_average=True, name='bn10')

        self.linear = nn.Dense(10, name='linear')

    @nn.compact
    def __call__(self, x):

        # x = nn.relu(self.bn1(self.conv1(x)))
        # x = nn.relu(self.bn2(self.conv2(x)))
        # x = nn.relu(self.bn3(self.conv3(x)))
        # x = nn.relu(self.bn4(self.conv4(x)))
        # x = nn.relu(self.bn5(self.conv5(x)))
        # x = nn.relu(self.bn6(self.conv6(x)))
        # x = nn.relu(self.bn7(self.conv7(x)))
        # x = nn.relu(self.bn8(self.conv8(x)))
        # x = nn.relu(self.bn9(self.conv9(x)))
        # x = nn.relu(self.bn10(self.conv10(x)))

        for i in range(1, 11):
            x = nn.relu(getattr(self, f'bn{i}')(getattr(self, f'conv{i}')(x)))

        x = x.reshape((x.shape[0], -1))  # flatten

        x = self.linear(x)
        x = nn.log_softmax(x)
        return x


class Mask(nn.Module):
    """Mask network for MNIST."""
    def __init__(self, base_model):
        super(Mask, self).__init__()

        in_features = base_model.linear.in_features
        out_features = base_model.linear.out_features
        self.mask_size = in_features * out_features

    @nn.compact
    def __call__(self, x, round_output=True):
        x = nn.Dense(features=10)(x)
        x = nn.relu(x)
        x = nn.Dense(features=100)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.mask_size)(x)
        x = nn.sigmoid(x)
        if round_output:
            x = jnp.round(x)
        return x


class MaskPolicy(PolicyNetwork):
    """A dense neural network for masking the MNIST classification task."""

    def __init__(self, logger: logging.Logger = None, mnist_model=None):
        if logger is None:
            self._logger = create_logger('ConvNetPolicy')
        else:
            self._logger = logger

        model = Mask(mnist_model)
        params = model.init(random.PRNGKey(0), jnp.zeros([1, 1]))
        self.num_params, format_params_fn = get_params_format_fn(params)
        self._logger.info(f'Mask.num_params = {self.num_params}')
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states
