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

import numpy as np
from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp


def normalize(obs: jnp.ndarray,
              obs_params: jnp.ndarray,
              obs_shape: Tuple,
              clip_value: float,
              std_min_value: float,
              std_max_value: float) -> jnp.ndarray:
    """Normalize the given observation."""

    obs_steps = obs_params[0]
    running_mean, running_var = jnp.split(obs_params[1:], 2)
    running_mean = running_mean.reshape(obs_shape)
    running_var = running_var.reshape(obs_shape)

    variance = running_var / (obs_steps + 1.0)
    variance = jnp.clip(variance, std_min_value, std_max_value)
    return jnp.clip(
        (obs - running_mean) / jnp.sqrt(variance), -clip_value, clip_value)


def update_obs_params(obs_buffer: jnp.ndarray,
                      obs_mask: jnp.ndarray,
                      obs_params: jnp.ndarray) -> jnp.ndarray:
    """Update observation normalization parameters."""

    obs_steps = obs_params[0]
    running_mean, running_var = jnp.split(obs_params[1:], 2)
    if obs_mask.ndim != obs_buffer.ndim:
        obs_mask = obs_mask.reshape(
            obs_mask.shape + (1,) * (obs_buffer.ndim - obs_mask.ndim))

    new_steps = jnp.sum(obs_mask)
    total_steps = obs_steps + new_steps

    input_to_old_mean = (obs_buffer - running_mean) * obs_mask
    mean_diff = jnp.sum(input_to_old_mean / total_steps, axis=(0, 1))
    new_mean = running_mean + mean_diff

    input_to_new_mean = (obs_buffer - new_mean) * obs_mask
    var_diff = jnp.sum(input_to_new_mean * input_to_old_mean, axis=(0, 1))
    new_var = running_var + var_diff

    return jnp.concatenate([jnp.ones(1) * total_steps, new_mean, new_var])


class ObsNormalizer(object):
    """Observation normalizer."""

    def __init__(self,
                 obs_shape: Tuple,
                 clip_value: float = 5.,
                 std_min_value: float = 1e-6,
                 std_max_value: float = 1e6,
                 dummy: bool = False):
        """Initialization.

        Args:
            obs_shape - Shape of the observations.
            std_min_value - Minimum standard deviation.
            std_max_value - Maximum standard deviation.
            dummy - Whether this is a dummy normalizer.
        """

        self._obs_shape = obs_shape
        self._obs_size = np.prod(obs_shape)
        self._std_min_value = std_min_value
        self._std_max_value = std_max_value
        self._clip_value = clip_value
        self.is_dummy = dummy

    @partial(jax.jit, static_argnums=(0,))
    def normalize_obs(self,
                      obs: jnp.ndarray,
                      obs_params: jnp.ndarray) -> jnp.ndarray:
        """Normalize the given observation.

        Args:
            obs - The observation to be normalized.
        Returns:
            Normalized observation.
        """

        if self.is_dummy:
            return obs
        else:
            return normalize(
                obs=obs,
                obs_params=obs_params,
                obs_shape=self._obs_shape,
                clip_value=self._clip_value,
                std_min_value=self._std_min_value,
                std_max_value=self._std_max_value)

    @partial(jax.jit, static_argnums=(0,))
    def update_normalization_params(self,
                                    obs_buffer: jnp.ndarray,
                                    obs_mask: jnp.ndarray,
                                    obs_params: jnp.ndarray) -> jnp.ndarray:
        """Update internal parameters."""

        if self.is_dummy:
            return jnp.zeros_like(obs_params)
        else:
            return update_obs_params(
                obs_buffer=obs_buffer,
                obs_mask=obs_mask,
                obs_params=obs_params,
            )

    @partial(jax.jit, static_argnums=(0,))
    def get_init_params(self) -> jnp.ndarray:
        return jnp.zeros(1 + self._obs_size * 2)
