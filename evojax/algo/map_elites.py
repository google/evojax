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

"""Implementation of the MAP-ELITES algorithm in JAX.

Ref: https://arxiv.org/pdf/1504.04909.pdf

In this implementation, we use a modified version iso+line emitter.
"""

import logging
import numpy as np
from typing import Optional
from typing import Union

import jax
import jax.numpy as jnp

from evojax.algo.base import QualityDiversityMethod
from evojax.task.base import TaskState
from evojax.task.base import BDExtractor
from evojax.util import create_logger


class MAPElites(QualityDiversityMethod):
    """The MAP-ELITES algorithm."""

    def __init__(self,
                 pop_size: int,
                 param_size: int,
                 bd_extractor: BDExtractor,
                 init_params: Optional[Union[jnp.ndarray, np.ndarray]] = None,
                 iso_sigma: float = 0.05,
                 line_sigma: float = 0.5,
                 seed: int = 0,
                 logger: logging.Logger = None):
        """Initialization function.

        Args:
            pop_size - Population size.
            param_size - Parameter size.
            bd_extractors - A list of behavior descriptor extractors.
            init_params - Initial parameters, all zeros if not given.
            iso_sigma - Standard deviation for Gaussian sampling for mutation.
            line_sigma - Parameter for cross over.
            seed - Random seed for parameters sampling.
            logger - Logging utility.
        """

        if logger is None:
            self._logger = create_logger(name='MAP_ELITES')
        else:
            self._logger = logger

        self.pop_size = abs(pop_size)
        self.param_size = param_size
        self.bd_names = [x[0] for x in bd_extractor.bd_spec]
        self.bd_n_bins = [x[1] for x in bd_extractor.bd_spec]
        self.params_lattice = jnp.zeros((np.prod(self.bd_n_bins), param_size))
        self.fitness_lattice = -float('Inf') * jnp.ones(np.prod(self.bd_n_bins))
        self.occupancy_lattice = jnp.zeros(
            np.prod(self.bd_n_bins), dtype=jnp.int32)
        self.population = None
        self.bin_idx = jnp.zeros(self.pop_size, dtype=jnp.int32)
        self.key = jax.random.PRNGKey(seed)

        init_param = (
            jnp.array(init_params)
            if init_params is not None else jnp.zeros(param_size))

        def sample_parents(key, occupancy, params):
            new_key, sample_key, mutate_key = jax.random.split(key, 3)
            parents = jax.lax.cond(
                occupancy.sum() > 0,
                lambda: jnp.take(params, axis=0, indices=jax.random.choice(
                    key=sample_key, a=jnp.arange(occupancy.size), replace=True,
                    p=occupancy / occupancy.sum(), shape=(2 * self.pop_size,))),
                lambda: init_param[None, :] + iso_sigma * jax.random.normal(
                    sample_key, shape=(2 * self.pop_size, self.param_size)),
            )
            return new_key, mutate_key, parents.reshape((2, self.pop_size, -1))
        self._sample_parents = jax.jit(sample_parents)

        def generate_population(parents, key):
            k1, k2 = jax.random.split(key, 2)
            return (parents[0] + iso_sigma *
                    jax.random.normal(key=k1, shape=parents[0].shape) +
                    # Uniform sampling instead of Gaussian.
                    jax.random.uniform(
                        key=k2, minval=0, maxval=line_sigma, shape=()) *
                    (parents[1] - parents[0]))
        self._gen_pop = jax.jit(generate_population)

        def get_bin_idx(task_state):
            bd_idx = [
                task_state.__dict__[name].astype(int) for name in self.bd_names]
            return jnp.ravel_multi_index(bd_idx, self.bd_n_bins, mode='clip')
        self._get_bin_idx = jax.jit(jax.vmap(get_bin_idx))

        def update_fitness_and_param(
                target_bin, bin_idx,
                fitness, fitness_lattice, param, param_lattice):
            best_ix = jnp.where(
                bin_idx == target_bin, fitness, fitness_lattice.min()).argmax()
            best_fitness = fitness[best_ix]
            new_fitness_lattice = jnp.where(
                best_fitness > fitness_lattice[target_bin],
                best_fitness, fitness_lattice[target_bin])
            new_param_lattice = jnp.where(
                best_fitness > fitness_lattice[target_bin],
                param[best_ix], param_lattice[target_bin])
            return new_fitness_lattice, new_param_lattice
        self._update_lattices = jax.jit(jax.vmap(
            update_fitness_and_param,
            in_axes=(0, None, None, None, None, None)))

    def ask(self) -> jnp.ndarray:
        self.key, mutate_key, parents = self._sample_parents(
            key=self.key,
            occupancy=self.occupancy_lattice,
            params=self.params_lattice)
        self.population = self._gen_pop(parents, mutate_key)
        return self.population

    def observe_bd(self, task_state: TaskState) -> None:
        self.bin_idx = self._get_bin_idx(task_state)

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        unique_bins = jnp.unique(self.bin_idx)
        fitness_lattice, params_lattice = self._update_lattices(
            unique_bins, self.bin_idx,
            fitness, self.fitness_lattice,
            self.population, self.params_lattice)
        self.occupancy_lattice = self.occupancy_lattice.at[unique_bins].set(1)
        self.fitness_lattice = self.fitness_lattice.at[unique_bins].set(
            fitness_lattice)
        self.params_lattice = self.params_lattice.at[unique_bins].set(
            params_lattice)

    @property
    def best_params(self) -> jnp.ndarray:
        ix = jnp.argmax(self.fitness_lattice, axis=0)
        return self.params_lattice[ix]
