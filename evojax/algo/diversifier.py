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

"""Implementation of the Diversifier algorithm in JAX.

Diversifier is a JAX based meta-algorithm which is a generalization of CMA-ME
(see https://arxiv.org/pdf/1912.02400.pdf). It uses a MAP-Elites 
archive not for solution candidate generation, but only to modify
the fitness values told (via tell) to the wrapped algorithm. 
This modification changes the fitness ranking of the population
to favor exploration over exploitation. Tested with 
CR-FM-NES and CMA-ES, but other wrapped algorithms may work as well. 
Based on https://github.com/dietmarwo/fast-cma-es/blob/master/fcmaes/diversifier.py
(see https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/MapElites.adoc)

For Brax Ant CR-FM-NES-ME (wrapping CR-FM-NES), compared with MAP-Elites, reaches
a higher QD-score (sum of fitness values of all elites in the map) for high
iteration numbers. Use MAP-Elites instead for a low evaluation budget or if you
want to maximize the number of occupied niches. 
"""

import logging
import numpy as np

from typing import Union

import jax
import jax.numpy as jnp

from evojax.algo.base import QualityDiversityMethod, NEAlgorithm
from evojax.task.base import TaskState
from evojax.task.base import BDExtractor
from evojax.util import create_logger

from time import time

class Diversifier(QualityDiversityMethod):
    """The Diversifier meta algorithm."""

    def __init__(self,
                 solver: NEAlgorithm,
                 pop_size: int,
                 param_size: int,
                 bd_extractor: BDExtractor,
                 fitness_weight: float = 0,
                 seed: int = 0,
                 logger: logging.Logger = None):
        """Initialization function.

        Args:
            solver: wrapped solver, CR-FM-NES and CMA-ES work well. 
            pop_size - Population size.
            param_size - Parameter size.
            bd_extractor - A list of behavior descriptor extractors.
            fitness_weight - factor applied to the original fitness. 
                Should be in interval [0,1]. Higher value means:
                QD-score grows faster, but will stop growing earlier. 
                Choose fitness_weight = 1 if you are also interested in 
                a good global optimum. 
            seed - Random seed for parameters sampling.
            logger - Logging utility.
        """

        if logger is None:
            self._logger = create_logger(name=solver._logger.name + '-ME')
        else:
            self._logger = logger

        self.solver = solver
        self.pop_size = abs(pop_size)
        self.param_size = param_size
        self.fitness_weight = fitness_weight
        self.bd_names = [x[0] for x in bd_extractor.bd_spec]
        self.bd_n_bins = [x[1] for x in bd_extractor.bd_spec]
        self.params_lattice = jnp.zeros((np.prod(self.bd_n_bins), param_size))
        self.fitness_lattice = -float('Inf') * jnp.ones(np.prod(self.bd_n_bins))
        self.occupancy_lattice = jnp.zeros(
            np.prod(self.bd_n_bins), dtype=jnp.int32)
        self.population = None
        self.bin_idx = jnp.zeros(self.pop_size, dtype=jnp.int32)
        self.key = jax.random.PRNGKey(seed)

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
        self.population = self.solver.ask()
        return self.population

    def observe_bd(self, task_state: TaskState) -> None:
        self.bin_idx = self._get_bin_idx(task_state)

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:              
        lattice_fitness = self.fitness_lattice[self.bin_idx]
        is_empty = (lattice_fitness == -np.inf)
        # change -np.inf to minimal fitness to encourage visiting empty niches
        lattice_fitness = lattice_fitness.at[is_empty].set(jnp.amin(fitness)-1E-9)
        improvement = fitness - lattice_fitness
        to_tell = improvement if self.fitness_weight <= 0 else \
                  improvement + self.fitness_weight * fitness
        # tell the wrapped solver the modified fitness including improvement relative to the lattice
        self.solver.tell(to_tell)
        # update lattice  
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
