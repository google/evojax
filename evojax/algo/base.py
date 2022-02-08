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

from abc import ABC
from abc import abstractmethod
from typing import Union
import numpy as np
import jax.numpy as jnp


class NEAlgorithm(ABC):
    """Interface of all Neuro-evolution algorithms in EvoJAX."""

    pop_size: int

    @abstractmethod
    def ask(self) -> jnp.ndarray:
        """Ask the algorithm for a population of parameters.

        Returns
            A Jax array of shape (population_size, param_size).
        """
        raise NotImplementedError()

    @abstractmethod
    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        """Report the fitness of the population to the algorithm.

        Args:
            fitness - The fitness scores array.
        """
        raise NotImplementedError()

    @property
    def best_params(self) -> jnp.ndarray:
        raise NotImplementedError()

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        raise NotImplementedError()
