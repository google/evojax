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

"""This is a wrapper of the CMA-ES algorithm.

This is for users who want to use CMA-ES before we release a pure JAX version.

CMA-ES paper: https://arxiv.org/abs/1604.00772
"""

import sys

import logging
import numpy as np
from typing import Union

import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


class CMA(NEAlgorithm):
    """A wrapper of CMA-ES."""

    def __init__(self,
                 param_size: int,
                 pop_size: int,
                 init_stdev: float = 0.1,
                 seed: int = 0,
                 logger: logging.Logger = None):
        if logger is None:
            self.logger = create_logger(name='CMA')
        else:
            self.logger = logger
        self.pop_size = pop_size

        try:
            import cma
        except ModuleNotFoundError:
            print("You need to install cma for its CMA-ES:")
            print("  pip install cma")
            sys.exit(1)

        self.cma = cma.CMAEvolutionStrategy(
            x0=np.zeros(param_size),
            sigma0=init_stdev,
            inopts={
                'popsize': pop_size,
                'seed': seed if seed > 0 else 42,
                'randn': np.random.randn,
            },
        )
        self.params = None
        self._best_params = None

        self.jnp_array = jax.jit(jnp.array)
        self.jnp_stack = jax.jit(jnp.stack)

    def ask(self) -> jnp.ndarray:
        self.params = self.cma.ask()
        return self.jnp_stack(self.params)

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        self.cma.tell(self.params, -np.array(fitness))
        self._best_params = np.array(self.cma.result.xfavorite)

    @property
    def best_params(self) -> jnp.ndarray:
        return self.jnp_array(self._best_params)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self._best_params = np.array(params)
        self.cma.x0 = self._best_params.copy()
