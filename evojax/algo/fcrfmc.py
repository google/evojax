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

"""This is a wrapper of the CR-FM-NES algorithm.
    See https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/crfmnes.cpp .
    Eigen based implementation of Fast Moving Natural Evolution Strategy 
    for High-Dimensional Problems (CR-FM-NES), see https://arxiv.org/abs/2201.11422 .
    Derived from https://github.com/nomuramasahir0/crfmnes .
"""

import sys

import logging
import numpy as np
import math
from typing import Union
from typing import Optional

import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger

class FCRFMC(NEAlgorithm):
    """A wrapper of CR-FM-NES Eigen version."""

    def __init__(self,
                 param_size: int,
                 pop_size: int,
                 init_params: Optional[Union[jnp.ndarray, np.ndarray]] = None,
                 init_stdev: float = 0.1,
                 seed: int = 0,
                 logger: logging.Logger = None):
        if logger is None:
            self.logger = create_logger(name='FCRFMC')
        else:
            self.logger = logger
        self.pop_size = pop_size
        
        if init_params is None:
            center = np.zeros(abs(param_size))
        else:
            center = init_params
        
        try:
            from numpy.random import MT19937, Generator
            from fcmaes import crfmnescpp

        except ModuleNotFoundError:
            print("You need to install fcmaes:")
            print("  pip install fcmaes --upgrade")
            sys.exit(1)
                 
        self.fcrfm = crfmnescpp.CRFMNES_C(param_size, None, center,
                init_stdev, pop_size, Generator(MT19937(seed)))    

        self.params = None
        self._best_params = None
        self.maxy = -math.inf
        
        self.jnp_array = jax.jit(jnp.array)
        self.jnp_stack = jax.jit(jnp.stack)
        
    def ask(self) -> jnp.ndarray:
        self.params = self.fcrfm.ask()
        return self.jnp_stack(self.params)

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        y = np.array(fitness)
        self.fcrfm.tell(-y)

        maxy = np.max(y)
        if self.maxy < maxy:
            self.maxy = maxy
            self._best_params = self.params[np.argmax(y)]
            
    @property
    def best_params(self) -> jnp.ndarray:
        return self.jnp_array(self._best_params)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self._best_params = np.array(params)
