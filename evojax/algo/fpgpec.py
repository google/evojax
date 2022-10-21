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

"""This is a wrapper of the PGPE algorithm.
    See https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/pgpe.cpp .
    Eigen based implementation of PGPE see http://mediatum.ub.tum.de/doc/1099128/631352.pdf .
    Derived from https://github.com/google/evojax/blob/main/evojax/algo/pgpe.py .
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

from time import time

class FPGPEC(NEAlgorithm):
    """A wrapper of PGPE Eigen version."""

    def __init__(self,
            pop_size: int,
            param_size: int,
            init_params: Optional[Union[jnp.ndarray, np.ndarray]] = None,
            optimizer: Optional[str] = None,
            optimizer_config: Optional[dict] = None,
            center_learning_rate: float = 0.15,
            stdev_learning_rate: float = 0.1,
            init_stdev: Union[float, jnp.ndarray, np.ndarray] = 0.1,
            stdev_max_change: float = 0.2,
            solution_ranking: bool = False,
            seed: int = 0,
            logger: logging.Logger = None,
        ):
        if logger is None:
            self.logger = create_logger(name='FPGPEC')
        else:
            self.logger = logger
        self.pop_size = pop_size

        try:
            from numpy.random import MT19937, Generator
            from fcmaes import pgpecpp

        except ModuleNotFoundError:
            print("You need to install fcmaes:")
            print("  pip install fcmaes --upgrade")
            sys.exit(1)
        
        if not optimizer is None and optimizer != "adam":
            print("FPGPEC currently only supports adam optimizer")
            sys.exit(1)            
        
        if init_params is None:
            center = np.zeros(abs(param_size))
        else:
            center = init_params
        
        if isinstance(init_stdev, float):
            init_stdev = np.full(abs(param_size), abs(init_stdev))
        init_stdev = jnp.array(init_stdev)
        
        if optimizer_config is None:
            optimizer_config = {}
        decay_coef = optimizer_config.get("center_lr_decay_coef", 1.0)
        lr_decay_steps = optimizer_config.get(
            "center_lr_decay_steps", 1000
        )
        bounds = None # Bounds([-3]*param_size,[3]*param_size)         
        self.fcrfm = pgpecpp.PGPE_C(param_size, bounds, center,
                init_stdev, pop_size, Generator(MT19937(seed)), 
                lr_decay_steps = lr_decay_steps,
                use_ranking = solution_ranking, 
                center_learning_rate = center_learning_rate,
                stdev_learning_rate = stdev_learning_rate, 
                stdev_max_change = stdev_max_change, 
                b1 = optimizer_config.get("beta1", 0.9),
                b2 = optimizer_config.get("beta2", 0.999), 
                eps = optimizer_config.get("epsilon", 1e-8), 
                decay_coef = decay_coef,                
                )    

        self.params = None
        self._best_params = None
        self.maxy = -math.inf
        
        self.jnp_array = jax.jit(jnp.array)
        self.jnp_stack = jax.jit(jnp.stack)
        
        self.t0 = time()
        self.evals = 0

    def ask(self) -> jnp.ndarray:
        self.params = self.fcrfm.ask()
        return self.jnp_stack(self.params)

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        y = np.array(fitness)
        self.fcrfm.tell(-y)

        maxy = np.max(y)
        self.evals += self.pop_size
        if self.maxy < maxy:
            self._best_params = self.params[np.argmax(y)]   
            self.maxy = maxy
            
    @property
    def best_params(self) -> jnp.ndarray:
        return self.jnp_array(self._best_params)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        print("best_params called", params[0], self._best_params[0])
        self._best_params = np.array(params)
