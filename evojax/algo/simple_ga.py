import logging
import numpy as np
from typing import Union
from typing import Tuple

import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


class SimpleGA(NEAlgorithm):
    """A simple genetic algorithm implementing truncation selection."""

    def __init__(self,
                 param_size: int,
                 pop_size: int,
                 truncation_divisor: int = 2,
                 sigma: float = 0.01,
                 seed: int = 0,
                 logger: logging.Logger = None):
        """Initialization function.

        Args:
            param_size - Parameter size.
            pop_size - Population size.
            truncation_divisor - Number by which the population is truncated
                                 every iteration.
            sigma - Variance of normal distribution for parameter perturbation
            seed - Random seed for parameters sampling.
            logger - Logger
        """

        if logger is None:
            self.logger = create_logger(name='SimpleGA')
        else:
            self.logger = logger

        self.param_size = param_size

        self.pop_size = abs(pop_size)
        self.truncation_divisor = abs(truncation_divisor)

        if self.pop_size % 2 == 1:
            self.pop_size += 1
            self.logger.info(
                'Population size should be an even number, set to {}'.format(
                    self.pop_size))

        if self.pop_size % self.truncation_divisor != 0:
            self.truncation_divisor = 2
            self.logger.info(
                'Population size must be a multiple of truncation divisor, \
                 set to {}'.format(self.truncation_divisor))

        self.truncation = self.pop_size // self.truncation_divisor
        self.sigma = sigma

        self.params = jnp.zeros((pop_size, param_size))
        self._best_params = None

        self.rand_key = jax.random.PRNGKey(seed=seed)

        self.jnp_array = jax.jit(jnp.array)

        def ask_fn(key: jnp.ndarray,
                   params: Union[np.ndarray,
                                 jnp.ndarray]) -> Tuple[jnp.ndarray,
                                                        Union[np.ndarray,
                                                              jnp.ndarray]]:

            next_key, sample_key = jax.random.split(key=key, num=2)

            perturbations = jax.random.normal(key=sample_key,
                                              shape=(self.pop_size,
                                                     self.param_size))

            return next_key, params + perturbations * self.sigma

        self.ask_fn = jax.jit(ask_fn)

        def tell_fn(fitness: Union[np.ndarray,
                                   jnp.ndarray],
                    params: Union[np.ndarray,
                                  jnp.ndarray]) -> Union[np.ndarray,
                                                         jnp.ndarray]:

            params = params[fitness.argsort(axis=0)]
            params = params[-self.truncation:].repeat(self.truncation_divisor,
                                                      axis=0)
            return params

        self.tell_fn = jax.jit(tell_fn)

    def ask(self) -> jnp.ndarray:
        self.rand_key, self.params = self.ask_fn(self.rand_key, self.params)
        return self.params

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        self.params = self.tell_fn(fitness, self.params)
        self._best_params = self.params[-1]

    @property
    def best_params(self) -> jnp.ndarray:
        return self.jnp_array(self._best_params)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self.params = jnp.repeat(params[None, :], self.pop_size, axis=0)
        self._best_params = jnp.array(params, copy=True)
