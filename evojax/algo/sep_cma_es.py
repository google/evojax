import sys

import logging
from typing import Union
import numpy as np
import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


class Sep_CMA_ES(NEAlgorithm):
    """A wrapper around evosax's Sep-CMA-ES.
    Implementation: https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/sep_cma_es.py
    Reference: Ros & Hansen (2008) - https://hal.inria.fr/inria-00287367/document
    """

    def __init__(
        self,
        param_size: int,
        pop_size: int,
        elite_ratio: float = 0.5,
        init_stdev: float = 0.1,
        w_decay: float = 0.0,
        seed: int = 0,
        logger: logging.Logger = None,
    ):
        """Initialization function.

        Args:
            param_size - Parameter size.
            pop_size - Population size.
            elite_ratio - Population elite fraction used for gradient estimate.
            init_stdev - Initial scale of istropic part of covariance.
            w_decay - L2 weight regularization coefficient.
            seed - Random seed for parameters sampling.
            logger - Logger.
        """

        # Delayed importing of evosax

        if sys.version_info.minor < 7:
            print("evosax, which is needed by Sep-CMA-ES, requires python>=3.7")
            print("  please consider upgrading your Python version.")
            sys.exit(1)

        try:
            import evosax
        except ModuleNotFoundError:
            print("You need to install evosax for its Sep-CMA-ES:")
            print("  pip install evosax")
            sys.exit(1)

        # Set up object variables.

        if logger is None:
            self.logger = create_logger(name="Sep_CMA_ES")
        else:
            self.logger = logger

        self.param_size = param_size
        self.pop_size = abs(pop_size)
        self.elite_ratio = elite_ratio
        self.rand_key = jax.random.PRNGKey(seed=seed)

        # Instantiate evosax's ARS strategy
        self.es = evosax.Sep_CMA_ES(
            popsize=pop_size,
            num_dims=param_size,
            elite_ratio=elite_ratio,
        )

        # Set hyperparameters according to provided inputs
        self.es_params = self.es.default_params
        self.es_params["sigma_init"] = init_stdev

        # Initialize the evolution strategy state
        self.rand_key, init_key = jax.random.split(self.rand_key)
        self.es_state = self.es.initialize(init_key, self.es_params)

        # By default evojax assumes maximization of fitness score!
        # Evosax, on the other hand, minimizes!
        self.fit_shaper = evosax.FitnessShaper(w_decay=w_decay, maximize=True)

    def ask(self) -> jnp.ndarray:
        self.rand_key, ask_key = jax.random.split(self.rand_key)
        self.params, self.es_state = self.es.ask(
            ask_key, self.es_state, self.es_params
        )
        return self.params

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        # Reshape fitness to conform with evosax minimization
        fit_re = self.fit_shaper.apply(self.params, fitness)
        self.es_state = self.es.tell(
            self.params, fit_re, self.es_state, self.es_params
        )

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.array(self.es_state["mean"], copy=True)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self.es_state["best_member"] = jnp.array(params, copy=True)
        self.es_state["mean"] = jnp.array(params, copy=True)
