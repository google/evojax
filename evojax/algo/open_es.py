import sys

import logging
from typing import Union
import numpy as np
import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


class OpenES(NEAlgorithm):
    """A wrapper around evosax's OpenAI Evolution Strategies.
    Implementation:
    https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/open_es.py
    Reference: Salimans et al. (2017) - https://arxiv.org/pdf/1703.03864.pdf

    NOTE: More details on the optimizer configuration can be found here
    https://github.com/RobertTLange/evosax/blob/main/evosax/utils/optimizer.py
    """

    def __init__(
        self,
        param_size: int,
        pop_size: int,
        optimizer: str = "adam",
        optimizer_config: dict = {
            "lrate_init": 0.01,  # Initial learning rate
            "lrate_decay": 0.999,  # Multiplicative decay factor
            "lrate_limit": 0.001,  # Smallest possible lrate
            "beta_1": 0.99,  # beta_1 Adam
            "beta_2": 0.999,  # beta_2 Adam
            "eps": 1e-8,  # eps constant Adam denominator
        },
        init_stdev: float = 0.01,
        decay_stdev: float = 0.999,
        limit_stdev: float = 0.001,
        w_decay: float = 0.0,
        seed: int = 0,
        logger: logging.Logger = None,
    ):
        """Initialization function.

        Args:
            param_size - Parameter size.
            pop_size - Population size.
            elite_ratio - Population elite fraction used for gradient estimate.
            optimizer - Optimizer name ("sgd", "adam", "rmsprop", "clipup").
            optimizer_config - Configuration of optimizer hyperparameters.
            init_stdev - Initial scale of Gaussian perturbation.
            decay_stdev - Multiplicative scale decay between tell iterations.
            limit_stdev - Smallest scale (clipping limit).
            w_decay - L2 weight regularization coefficient.
            seed - Random seed for parameters sampling.
            logger - Logger.
        """

        # Delayed importing of evosax

        if sys.version_info.minor < 7:
            print("evosax, which is needed byOpenES, requires python>=3.7")
            print("  please consider upgrading your Python version.")
            sys.exit(1)

        try:
            import evosax
        except ModuleNotFoundError:
            print("You need to install evosax for its OpenES implementation:")
            print("  pip install evosax")
            sys.exit(1)

        if logger is None:
            self.logger = create_logger(name="OpenES")
        else:
            self.logger = logger

        self.param_size = param_size
        self.pop_size = abs(pop_size)
        self.rand_key = jax.random.PRNGKey(seed=seed)

        # Instantiate evosax's Open ES strategy
        self.es = evosax.OpenES(
            popsize=pop_size,
            num_dims=param_size,
            opt_name=optimizer,
        )

        # Set hyperparameters according to provided inputs
        self.es_params = self.es.default_params
        for k, v in optimizer_config.items():
            self.es_params[k] = v
        self.es_params["sigma_init"] = init_stdev
        self.es_params["sigma_decay"] = decay_stdev
        self.es_params["sigma_limit"] = limit_stdev
        self.es_params["init_min"] = 0.0
        self.es_params["init_max"] = 0.0

        # Initialize the evolution strategy state
        self.rand_key, init_key = jax.random.split(self.rand_key)
        self.es_state = self.es.initialize(init_key, self.es_params)

        # By default evojax assumes maximization of fitness score!
        # Evosax, on the other hand, minimizes!
        self.fit_shaper = evosax.FitnessShaper(
            centered_rank=True, z_score=True, w_decay=w_decay, maximize=True
        )

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
