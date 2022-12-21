import sys

import logging
from typing import Union, Optional
import numpy as np
import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


class iAMaLGaM(NEAlgorithm):
    """A wrapper around evosax's iAMaLGaM.
    Implementation: https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/indep_iamalgam.py
    Reference: Bosman et al. (2013) - https://tinyurl.com/y9fcccx2
    """

    def __init__(
        self,
        param_size: int,
        pop_size: int,
        elite_ratio: float = 0.35,
        full_covariance: bool = False,
        eta_sigma: Optional[float] = None,
        eta_shift: Optional[float] = None,
        init_stdev: float = 0.01,
        decay_stdev: float = 0.999,
        limit_stdev: float = 0.001,
        w_decay: float = 0.0,
        seed: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialization function.

        Args:
            param_size - Parameter size.
            pop_size - Population size.
            elite_ratio - Population elite fraction used for mean update.
            full_covariance - Whether to estimate full covariance or only diag.
            eta_sigma - Lrate for covariance (use default if not provided).
            eta_shift - Lrate for mean shift (use default if not provided).
            init_stdev - Initial scale of Gaussian perturbation.
            decay_stdev - Multiplicative scale decay between tell iterations.
            limit_stdev - Smallest scale (clipping limit).
            w_decay - L2 weight regularization coefficient.
            seed - Random seed for parameters sampling.
            logger - Logger.
        """

        # Delayed importing of evosax

        if sys.version_info.minor < 7:
            print("evosax, which is needed by iAMaLGaM, requires python>=3.7")
            print("  please consider upgrading your Python version.")
            sys.exit(1)

        try:
            import evosax
        except ModuleNotFoundError:
            print("You need to install evosax for its iAMaLGaM:")
            print("  pip install evosax")
            sys.exit(1)

        # Set up object variables.

        if logger is None:
            self.logger = create_logger(name="iAMaLGaM")
        else:
            self.logger = logger

        self.param_size = param_size
        self.pop_size = abs(pop_size)
        self.rand_key = jax.random.PRNGKey(seed=seed)

        # Instantiate evosax's iAMaLGaM - choice between full cov & diagonal
        if full_covariance:
            self.es = evosax.Full_iAMaLGaM(
                popsize=pop_size, num_dims=param_size, elite_ratio=elite_ratio
            )
        else:
            self.es = evosax.Indep_iAMaLGaM(
                popsize=pop_size, num_dims=param_size, elite_ratio=elite_ratio
            )

        # Set hyperparameters according to provided inputs
        self.es_params = self.es.default_params.replace(
            sigma_init=init_stdev,
            sigma_decay=decay_stdev,
            sigma_limit=limit_stdev,
            init_min=0.0,
            init_max=0.0,
        )

        # Only replace learning rates for mean shift and sigma if provided!
        if eta_shift is not None:
            self.es_params = self.es_params.replace(eta_shift=eta_shift)
        if eta_sigma is not None:
            self.es_params = self.es_params.replace(eta_sigma=eta_sigma)

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
        return jnp.array(self.es_state.mean, copy=True)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self.es_state = self.es_state.replace(
            best_member=jnp.array(params, copy=True),
            mean=jnp.array(params, copy=True),
        )
