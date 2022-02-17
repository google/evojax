""" Implementation of CMA-ES in JAX.

Ref: https://github.com/CyberAgentAILab/cmaes/blob/main/cmaes/_cma.py
"""

from __future__ import annotations

import functools
import logging
import math

import jax
import numpy as np
from evojax.algo.base import NEAlgorithm, process_scores
from evojax.util import create_logger
from jax import numpy as jnp

EPS = 1e-8
MAX = 1e32


class CMAES_CyberAgent(NEAlgorithm):
    """CMA-ES

    Ref: https://arxiv.org/abs/1604.00772
    Ref: https://github.com/CyberAgentAILab/cmaes/
    """

    def __init__(
        self,
        pop_size: int = None,
        param_size: int = None,
        init_params: np.ndarray | jnp.ndarray = None,
        init_stdev: float = 0.1,
        init_cov: np.ndarray | jnp.ndarray = None,
        solution_ranking: bool = True,
        seed: int = 0,
        logger: logging.Logger = None,
    ):
        """Initialization function. Equation numbers are from Hansen's tutorial.

        Args:
            pop_size - Population size, recommended population size if not given.
            param_size - Parameter size.
            init_params - Initial parameters, all zeros if not given.
            init_stdev - Initial sigma value.
            init_cov - Intial covariance matrix, identity if not given.
            solution_ranking - Should we treat the fitness as rankings or not.
            seed - Random seed for parameters sampling.
        """

        assert init_stdev > 0
        if logger is None:
            self._logger = create_logger("cmaes")
        else:
            self._logger = logger

        if init_params is None:
            init_params = jnp.zeros(param_size)
        mean = init_params

        if init_cov is None:
            self._C = jnp.eye(param_size)
        else:
            self._C = init_cov

        if pop_size is None:
            # eq (48)
            pop_size = 4 + math.floor(3 * math.log(param_size))
            self._logger.info(f"population size (pop_size) is set to {pop_size} (recommended size)")

        mu = pop_size // 2

        # eq (49)
        weights_prime = jnp.array([math.log((pop_size + 1) / 2) - math.log1p(i) for i in range(pop_size)])
        mu_eff = (jnp.sum(weights_prime[:mu]) ** 2) / jnp.sum(weights_prime[:mu] ** 2)
        mu_eff_minus = (jnp.sum(weights_prime[mu:]) ** 2) / jnp.sum(weights_prime[mu:] ** 2)

        # learning rate for rank-one update eq (57)
        alpha_cov = 2
        c_1 = alpha_cov / ((param_size + 1.3) ** 2 + mu_eff)

        # learning rate for rank-mu update # eq (58)
        c_mu = min(
            1 - c_1 - 1e-8,
            alpha_cov * (mu_eff - 2 + 1 / mu_eff) / ((param_size + 2) ** 2 + alpha_cov * mu_eff / 2),
        )

        assert c_1 <= 1 - c_mu
        assert c_mu <= 1 - c_1

        min_alpha = min(
            1 + c_1 / c_mu,  # eq (50)
            1 + (2 * mu_eff_minus) / (mu_eff + 2),  # eq (51)
            (1 - c_1 - c_mu) / (param_size * c_mu),  # eq (52)
        )

        # eq (53)
        positive_sum = jnp.sum(weights_prime[weights_prime > 0])
        negative_sum = jnp.sum(jnp.abs(weights_prime[weights_prime < 0]))
        weights = jnp.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            min_alpha / negative_sum * weights_prime,
        )
        c_m = 1  # eq (54)

        # learning rate for the cumulation for the step-size control, eq (55)
        c_sigma = (mu_eff + 2) / (param_size + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (param_size + 1)) - 1) + c_sigma
        assert c_sigma < 1

        # learning rate for cumulation for the rank-one update, eq (56)
        c_c = (4 + mu_eff / param_size) / (param_size + 4 + 2 * mu_eff / param_size)
        assert c_c <= 1

        self._n_dim = param_size
        self.pop_size = pop_size
        self._mu = mu
        self._mu_eff = mu_eff
        self._c_c = c_c
        self._c_1 = c_1
        self._c_mu = c_mu
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma
        self._c_m = c_m

        # approx of E||N(0, I)||
        self._chi_n = math.sqrt(param_size) * (1 - (1 / (4 * param_size)) + 1 / (21 * param_size**2))

        self._weights = weights

        # path
        self._p_sigma = jnp.zeros(param_size)
        self._pc = jnp.zeros(param_size)

        self._mean = mean
        self._sigma = init_stdev
        self._D = None
        self._B = None
        self._solutions = None

        self._t = 0
        self._solution_ranking = solution_ranking
        self._key = jax.random.PRNGKey(seed=seed)

    def _eigen_decomposition(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self._B is None or self._D is None:
            self._C, self._B, self._D = _eigen_decomposition(self._C)

        return self._B, self._D

    def ask(self) -> jnp.ndarray:
        # resampling is skipped in this implementation
        # see cmaes for more details
        B, D = self._eigen_decomposition()
        self._key, key = jax.random.split(self._key)
        z = jax.random.normal(key, (self.pop_size, self._n_dim))
        self._solutions = _ask_impl(z, B, D, self._mean, self._sigma)
        return self._solutions

    def tell(self, fitness: np.ndarray | jnp.ndarray) -> None:

        fitness_scores = process_scores(fitness, self._solution_ranking)
        self._t += 1
        B, D = self._eigen_decomposition()
        self._B, self._D = None, None

        # highest score, ..., lowest score
        idx = jnp.argsort(-fitness_scores)
        x_k = self._solutions[idx]
        y_k = (x_k - self._mean) / self._sigma

        # selection and recombination
        y_w = jnp.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)  # eq (41)
        self._mean += self._c_m * self._sigma * y_w  # eq (42)

        # step-size control
        C_2 = B.dot(jnp.diag(1 / D)).dot(B.T)  # C^{-0.5}
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + jnp.sqrt(  # eq (43)
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * C_2.dot(y_w)

        norm_p_sigma = jnp.linalg.norm(self._p_sigma)
        self._sigma *= jnp.minimum(
            jnp.exp(self._c_sigma / self._d_sigma * (norm_p_sigma / self._chi_n - 1)),
            MAX,
        )  # eq (44)

        # covariance matrix adaptation (p. 28)
        h_sigma_cond_left = norm_p_sigma / jnp.sqrt((1 - (1 - self._c_sigma) ** (2 * (self._t + 1))))
        h_sigma_cond_right = (1.4 + 2 / (self._n_dim + 1)) * self._chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0

        self._pc = (1 - self._c_c) * self._pc + h_sigma * jnp.sqrt(
            self._c_c * (2 - self._c_c) * self._mu_eff
        ) * y_w  # eq (45)

        w_io = self._weights * jnp.where(
            self._weights >= 0,
            1,
            self._n_dim / (jnp.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + EPS),
        )  # eq (46)

        delta_h_sigma = (1 - h_sigma) * self._c_c * (2 - self._c_c)
        assert delta_h_sigma <= 1

        rank_one = jnp.outer(self._pc, self._pc)
        rank_mu = jnp.sum(jnp.array([w * jnp.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0)
        self._C = (
            (1 + self._c_1 * delta_h_sigma - self._c_1 - self._c_mu * jnp.sum(self._weights)) * self._C
            + self._c_1 * rank_one
            + self._c_mu * rank_mu
        )  # eq (47)

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.array(self._mean, copy=True)

    @best_params.setter
    def best_params(self, params: np.ndarray | jnp.ndarray) -> None:
        self._mean = jnp.array(params, copy=True)


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, None, None, None, None))
def _ask_impl(z, b, d, mean, sigma) -> jnp.ndarray:
    y = b.dot(jnp.diag(d)).dot(z)  # ~N(0, C)
    return mean + sigma * y


@jax.jit
def _eigen_decomposition(c):
    c = (c + c.T) / 2
    d2, b = jnp.linalg.eigh(c)
    d = jnp.where(d2 < 0, EPS, d2)
    return b.dot(jnp.diag(d)).dot(b.T), b, jnp.sqrt(d)
