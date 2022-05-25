"""
**Experimental** CMA-ES optimizer using JAX backend.

Code in this file is an adaption from <https://github.com/CyberAgentAILab/cmaes/blob/main/cmaes/_cma.py>
which is a faithful implementation of CMA-ES described in <https://arxiv.org/abs/1604.00772>.

The adaption mainly replaces the Numpy backend with JAX one, and introduces several adjustments to facilitate
efficient computation in JAX:
- Use JAX's random paradigm
- Disable pickling funcionality (`__getstate__()` and `__setstate__()`) for now.
- Enable paralling sampling, completly change the logic of `ask()`
- Adjust init parameters to confer with existing API.
- In `tell()`, change the logic of reporting parameters and value together.

This is still an experimental implementation and has not been well-tested (yet).
"""

from functools import partial
import logging
import math
from typing import Optional

import jax
from jax import numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32

USE_JITTED_EIGEN = True

class CMA_ES_JAX(NEAlgorithm):
    """CMA-ES stochastic optimizer class with ask-and-tell interface, using JAX backend.

    Args:
        param_size:
            A parameter size.
        pop_size:
            A population size (optional).
            If not specified, a proper value will be inferred from param_size.
        mean:
            Initial mean vector of multi-variate gaussian distributions (optional).
            If specified, it must have a dimension of (param_size, ).
            If not specified, a default value of all-zero array will be used.
        init_stdev:
            Initial standard deviation of covariance matrix (optional).
            If not specified, a default value of 0.1 will be used.
        seed:
            A seed number (optional).
        bounds:
            Lower and upper domain boundaries for each parameter (optional).
        n_max_resampling:
            A maximum number of resampling parameters (optional).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.
            Only used for bounded optimization.
            If not specified, a default value of 100 will be used.
        cov:
            A covariance matrix (optional).
        logger:
            A logging.Logger instance (optional).
            If not specified, a new one will be created.
    """

    def __init__(
        self,
        param_size: int,
        pop_size: Optional[int] = None,
        mean: Optional[jnp.ndarray] = None,
        init_stdev: Optional[float] = 0.1,
        seed: Optional[int] = 0,
        bounds: Optional[jnp.ndarray] = None,
        n_max_resampling: int = 100,
        cov: Optional[jnp.ndarray] = None,
        logger: logging.Logger = None,
    ):
        if mean is None:
            mean = jnp.zeros(param_size)
        else:
            mean = ensure_jnp(mean)
            assert mean.shape == (param_size,), \
                f"Both params_size and mean are specified." \
                f"In this case,  mean (whose shape is {mean.shape}) must have a dimension of (param_size, )" \
                f" (i.e. {(param_size, )}), which is not true."
        mean = ensure_jnp(mean)
        assert jnp.all(
            jnp.abs(mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

        n_dim = len(mean)
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        assert init_stdev > 0, "init_stdev must be non-zero positive value"
        sigma = init_stdev
        sigma = ensure_jnp(sigma)

        population_size = pop_size
        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))  # (eq. 48)
        assert population_size > 0, "popsize must be non-zero positive value."

        mu = population_size // 2

        # (eq.49)
        weights_prime = jnp.array(
            [
                math.log((population_size + 1) / 2) - math.log(i + 1)
                for i in range(population_size)
            ]
        )
        mu_eff = (jnp.sum(weights_prime[:mu]) ** 2) / jnp.sum(weights_prime[:mu] ** 2)
        mu_eff_minus = (jnp.sum(weights_prime[mu:]) ** 2) / jnp.sum(weights_prime[mu:] ** 2)

        # learning rate for the rank-one update
        alpha_cov = 2.0
        c1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
        assert isinstance(c1, jnp.ndarray)
        # learning rate for the rank-μ update
        cmu = min(
            1 - c1 - 1e-8,  # 1e-8 is for large popsize.
            alpha_cov
            * (mu_eff - 2 + 1 / mu_eff)
            / ((n_dim + 2) ** 2 + alpha_cov * mu_eff / 2),
        )
        assert c1 <= 1 - cmu, "invalid learning rate for the rank-one update"
        assert cmu <= 1 - c1, "invalid learning rate for the rank-μ update"

        min_alpha = min(
            1 + c1 / cmu,  # eq.50
            1 + (2 * mu_eff_minus) / (mu_eff + 2),  # eq.51
            (1 - c1 - cmu) / (n_dim * cmu),  # eq.52
        )

        # (eq.53)
        positive_sum = jnp.sum(weights_prime[weights_prime > 0])
        negative_sum = jnp.sum(jnp.abs(weights_prime[weights_prime < 0]))
        weights = jnp.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            min_alpha / negative_sum * weights_prime,
        )
        cm = 1  # (eq. 54)

        # learning rate for the cumulation for the step-size control (eq.55)
        c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)
        d_sigma = 1 + 2 * \
            max(0, math.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + c_sigma
        assert (
            c_sigma < 1
        ), "invalid learning rate for cumulation for the step-size control"

        # learning rate for cumulation for the rank-one update (eq.56)
        cc = (4 + mu_eff / n_dim) / (n_dim + 4 + 2 * mu_eff / n_dim)
        assert cc <= 1, "invalid learning rate for cumulation for the rank-one update"

        self._n_dim = n_dim
        # self.pop_size is expected by EvoJAX trainer.
        self._popsize = self.pop_size = population_size
        self._mu = mu
        self._mu_eff = mu_eff

        self._cc = cc
        self._c1 = c1
        self._cmu = cmu
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma
        self._cm = cm

        # E||N(0, I)|| (p.28)
        self._chi_n = math.sqrt(self._n_dim) * (
            1.0 - (1.0 / (4.0 * self._n_dim)) +
            1.0 / (21.0 * (self._n_dim ** 2))
        )

        self._weights = weights

        # evolution path
        self._p_sigma = jnp.zeros(n_dim)
        self._pc = jnp.zeros(n_dim)

        self._mean = mean

        if cov is None:
            self._C = jnp.eye(n_dim)
        else:
            assert cov.shape == (n_dim, n_dim), "Invalid shape of covariance matrix"
            self._C = cov

        self._sigma = sigma
        self._D: Optional[jnp.ndarray] = None
        self._B: Optional[jnp.ndarray] = None

        # bounds contains low and high of each parameter.
        assert bounds is None or _is_valid_bounds(bounds, mean), "invalid bounds"
        self._bounds = bounds
        self._n_max_resampling = n_max_resampling

        self._g = 0

        self._key = jax.random.PRNGKey(seed)

        # Termination criteria
        self._tolx = 1e-12 * sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * n_dim / population_size)
        self._funhist_values = jnp.empty(self._funhist_term * 2)

        # Saved latest solutions
        self._latest_solutions = None

        # Logger
        if logger is None:
            # Change this name accordingly
            self.logger = create_logger(name="CMA")
        else:
            self.logger = logger

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._n_dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    def set_bounds(self, bounds: Optional[jnp.ndarray]) -> None:
        """Update boundary constraints"""
        assert bounds is None or _is_valid_bounds(bounds, self._mean), "invalid bounds"
        self._bounds = bounds

    def _eigen_decomposition(self) -> jnp.ndarray:
        if self._B is not None and self._D is not None:
            return self._B, self._D

        if USE_JITTED_EIGEN:
            _B, _D, _C = _eigen_decomposition_core_jitted(self._C)
            self._B, self._D, self._C = _B, _D, _C
            B, D = _B, _D
        else:
            self._C = (self._C + self._C.T) / 2
            D2, B = jnp.linalg.eigh(self._C)
            D = jnp.sqrt(jnp.where(D2 < 0, _EPS, D2))
            self._C = jnp.dot(jnp.dot(B, jnp.diag(D ** 2)), B.T)
            self._B, self._D = B, D
        return B, D

    def _ask(self, n_samples) -> jnp.ndarray:
        """Real implementaiton of ask, which samples multiple parameters in parallel.."""

        B, D = self._eigen_decomposition()
        n_dim, mean, sigma = self._n_dim, self._mean, self._sigma
        bounds = self._bounds
        mask = jnp.zeros(shape=(n_samples,), dtype=bool)

        if self._bounds is None:  # We accept any sampled solutions
            self._key, subkey = jax.random.split(self._key)
            subkey = jax.random.split(subkey, n_samples)
            x = _v_masked_sample_solution(mask, subkey, B, D, n_dim, mean, sigma)
            return x
        else:  # This means we have valid bounds to respect through rejection sampling.
            for i in range(0, self._n_max_resampling):
                self._key, subkey = jax.random.split(self._key)
                subkey = jax.random.split(subkey, n_samples)
                if i == 0:
                    x = _v_masked_sample_solution(mask, subkey, B, D, n_dim, mean, sigma)
                else:
                    x = (
                        _v_masked_sample_solution(mask, subkey, B, D, n_dim, mean, sigma) * (1 - jnp.expand_dims(mask, -1)) +
                        x * jnp.expand_dims(mask, -1)
                    )
                mask = _v_is_feasible(x, bounds)
                if jnp.all(mask):
                    return x
            x = _v_masked_sample_solution(mask, subkey, B, D, n_dim, mean, sigma)
            x = _v_repair_infeasible_params(x, bounds)
            return x

    def ask(self, n_samples: int = None) -> jnp.ndarray:
        """A wrapper of _ask, which handles optional n_samples and saves latest samples."""

        if n_samples is None:  # by default, do self._popsize samples.
            n_samples = self._popsize

        x = self._ask(n_samples)
        self._latest_solutions = x
        return x

    def tell(self, fitness: jnp.ndarray, solutions: Optional[jnp.ndarray] = None) -> None:
        """Tell evaluation values as fitness."""

        if solutions is None:
            assert self._latest_solutions is not None, \
                "`soltuions` is not given, expecting using latest samples but this was not done."
            assert self._latest_solutions.shape[0] == self._popsize, \
                f"Latest samples (shape={self._latest_solutions.shape}) not having popsize-length ({self._popsize})."
            solutions = self._latest_solutions
        else:
            solutions = ensure_jnp(solutions)
            assert solutions.shape[0] == self._popsize, "Given solutions must have popsize-length, which is not ture."

        for s in solutions:
            assert jnp.all(
                jnp.abs(s[0]) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

        # We want maximization, while the following logics is for minimimzation.
        # Handle this calse by simply revert fitness
        fitness = - fitness
        self._g += 1

        fitness = ensure_jnp(fitness)
        ranking = jnp.argsort(fitness, axis=0)

        # Stores 'best' and 'worst' values of the
        # last 'self._funhist_term' generations.
        funhist_idx = 2 * (self.generation % self._funhist_term)
        self._funhist_values = self._funhist_values.at[funhist_idx].set(fitness[ranking[0]])
        self._funhist_values = self._funhist_values.at[funhist_idx+1].set(fitness[ranking[-1]])

        sorted_solutions = solutions[ranking]

        # Sample new population of search_points, for k=1, ..., popsize
        B, D = self._eigen_decomposition()
        self._B, self._D = None, None

        x_k = jnp.array(sorted_solutions)  # ~ N(m, σ^2 C)
        y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)

        # Selection and recombination
        y_w = jnp.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)  # eq.41
        self._mean += self._cm * self._sigma * y_w

        # Step-size control
        C_2 = B.dot(jnp.diag(1 / D)).dot(B.T)  # C^(-1/2) = B D^(-1) B^T
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * C_2.dot(y_w)

        norm_p_sigma = jnp.linalg.norm(self._p_sigma)
        self._sigma *= jnp.exp(
            (self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1)
        )
        self._sigma = min(self._sigma, _SIGMA_MAX)

        # Covariance matrix adaption
        h_sigma_cond_left = norm_p_sigma / math.sqrt(
            1 - (1 - self._c_sigma) ** (2 * (self._g + 1))
        )
        h_sigma_cond_right = (1.4 + 2 / (self._n_dim + 1)) * self._chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)

        # (eq.45)
        self._pc = (1 - self._cc) * self._pc + h_sigma * math.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * y_w

        # (eq.46)
        w_io = self._weights * jnp.where(
            self._weights >= 0,
            1,
            self._n_dim / (jnp.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + _EPS),
        )

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)  # (p.28)
        assert delta_h_sigma <= 1

        # (eq.47)
        rank_one = jnp.outer(self._pc, self._pc)

        # This way of computing rank_mu can lead to OOM:
        # rank_mu = jnp.sum(
        #    jnp.array([w * jnp.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0
        # )
        # Try another way of computing rank_mu
        rank_mu = jnp.zeros_like(self._C)
        for w, y in zip(w_io, y_k):
            rank_mu = rank_mu + w * jnp.outer(y, y)

        self._C = (
            (
                1
                + self._c1 * delta_h_sigma
                - self._c1
                - self._cmu * jnp.sum(self._weights)
            )
            * self._C
            + self._c1 * rank_one
            + self._cmu * rank_mu
        )

    def should_stop(self) -> bool:
        B, D = self._eigen_decomposition()
        dC = jnp.diag(self._C)

        # Stop if the range of function values of the recent generation is below tolfun.
        if (
            self.generation > self._funhist_term
            and jnp.max(self._funhist_values) - jnp.min(self._funhist_values)
            < self._tolfun
        ):
            return True

        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        if jnp.all(self._sigma * dC < self._tolx) and jnp.all(
            self._sigma * self._pc < self._tolx
        ):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * jnp.max(D) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if jnp.any(self._mean == self._mean + (0.2 * self._sigma * jnp.sqrt(dC))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self.generation % self.dim
        if jnp.all(self._mean == self._mean + (0.1 * self._sigma * D[i] * B[:, i])):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = jnp.max(D) / jnp.min(D)
        if condition_cov > self._tolconditioncov:
            return True

        return False

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.array(self._mean, copy=True)


def ensure_jnp(x):
    '''Return a copy of x that is ensured to be jnp.ndarray.'''
    return jnp.array(x)


@partial(jax.jit, static_argnames=['n_dim'])
def _sample_solution(key, B, D, n_dim, mean, sigma) -> jnp.ndarray:
    z = jax.random.normal(key, shape=(n_dim,))   # ~ N(0, I)
    y = B.dot(jnp.diag(D)).dot(z)  # ~ N(0, C)
    x = mean + sigma * y  # ~ N(m, σ^2 C)
    return x


@partial(jax.jit, static_argnames=['n_dim'])
def _masked_sample_solution(mask, key, B, D, n_dim, mean, sigma):
    # TODO(alantian): Make `mask` effective in this bached sampling.
    return _sample_solution(key, B, D, n_dim, mean, sigma)


_v_masked_sample_solution = jax.vmap(
    _masked_sample_solution,
    in_axes=(0, 0, None, None, None, None, None),
    out_axes=0
)


def _is_feasible(param: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
    return jnp.logical_and(
        jnp.all(param >= bounds[:, 0]),
        jnp.all(param <= bounds[:, 1]),
    )


_v_is_feasible = jax.vmap(_is_feasible, in_axes=(0, None), out_axes=0)

@jax.jit
def _eigen_decomposition_core_jitted(_C):
    _C = (_C + _C.T) / 2
    D2, B = jnp.linalg.eigh(_C)
    D = jnp.sqrt(jnp.where(D2 < 0, _EPS, D2))
    _C = jnp.dot(jnp.dot(B, jnp.diag(D ** 2)), B.T)
    _B, _D = B, D
    return _B, _D, _C


def _repair_infeasible_params(param: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
    # clip with lower and upper bound.
    param = jnp.where(param < bounds[:, 0], bounds[:, 0], param)
    param = jnp.where(param > bounds[:, 1], bounds[:, 1], param)
    return param


_v_repair_infeasible_params = jax.vmap(
    _repair_infeasible_params, in_axes=(0, None), out_axes=0)


def _is_valid_bounds(bounds: Optional[jnp.ndarray], mean: jnp.ndarray) -> bool:
    if bounds is None:
        return True
    if (mean.size, 2) != bounds.shape:
        return False
    if not jnp.all(bounds[:, 0] <= mean):
        return False
    if not jnp.all(mean <= bounds[:, 1]):
        return False
    return True


def _compress_symmetric(sym2d: jnp.ndarray) -> jnp.ndarray:
    assert len(sym2d.shape) == 2 and sym2d.shape[0] == sym2d.shape[1]
    n = sym2d.shape[0]
    dim = (n * (n + 1)) // 2
    sym1d = jnp.zeros(dim)
    start = 0
    for i in range(n):
        sym1d[start: start + n - i] = sym2d[i][i:]  # noqa: E203
        start += n - i
    return sym1d


def _decompress_symmetric(sym1d: jnp.ndarray) -> jnp.ndarray:
    n = int(jnp.sqrt(sym1d.size * 2))
    assert (n * (n + 1)) // 2 == sym1d.size
    R, C = jnp.triu_indices(n)
    out = jnp.zeros((n, n), dtype=sym1d.dtype)
    out[R, C] = sym1d
    out[C, R] = sym1d
    return out
