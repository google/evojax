"""
**Experimental** CMA-ES optimizer using JAX backend.

Code in this file is an adaption from <https://github.com/CyberAgentAILab/cmaes/blob/main/cmaes/_cma.py>
which is a faithful implementation of CMA-ES (Covariance matrix adaptation evolution strategy)
described in <https://arxiv.org/abs/1604.00772>.

Overall, the adaption replaces the NumPy backend with JAX. To facilitate efficient computation in JAX, it:
- Uses stateful computation for JAX that is JIT-able.
- Enables paralleling sampling.

Some edge-case are not concerned for now:
- No pickling functionality (`__getstate__()` and `__setstate__()`).
- No bound / repeated sampling for bounded parameters. This makes `ask` easier.
- No `funhist` / early stop. This makes `tell` and stateful computation easier.

This is still an experimental implementation and has not been thoroughly-tested yet.
"""

from typing import NamedTuple
from functools import partial
import logging
import math
from typing import Optional

import numpy as np

import jax
from jax import numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


_EPS = 1e-8


# We set thresholds for mean and sigma respectively.
_MEAN_MAX_X64 = 1e302
_SIGMA_MAX_X64 = 1e302
_MEAN_MAX_X32 = 1e32
_SIGMA_MAX_X32 = 1e32


class _HyperParameters(NamedTuple):
    n_dim: int
    pop_size: int
    mu: int


class _Coefficients(NamedTuple):
    mu_eff: jnp.ndarray
    cc: jnp.ndarray
    c1: jnp.ndarray
    cmu: jnp.ndarray
    c_sigma: jnp.ndarray
    d_sigma: jnp.ndarray
    cm: jnp.ndarray
    chi_n: jnp.ndarray
    weights: jnp.ndarray
    sigma_max: jnp.ndarray


class _State(NamedTuple):
    p_sigma: jnp.ndarray
    pc: jnp.ndarray
    mean: jnp.ndarray
    C: jnp.ndarray
    sigma: jnp.ndarray
    D: jnp.ndarray
    B: jnp.ndarray
    g: int
    key: jnp.ndarray


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
        cov:
            A covariance matrix (optional).
        logger:
            A logging.Logger instance (optional).
            If not specified, a new one will be created.
        enable_numeric_check:
            A bool indicating whether to enable numeric check (optional).
            If True, a numeric check for mean and standard deviation is enabled.
            The default value is False.
    """

    def __init__(
        self,
        param_size: int,
        pop_size: Optional[int] = None,
        mean: Optional[jnp.ndarray] = None,
        init_stdev: Optional[float] = 0.1,
        seed: Optional[int] = 0,
        cov: Optional[jnp.ndarray] = None,
        logger: logging.Logger = None,
        enable_numeric_check: Optional[bool] = False,
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

        if enable_numeric_check:
            mean_max = ensure_jnp(_MEAN_MAX_X64 if jax.config.jax_enable_x64 else _MEAN_MAX_X32)
            sigma_max = ensure_jnp(_SIGMA_MAX_X64 if jax.config.jax_enable_x64 else _SIGMA_MAX_X32)
        else:
            mean_max = ensure_jnp(jnp.inf)
            sigma_max = ensure_jnp(jnp.inf)

        if enable_numeric_check:
            assert jnp.all(
                jnp.abs(mean) <= mean_max
            ), f"Abs of all elements of mean vector must be less than {mean_max}"

        n_dim = len(mean)
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        assert init_stdev > 0, "init_stdev must be non-zero positive value"
        sigma = init_stdev
        sigma = ensure_jnp(sigma)

        population_size = pop_size
        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))  # (eq. 48)
        assert population_size > 0, "pop_size must be non-zero positive value."

        mu = population_size // 2

        # (eq.49)
        weights_prime = jnp.array(
            [
                math.log((population_size + 1) / 2) - math.log(i + 1)
                for i in range(population_size)
            ]
        )
        mu_eff = (jnp.sum(weights_prime[:mu]) **
                  2) / jnp.sum(weights_prime[:mu] ** 2)
        mu_eff_minus = (
            jnp.sum(weights_prime[mu:]) ** 2) / jnp.sum(weights_prime[mu:] ** 2)

        # learning rate for the rank-one update
        alpha_cov = 2.0
        c1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
        assert isinstance(c1, jnp.ndarray)
        # learning rate for the rank-μ update
        cmu = min(
            1 - c1 - 1e-8,  # 1e-8 is for large pop_size.
            alpha_cov
            * (mu_eff - 2 + 1 / mu_eff)
            / ((n_dim + 2) ** 2 + alpha_cov * mu_eff / 2),
        )
        assert c1 + cmu <= 1, "invalid learning rate for the rank-one and/or rank-μ update"

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

        self.hyper_parameters = _HyperParameters(
            n_dim=n_dim,
            pop_size=population_size,
            mu=mu,
        )

        self.coefficients = _Coefficients(
            mu_eff=mu_eff,
            cc=cc,
            c1=c1,
            cmu=cmu,
            c_sigma=c_sigma,
            d_sigma=d_sigma,
            cm=cm,
            # E||N(0, I)|| (p.28)
            chi_n=math.sqrt(n_dim) * (
                1.0 - (1.0 / (4.0 * n_dim)) +
                1.0 / (21.0 * (n_dim ** 2))
            ),
            weights=weights,
            sigma_max=sigma_max,
        )

        # evolution path (state)
        if cov is None:
            cov = jnp.eye(n_dim)
        else:
            assert cov.shape == (
                n_dim, n_dim), "Invalid shape of covariance matrix"

        self.state = _State(
            p_sigma=jnp.zeros(n_dim),
            pc=jnp.zeros(n_dim),
            mean=mean,
            C=cov,
            sigma=sigma,
            D=None,
            B=None,
            g=0,
            key=jax.random.PRNGKey(seed),
        )

        # Below are helper vars.
        self._latest_solutions = None
        if logger is None:
            # Change this name accordingly
            self.logger = create_logger(name="CMA")
        else:
            self.logger = logger

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self.hyper_parameters.n_dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self.hyper_parameters.pop_size

    @property
    def pop_size(self) -> int:
        """A population size, as expected by EvoJAX trainer."""
        return self.hyper_parameters.pop_size

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self.state.g

    def _eigen_decomposition(self) -> jnp.ndarray:
        if self.state.B is not None and self.state.D is not None:
            return self.state.B, self.state.D
        B, D, C = _eigen_decomposition_core(self.state.C)
        self.state = self.state._replace(B=B, D=D, C=C)
        return B, D

    def _ask(self, n_samples) -> jnp.ndarray:
        """Real implementaiton of ask, which samples multiple parameters in parallel."""
        n_dim = self.hyper_parameters.n_dim
        mean, sigma = self.state.mean, self.state.sigma
        B, D = self._eigen_decomposition()

        key, subkey = jax.random.split(self.state.key)
        self.state = self.state._replace(key=key)
        subkey = jax.random.split(subkey, n_samples)
        x = _batch_sample_solution(subkey, B, D, n_dim, mean, sigma)

        return x

    def ask(self, n_samples: int = None) -> jnp.ndarray:
        """A wrapper of _ask, which handles optional n_samples and saves latest samples."""

        if n_samples is None:
            n_samples = self.pop_size

        x = self._ask(n_samples)
        self._latest_solutions = x
        return x

    def tell(self, fitness: jnp.ndarray, solutions: Optional[jnp.ndarray] = None) -> None:
        """Tell evaluation values as fitness."""

        if solutions is None:
            assert self._latest_solutions is not None, \
                "`soltuions` is not given, expecting using latest samples but this was not done."
            assert self._latest_solutions.shape[0] == self.hyper_parameters.pop_size, \
                f"Latest samples (shape={self._latest_solutions.shape}) not having pop_size-length ({self._popsize})."
            solutions = self._latest_solutions
        else:
            assert solutions.shape[0] == self.hyper_parameters.pop_size, \
                "Given solutions must have pop_size-length, which is not ture."

        # We want maximization, while the following logics is for minimimzation.
        # Handle this calse by simply revert fitness
        fitness = - fitness

        # real computation

        # - must do it as _tell_core below expects B, C, D to be computed.
        B, D = self._eigen_decomposition()
        self.state = self.state._replace(B=B, D=D)

        next_state = _tell_core(hps=self.hyper_parameters, coeff=self.coefficients,
                                state=self.state, fitness=fitness, solutions=solutions)

        self.state = next_state

    def save_state(self) -> dict:
        fn = lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x

        state = {
            key: fn(value)
            for key, value in self.state._asdict().items()
        }
        return state

    def load_state(self, saved_state: dict):
        fn = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x

        self.state = _State(**{
            key: fn(value)
            for key, value in saved_state.items()
        })

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.array(self.state.mean, copy=True)

    def get_best_params_ref(self) -> jnp.ndarray:
        return self.state.mean


def ensure_jnp(x):
    '''Return a copy of x that is ensured to be jnp.ndarray.'''
    return jnp.array(x)


@partial(jax.jit, static_argnums=0)
def _tell_core(
    hps: _HyperParameters,
    coeff: _Coefficients,
    state: _State,
    fitness: jnp.ndarray,
    solutions: jnp.ndarray,
) -> _State:
    next_state = state

    g = state.g + 1
    next_state = next_state._replace(g=g)

    ranking = jnp.argsort(fitness, axis=0)

    sorted_solutions = solutions[ranking]

    B, D = state.B, state.D

    # Sample new population of search_points, for k=1, ..., pop_size
    B, D = state.B, state.D  # already computed.
    next_state = next_state._replace(B=None, D=None)

    x_k = jnp.array(sorted_solutions)  # ~ N(m, σ^2 C)
    y_k = (x_k - state.mean) / state.sigma  # ~ N(0, C)

    # Selection and recombination
    # y_w = jnp.sum(y_k[: hps.mu].T * coeff.weights[: hps.mu], axis=1)  # eq.41
    # use lax.dynamic_slice_in_dim here:
    y_w = jnp.sum(
        jax.lax.dynamic_slice_in_dim(y_k, 0, hps.mu, axis=0).T *
        jax.lax.dynamic_slice_in_dim(coeff.weights,  0, hps.mu, axis=0),
        axis=1,
    )
    mean = state.mean + coeff.cm * state.sigma * y_w
    next_state = next_state._replace(mean=mean)

    # Step-size control
    C_2 = B.dot(jnp.diag(1 / D)).dot(B.T)  # C^(-1/2) = B D^(-1) B^T
    p_sigma = (1 - coeff.c_sigma) * state.p_sigma + jnp.sqrt(
        coeff.c_sigma * (2 - coeff.c_sigma) * coeff.mu_eff
    ) * C_2.dot(y_w)
    next_state = next_state._replace(p_sigma=p_sigma)

    norm_p_sigma = jnp.linalg.norm(state.p_sigma)
    sigma = state.sigma * jnp.exp(
        (coeff.c_sigma / coeff.d_sigma) * (norm_p_sigma / coeff.chi_n - 1)
    )
    sigma = jnp.min(jnp.array([sigma, coeff.sigma_max]))
    next_state = next_state._replace(sigma=sigma)

    # Covariance matrix adaption
    h_sigma_cond_left = norm_p_sigma / jnp.sqrt(
        1 - (1 - coeff.c_sigma) ** (2 * (state.g + 1))
    )
    h_sigma_cond_right = (1.4 + 2 / (hps.n_dim + 1)) * coeff.chi_n
    # h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)
    h_sigma = jax.lax.cond(
        pred=(h_sigma_cond_left < h_sigma_cond_right),
        true_fun=(lambda: 1.0),
        false_fun=(lambda: 0.0),
    )

    # (eq.45)
    pc = (1 - coeff.cc) * state.pc + h_sigma * \
        jnp.sqrt(coeff.cc * (2 - coeff.cc) * coeff.mu_eff) * y_w
    next_state = next_state._replace(pc=pc)

    # (eq.46)
    w_io = coeff.weights * jnp.where(
        coeff.weights >= 0,
        1,
        hps.n_dim / (jnp.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + _EPS),
    )

    delta_h_sigma = (1 - h_sigma) * coeff.cc * (2 - coeff.cc)  # (p.28)
    # assert delta_h_sigma <= 1

    # (eq.47)
    rank_one = jnp.outer(state.pc, state.pc)

    # This way of computing rank_mu can lead to OOM:
    # rank_mu = jnp.sum(
    #    jnp.array([w * jnp.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0
    # )
    # Try another way of computing rank_mu
    rank_mu = jnp.zeros_like(state.C)
    for w, y in zip(w_io, y_k):
        rank_mu = rank_mu + w * jnp.outer(y, y)

    C = (
        (
            1
            + coeff.c1 * delta_h_sigma
            - coeff.c1
            - coeff.cmu * jnp.sum(coeff.weights)
        )
        * state.C
        + coeff.c1 * rank_one
        + coeff.cmu * rank_mu
    )
    next_state = next_state._replace(C=C)

    return next_state


def _sample_solution(key, B, D, n_dim, mean, sigma) -> jnp.ndarray:
    z = jax.random.normal(key, shape=(n_dim,))   # ~ N(0, I)
    y = B.dot(jnp.diag(D)).dot(z)  # ~ N(0, C)
    x = mean + sigma * y  # ~ N(m, σ^2 C)
    return x


_batch_sample_solution = jax.jit(
    jax.vmap(
        _sample_solution,
        in_axes=(0, None, None, None, None, None),
        out_axes=0
    ),
    static_argnums=3,
)


@jax.jit
def _eigen_decomposition_core(_C):
    _C = (_C + _C.T) / 2
    D2, B = jnp.linalg.eigh(_C)
    D = jnp.sqrt(jnp.where(D2 < 0, _EPS, D2))
    _C = jnp.dot(jnp.dot(B, jnp.diag(D ** 2)), B.T)
    _B, _D = B, D
    return _B, _D, _C
