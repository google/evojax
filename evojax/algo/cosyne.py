import logging
import numpy as np
from typing import Union
from typing import Tuple

import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


class CoSyNE(NEAlgorithm):
    """Cooperative Synapse Neuroevoluiton (CoSyNE) algorithm.

    Attempts to reproduce CoSyNE as described by Gomes et al. in:
    https://people.idsia.ch/~juergen/gomez08a.pdf

    TLDR: Cull bottom half of population, and replace with Cauchy-mutated multi-point-crossover offspring of the top
    quarter of the population, and then shuffle the survivors' columns
    """

    def __init__(self,
                 pop_size: int,
                 param_size: int,
                 alpha: float = 2,
                 prob_mutate: float = 0.3,
                 segment_size: int = 411,
                 reset_prob: float = 0.0001,
                 seed: int = 0,
                 shuffle: bool = True,
                 logger: logging.Logger = create_logger(name='CoSyNE')):
        """Initialization function.

        Args:
            param_size - Parameter size.
            pop_size - Population size.
            alpha - Learning rate multiplier of cauchy distribution of mutation.
            prob_mutate - Probability of mutation.
            segment_size - Parameter segment size, each to include a single random crossover point
            seed - Random seed for parameters sampling.
            logger - Logger
        """

        self.logger = logger
        self.param_size = param_size
        self.n = self.param_size
        self.prob_mutate = prob_mutate
        assert pop_size % 4 == 0, 'Population size should be a multiple of 4, set to {}'.format(pop_size)
        self.pop_size = abs(pop_size)
        self.m = self.pop_size
        self.reset_prob = reset_prob

        self.segment_size = segment_size

        self.alpha = alpha

        next_key, sample_key = jax.random.split(jax.random.PRNGKey(seed=seed))
        self.params = jax.random.uniform(sample_key, (pop_size, param_size), minval=-alpha, maxval=alpha)
        self._best_params = None

        self.rand_key = jax.random.PRNGKey(seed=seed)

        self.jnp_array = jax.jit(jnp.array)

        def ask_fn(key: jnp.ndarray,
                   params: Union[np.ndarray,
                   jnp.ndarray]) -> Tuple[jnp.ndarray,
        Union[np.ndarray,
        jnp.ndarray]]:
            # determine survivors
            survivors = params[0:self.pop_size // 2]

            # determine parents
            parents = params[0:self.pop_size // 4]

            # determine offspring

            # determine offspring: crossover once per segment at a random position
            num_segments = self.n // self.segment_size
            quarter_pop = self.m // 4
            next_key, sample_key = jax.random.split(key=key, num=2)
            crossovers = jax.random.randint(key=sample_key,
                                            shape=(quarter_pop, num_segments),
                                            minval=0,
                                            maxval=self.segment_size)

            # determine offspring: mates are uniformly chosen from fitter ranks
            def for_mate_ranks(parent_index_and_next_key, _):
                parent_rank = parent_index_and_next_key[0]
                next_key, sample_key = jax.random.split(parent_index_and_next_key[1], 2)
                mate_rank = jax.random.randint(key=sample_key,
                                               shape=(1,),
                                               minval=0,
                                               maxval=parent_rank)[0]

                return (parent_index_and_next_key[0] + 1, next_key), mate_rank

            next_key, sample_key = jax.random.split(key=key, num=2)
            parent_index_and_next_key, mate_ranks = jax.lax.scan(for_mate_ranks, (0, next_key), parents)

            next_key, sample_key = jax.random.split(key=key, num=2)
            reset_chances = jax.random.uniform(key=sample_key, shape=(self.m // 2, param_size))

            def offspring_elem(i, j):
                """
                Creates offspring matrix element (i, j) as being either a copy of a parent or their mate, conditional on
                whether they are offspring A or offspring B, and whether the genomes are currently crossed
                """
                i = i.astype(int)
                j = j.astype(int)
                is_offspring_a = i < (self.m // 4)
                is_offspring_b = i >= (self.m // 4)
                parent_rank = i % (self.m // 4)
                mate_rank = mate_ranks[parent_rank]
                crossover_file = j // self.segment_size
                crossover_pos = crossovers[parent_rank, crossover_file]
                segment_pos = j % self.segment_size
                crossed = jnp.any(jnp.array([jnp.all(jnp.array([crossover_file % 2 == 1, segment_pos < crossover_pos])),
                                             jnp.all(
                                                 jnp.array([crossover_file % 2 == 0, segment_pos >= crossover_pos]))]))

                conds = jnp.array([
                    reset_chances[i, j] > (1 - self.reset_prob),
                    jnp.all(jnp.array([is_offspring_a, crossed])),
                    jnp.all(jnp.array([is_offspring_b, crossed])),
                    is_offspring_a,
                    is_offspring_b
                ])

                index = jnp.argmax(conds)
                branches = [
                    lambda _x, _y: 0.0,
                    lambda x, y: parents[mate_rank, y],
                    lambda x, y: parents[parent_rank, y],
                    lambda x, y: parents[parent_rank, y],
                    lambda x, y: survivors[mate_rank, y]
                ]

                return jax.lax.switch(index, branches, i, j)

            clones = jnp.fromfunction(offspring_elem, (self.m // 2, self.param_size))

            # prob_mutate elements of unmutated_offspring should be mutated...
            next_key, sample_key = jax.random.split(next_key, 2)
            mutate = jax.random.choice(key=sample_key,
                                       a=jnp.array([0, 1]),
                                       p=jnp.array([1 - self.prob_mutate, self.prob_mutate]),
                                       shape=(self.pop_size // 2, self.param_size))

            # ...by a Cauchy distribution sample amount multiplied by an alpha learning rate
            # NB: alpha can be very task specific (eg. 0.0001 for MNIST and 0.3 for cartpole_easy)
            next_key, sample_key = jax.random.split(next_key, 2)
            mutation = jax.random.cauchy(key=sample_key,
                                         shape=(self.m // 2,
                                                self.param_size)) * alpha

            offspring = clones + mutation * mutate

            if shuffle:
                def shuffle_scan_body(carry, x):
                    next_key, sample_key = jax.random.split(carry[1], 2)
                    y = jax.random.permutation(key=sample_key, x=x)
                    return (carry[0] + 1, next_key), y

                next_key, sample_key = jax.random.split(next_key, 2)
                carry, shuffled = jax.lax.scan(shuffle_scan_body, (0, next_key), jnp.transpose(survivors))
                _, next_key = carry

                new_params = jnp.vstack((
                    jnp.transpose(shuffled),
                    offspring
                ))
            else:
                new_params = jnp.vstack((
                    survivors,
                    offspring
                ))

            return next_key, new_params

        self.ask_fn = jax.jit(ask_fn)

        def tell_fn(fitness: Union[np.ndarray, jnp.ndarray],
                    params: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
            # sort population in fitness descending order
            return params[(-fitness).argsort(axis=0)]

        self.tell_fn = jax.jit(tell_fn)

    def ask(self) -> jnp.ndarray:
        self.rand_key, self.params = self.ask_fn(self.rand_key, self.params)
        return self.params

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        self.params = self.tell_fn(fitness, self.params)
        self._best_params = self.params[0]

    @property
    def best_params(self) -> jnp.ndarray:
        return self.jnp_array(self._best_params)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self.params = jnp.repeat(params[None, :], self.pop_size, axis=0)
        self._best_params = jnp.array(params, copy=True)
