import jax
import jax.numpy as jnp
from evojax.algo.base import NEAlgorithm

class SimpleGA(NEAlgorithm):
    """A simple genetic algorithm implementing truncation selection."""
    
    def __init__(self, param_size, pop_size, trc_factor=2, sigma=0.01, seed=0):

        self.param_size = param_size
        self.pop_size = pop_size
        assert pop_size % 2 == 0, "pop_size must be a multiple of 2."
        self.trc_factor = trc_factor
        assert pop_size % trc_factor == 0, \
        "trc_factor must be a multiple of pop_size."
        self.sigma = sigma

        self.params = jnp.zeros((pop_size, param_size))
        self.rand_key = jax.random.PRNGKey(seed=seed)
        self._best_params = None

        self.jnp_array = jax.jit(jnp.array)
        
        def ask_fn(key, params):

            next_key, sample_key = jax.random.split(key=key, num=2)

            perturbations = jax.random.normal(key=sample_key,
                                              shape=(self.pop_size,
                                                     self.param_size))

            return next_key, params + perturbations * self.sigma

        self.ask_fn = jax.jit(ask_fn)

        def tell_fn(fitness, params):

            params = params[fitness.argsort(axis=0)]

            trc = self.pop_size // self.trc_factor
            
            params = params[-trc:].repeat(self.trc_factor, axis=0)
            
            return params

        self.tell_fn = jax.jit(tell_fn)

    def ask(self):
        self.rand_key, self.params = self.ask_fn(self.rand_key, self.params)
        return self.params

    def tell(self, fitness):
        self.params = self.tell_fn(fitness, self.params)
        self._best_params = self.params[-1]

    @property
    def best_params(self):
        return self.jnp_array(self._best_params)

    @best_params.setter
    def best_params(self, params):
        self._best_params = params
