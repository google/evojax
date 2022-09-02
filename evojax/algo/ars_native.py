import jax
import jax.numpy as jnp
import numpy as np
from evojax.algo.base import NEAlgorithm
from typing import Union

class ARS_native(NEAlgorithm):
	
	def __init__(self, param_size, pop_size, elite_ratio, decay_stdev=0.999, limit_stdev=0.01, init_stdev=0.03, seed=0, step_size=1e-2, version=1):
		assert pop_size % 2 == 0   # total perturbations considering positive and negative directions 
		self.pop_size = pop_size
		assert 0 < elite_ratio <= 1
		self.elite_ratio = elite_ratio
		self.elite_pop_size = int(self.pop_size / 2 * self.elite_ratio)
		self.directions = pop_size // 2
		self.params_population = None
		self.param_size = param_size

		self.noise = None
		self.stdev = init_stdev 
		self.rand_key, _key = jax.random.split(key=jax.random.PRNGKey(seed=seed), num=2)
		init_min, init_max = 0, 0
		self.mean_params = jax.random.uniform(_key, (param_size,), minval=init_min, maxval=init_max) 
		self.step_size = step_size

		self.version = version
		self.decay_stdev = decay_stdev
		self.limit_stdev = limit_stdev

	def ask(self):
		self.rand_key, _key = jax.random.split(key=self.rand_key, num=2)
		self.noise = jax.random.normal(key=_key, shape=(self.directions, self.param_size))
		perturbation = jnp.concatenate([self.noise, -self.noise], axis=0)

		# evaluate each perturbation in both directions 
		self.params_population = self.mean_params + self.stdev * perturbation
		return self.params_population

	def tell(self, fitness):
		pos = fitness[:self.directions]
		neg = fitness[self.directions:]

		selected_dir_ids = jnp.minimum(pos, neg).argsort()[:self.elite_pop_size]
		selected_fitness_stdev = jnp.std(
			jnp.concatenate(
				[pos[selected_dir_ids], neg[selected_dir_ids]]
			)
		) + 1e-05
		fitness_diff = pos[selected_dir_ids]-neg[selected_dir_ids]
		selected_noise = self.noise[selected_dir_ids]
		update = 1 / (self.elite_pop_size*selected_fitness_stdev) * jnp.dot(selected_noise.T, fitness_diff) 
		self.mean_params = (self.mean_params+ self.step_size * update)

		self.stdev = jnp.maximum(self.stdev*self.decay_stdev, self.limit_stdev)

	@property
	def best_params(self) -> jnp.ndarray:	
		return self.mean_params
	
	@best_params.setter
	def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
		self.mean_params = params 

