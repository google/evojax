import jax
import jax.numpy as jnp
import numpy as np
from evojax.algo.base import NEAlgorithm
from typing import Union

class ARS_native(NEAlgorithm):
	
	def __init__(self, param_size, pop_size, top_k, noise_stddev=0.1, seed=0, step_size=1e-2, version=1):
		assert pop_size % 2 == 0, "pop size needs to be even for the symmetric sampling "  
		self.pop_size = pop_size
		self.params_population = None
		self._best_params = jnp.zeros(param_size)
		self.param_size = param_size

		self.noise = None
		self.stddev = noise_stddev
		self.rand_key = jax.random.PRNGKey(seed=seed)
		self.step_size = step_size
		self.top_k = top_k
		self.diag_cov = jnp.ones(param_size)
		self.mu = jnp.ones(param_size)
		self.version = version

	def ask(self):
		self.rand_key, _key = jax.random.split(key=self.rand_key, num=2)
		self.noise = self.stddev*jax.random.normal(key=_key, shape=(self.pop_size//2, self.param_size))
		# evaluate each perturbation in both directions 
		self.noise = jnp.concatenate([self.noise, -self.noise], axis=0)   

		self.params_population = jnp.tile(self._best_params, (self.pop_size, 1)) + self.noise 
		return self.params_population

	def tell(self, fitness):
		n = self.pop_size//2    # splitting positive and negative direction perturbations 
		pos = fitness[:n]
		neg = fitness[n:]
		dir_fitnesses = jnp.maximum(pos, neg)
		selected_dirs = jnp.argsort(dir_fitnesses)[-self.top_k:]
		pos, neg = pos[selected_dirs], neg[selected_dirs]

		# take from the end because we want the highest scores 
		pos = pos[jnp.argsort(pos)[-self.top_k:]]
		neg = neg[jnp.argsort(neg)[-self.top_k:]] 

		stddev = max(1e-2, jnp.std(fitness))   ## NOTE: in some tasks sdddev ~ 0 this causes bad updates obviously 
		update = self.step_size / (self.top_k*stddev) * jnp.einsum('i,ij->j', (pos-neg), self.noise[selected_dirs])
		self._best_params = (self._best_params + update)

		# TODO no state normalising for now 
		# if self.version == 2:
		# 	self._best_params = self._best_params * jnp.diag(self.diag_cov)

	@property
	def best_params(self) -> jnp.ndarray:	
		return self._best_params
	
	@best_params.setter
	def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
		self._best_params = params 

