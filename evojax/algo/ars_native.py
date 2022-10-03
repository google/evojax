import jax
import jax.numpy as jnp
import numpy as np
from evojax.algo.base import NEAlgorithm
from typing import Union
try:
    from jax.example_libraries import optimizers
except ModuleNotFoundError:
    from jax.experimental import optimizers
import logging
from evojax.util import create_logger


def get_optim(opt, optimizer_config, init_params):
	''' convenience function for creating optimizer functions from configs '''
	lrate_init = optimizer_config.get('lrate_init', 1e-3) 
	lrate_limit = optimizer_config.get('lrate_limit', 1e-6)
	decay_coef = optimizer_config.get('decay_coef', 1)
	step_size=lambda x: jnp.maximum(lrate_init * jnp.power(decay_coef, x), lrate_limit)

	if opt == 'sgd':
		opt_init, opt_update, get_params = optimizers.sgd(
			step_size=step_size,
		)
	elif opt == 'adam':
		opt_init, opt_update, get_params = optimizers.adam(
			step_size=step_size,
			b1=optimizer_config.get("beta1", 0.9),
			b2=optimizer_config.get("beta2", 0.999),
			eps=optimizer_config.get("epsilon", 1e-8),
		)
	else:
		raise Exception("optimizer not supported. Choose between sgd and adam")
	opt_state = jax.jit(opt_init)(init_params)
	opt_update = jax.jit(opt_update)
	get_params = jax.jit(get_params)

	return opt_state, opt_update, get_params



class ARS_native(NEAlgorithm):
	''' Augmented Random Search 
	    Mania et al. (2018) https://arxiv.org/pdf/1803.07055.pdf'''
	
	def __init__(
		self, 
		param_size, 
		pop_size, 
		elite_ratio, 
		decay_stdev=0.999, 
		limit_stdev=0.01, 
		init_stdev=0.03, 
		seed=0, 
		version=1,
		optimizer='sgd',
		optimizer_config={},
		logger: logging.Logger = None
	):
		if logger is None:
			self.logger = create_logger(name="ARS_native")
		else:
			self.logger = logger

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

		self.version = version
		self.decay_stdev = decay_stdev
		self.limit_stdev = limit_stdev
		self.opt_state, self.opt_update, self.opt_get_params = get_optim(optimizer, optimizer_config, self.mean_params)
		self._t = 0

		self.lr_decay_steps = optimizer_config.get('lr_decay_steps', 1)


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

		self.opt_state = self.opt_update(self._t // self.lr_decay_steps, -update, self.opt_state)
		self._t += 1
		self.mean_params = self.opt_get_params(self.opt_state)

		self.stdev = jnp.maximum(self.stdev*self.decay_stdev, self.limit_stdev)

	@property
	def best_params(self) -> jnp.ndarray:	
		return self.mean_params
	
	@best_params.setter
	def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
		self.mean_params = params 
