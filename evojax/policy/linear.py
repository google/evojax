import jax 
import jax.numpy as jnp

import logging
from typing import Sequence
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn

from flax import linen as nn


class LinearPolicy(PolicyNetwork): 
	"""A very simple linear policy as used in arxiv.org/abs/1803.07055 """
	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		logger: logging.Logger = None, 
		seed=0):

		
		if logger is None:
			self._logger = create_logger(name='MLPPolicy')
		else:
			self._logger = logger

		model = nn.Dense(output_dim, use_bias=False)
		params = model.init(random.PRNGKey(seed), jnp.ones([1, input_dim]))

		self.num_params, format_params_fn = get_params_format_fn(params)
		self._logger.info('LinearPolicy.num_params = {}'.format(self.num_params))

		self._format_params_fn = jax.vmap(format_params_fn)
		self._forward_fn = jax.vmap(model.apply)

	def get_actions(self,
					t_states: TaskState,
					params: jnp.ndarray,
					p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
		params = self._format_params_fn(params)
		return self._forward_fn(params, t_states.obs), p_states
