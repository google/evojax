# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import numpy as np
from typing import Union
from typing import Tuple
from typing import Callable

import jax.numpy as jnp
from jax import tree_util
from flax.core import FrozenDict


def get_params_format_fn(init_params: FrozenDict) -> Tuple[int, Callable]:
    """Return a function that formats the parameters into a correct format."""

    flat, tree = tree_util.tree_flatten(init_params)
    params_sizes = np.cumsum([np.prod(p.shape) for p in flat])

    def params_format_fn(params: jnp.ndarray) -> FrozenDict:
        params = tree_util.tree_multimap(
            lambda x, y: x.reshape(y.shape),
            jnp.split(params, params_sizes, axis=-1)[:-1],
            flat)
        return tree_util.tree_unflatten(tree, params)

    return params_sizes[-1], params_format_fn


def create_logger(name: str,
                  log_dir: str = None,
                  debug: bool = False) -> logging.Logger:
    """Create a logger.

    Args:
        name - Name of the logger.
        log_dir - The logger will also log to an external file in the specified
                  directory if specified.
        debug - If we should log in DEBUG mode.

    Returns:
        logging.RootLogger.
    """

    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(name)s: %(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO, format=log_format)
    logger = logging.getLogger(name)
    if log_dir:
        log_file = os.path.join(log_dir, '{}.txt'.format(name))
        file_hdl = logging.FileHandler(log_file)
        formatter = logging.Formatter(fmt=log_format)
        file_hdl.setFormatter(formatter)
        logger.addHandler(file_hdl)
    return logger


def load_model(model_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load policy parameters from the specified directory.

    Args:
        model_dir - Directory to load the model from.
    Returns:
        A pair of parameters, the shapes of which are
        (param_size,) and (1 + 2 * obs_params_size,).
    """

    model_file = os.path.join(model_dir, 'model.npz')
    if not os.path.exists(model_file):
        raise ValueError('Model file {} does not exist.')
    with np.load(model_file) as data:
        params = data['params']
        obs_params = data['obs_params']
    return params, obs_params


def save_model(model_dir: str,
               model_name: str,
               params: Union[np.ndarray, jnp.ndarray],
               obs_params: Union[np.ndarray, jnp.ndarray] = None,
               best: bool = False) -> None:
    """Save policy parameters to the specified directory.

    Args:
        model_dir - Directory to save the model.
        model_name - Filename.
        params - The parameters to save.
        obs_params - The observation parameters to save
        best - Whether to save a copy as best.npz.
    """

    model_file = os.path.join(model_dir, '{}.npz'.format(model_name))
    np.savez(model_file,
             params=np.array(params),
             obs_params=np.array(obs_params))
    if best:
        model_file = os.path.join(model_dir, 'best.npz')
        np.savez(model_file,
                 params=np.array(params),
                 obs_params=np.array(obs_params))
