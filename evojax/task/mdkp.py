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
import sys
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask


@dataclass
class State(TaskState):
    obs: jnp.ndarray


class MDKP(VectorizedTask):
    """Multi-Dimensional Knapsack Problem (MDKP).

    Many real-world problems are MDKP in different contexts, and this task
    allows a user to simulate one by using their own data in the form of a csv
    file. If a csv file is not specified, we use synthesized data.

    The format of the csv file is:
        1st line contains the total cap of each attribute (M columns):
            attr1_cap, attr2_cap, ..., attrM_cap
        2nd line and beyond have details for each item (M+1 columns, N rows):
            item_attr1, item1_attr2, ..., item1_attrM, item1_value
            ...
            itemN_attr1, itemN_attr2, ..., itemN_attrM, itemN_value

    A valid solution should select k items that maximizes sum(item_k_value),
    while keeping sum(item_attr_i) < attr_i_cap for all i in [1, M].
    """

    def __init__(self, csv_file=None, test=False):
        self.max_steps = 1
        self.test = test

        if csv_file is None or not os.path.exists(csv_file):
            rnd = np.random.RandomState(seed=0)
            num_attrs = 3
            num_items = 3000
            upper_bound = 100
            data = rnd.uniform(1, upper_bound, size=(num_items, num_attrs + 1))
            caps = rnd.uniform(
                num_items / 3, num_items / 2, num_attrs) * upper_bound
        else:
            try:
                import pandas as pd
            except ModuleNotFoundError:
                print('This task requires pandas, '
                      'run "pip install -U pandas" to install.')
                sys.exit(1)

            data = pd.read_csv(csv_file, header=None).values
            num_items, num_attrs = data.shape
            num_items -= 1  # Exclude the 1st line.
            num_attrs -= 1  # Exclude the value column.
            caps = data[0][:num_attrs]
            data = data[1:]

        self.num_items = num_items
        self.num_attrs = num_attrs
        self.data = data
        self.caps = caps
        data = jnp.array(data)
        caps = jnp.array(caps)
        self.obs_shape = tuple([num_items, num_attrs + 1])
        self.act_shape = tuple([num_items, ])

        def reset_fn(key):
            return State(obs=data)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            selection = jnp.where(action > 0.5, 1, 0)
            total_attr_and_values = (state.obs * selection[:, None]).sum(axis=0)
            reward = total_attr_and_values[-1]
            # If there is any attribute violation, we set reward to zero.
            violation = jnp.prod(jnp.where(
                total_attr_and_values[:num_attrs] > caps, 0, 1))
            reward = reward * violation
            return state, reward, jnp.ones(())
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.array) -> TaskState:
        return self._reset_fn(key)

    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
