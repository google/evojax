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
    sel: jnp.ndarray


class MDKP(VectorizedTask):
    """Multi-Dimensional Knapsack Problem (MDKP).

    Many real-world problems are MDKP in different contexts, and this task
    allows a user to simulate one by using their own data in the form of csv
    files. In addition, users can also use synthesized data.

    The format of the csv files are:

    1. capacity_csv should have N lines and S columns.
        bin1_attr1_cap, bin1_attr2_cap, ..., bin1_attrS_cap
        bin2_attr1_cap, bin2_attr2_cap, ..., bin2_attrS_cap
        ...
        binN_attr1_cap, binN_attr2_cap, ..., binN_attrS_cap

    2. items_csv should have M lines and S + 1 columns.
        item1_attr1, item1_attr2, ..., item1_attrS, item1_value
        item2_attr1, item2_attr2, ..., item2_attrS, item2_value
        ...
        itemM_attr1, itemM_attr2, ..., itemM_attrS, itemM_value

    A valid solution should select K<=M items that maximizes sum(item{k}_value),
    while keeping sum(item{k}_attr{j}) < bin{i}_attr{j}_cap
    if item{k} is assigned to bin{i}, for i in [1, N] and j in [1, S].
    """

    def __init__(self,
                 items_csv=None,
                 capacity_csv=None,
                 use_synthesized_data=True,
                 test=False):
        self.max_steps = 1
        self.test = test

        if use_synthesized_data:
            rnd = np.random.RandomState(seed=0)
            num_attrs = 3
            num_items = 2000
            num_bins = 5
            upper_bound = 100
            items = rnd.randint(
                low=0, high=upper_bound, size=(num_items, num_attrs + 1))

            num_items_per_bin = int(num_items * 0.5 / num_bins)
            num_sel_items = num_items_per_bin * num_bins
            ix = np.arange(num_items)
            rnd.shuffle(ix)
            sel_ix = ix[:num_sel_items]
            sel_items = items[sel_ix, :-1]
            caps = np.array(
                [sel_items[
                 (i * num_items_per_bin):(i + 1) * num_items_per_bin
                 ].sum(axis=0) for i in range(num_bins)])
        else:
            try:
                import pandas as pd
            except ModuleNotFoundError:
                print('This task requires pandas, '
                      'run "pip install -U pandas" to install.')
                sys.exit(1)

            caps = pd.read_csv(capacity_csv, header=None).values
            num_bins, num_attrs = caps.shape
            items = pd.read_csv(items_csv, header=None).values
            num_items, num_cols = items.shape
            assert num_cols == num_attrs + 1

        self.num_items = num_items
        self.num_attrs = num_attrs
        self.num_bins = num_bins
        self.items = items
        self.caps = caps
        items = jnp.array(items)
        caps = jnp.array(caps)
        self.obs_shape = tuple([num_items, num_attrs + 1])
        self.act_shape = tuple([num_bins, num_items])

        def reset_fn(key):
            return State(obs=items, sel=jnp.zeros([num_bins, num_items]))
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            # selection.shape = (N, M), obs.shape = (M, S + 1)
            selection = jnp.where(action > 0.5, 1., 0.)
            # bin_items_attr_sum.shape = (N, S + 1)
            bin_items_attr_sum = jnp.matmul(selection, state.obs)
            reward = bin_items_attr_sum[:, -1].sum()
            # If any item is assigned more than once, we set reward to zero.
            times_selected = selection.sum(axis=0)
            assignment_violation = jnp.prod(jnp.where(times_selected > 1, 0, 1))
            reward = reward * assignment_violation
            # If there is any limit violation, we set reward to zero.
            violation = jnp.prod(
                jnp.where(bin_items_attr_sum[:, :-1] > caps, 0, 1).ravel())
            reward = reward * violation
            return State(obs=state.obs, sel=action), reward, jnp.ones(())
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.array) -> TaskState:
        return self._reset_fn(key)

    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
