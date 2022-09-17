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

"""Use GA to solve an MDKP problem.

This script shows that we can use EvoJAX to solve MDKP problems.
Please be noted that it is meant to be a demo, and may not always find an
approximated solution.

Example commands:
# Train on synthesized data.
python train_mdkp.py --gpu-id=0 --use-synthesized-data
# Train on user specified data.
python train_mdkp.py --gpu-id=0 --item-csv=my_data.csv --cap-csv=my_cap.csv
"""

import argparse
import os
import sys
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp

from evojax import Trainer
from evojax.task.mdkp import MDKP
from evojax.task.mdkp import TaskState
from evojax.policy import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.algo import SimpleGA
from evojax import util


class MDKPPolicy(PolicyNetwork):
    """This policy ignores inputs and is specific to a certain MDKP config."""

    def __init__(self, num_items, num_bins):
        self.num_params = num_items
        thresholds = jnp.ones([num_bins + 1, num_items]) / (num_bins + 1)
        thresholds = jnp.cumsum(thresholds, axis=0)
        thresholds = thresholds - thresholds.min(axis=0)

        def forward_fn(params, obs):
            if num_bins == 1:
                diff = jax.nn.sigmoid(params)[None, :] - 0.5
            else:
                diff = jnp.clip(params - 0.04, a_min=0, a_max=1)[None, :] - thresholds
            selection = jnp.where(diff >= 0, 1, 0).sum(axis=0) - 1
            act = jnp.zeros(
                [num_bins + 1, num_items]).at[
                selection, jnp.arange(num_items)].set(1)
            return act[1:]
        self._forward_fn = jax.vmap(forward_fn)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        return self._forward_fn(params, t_states.obs), p_states


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=256, help='NE population size.')
    parser.add_argument(
        '--max-iter', type=int, default=2000, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=100, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=100, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--item-csv', type=str, default=None, help='Items csv file.')
    parser.add_argument(
        '--cap-csv', type=str, default=None, help='Capacities csv file.')
    parser.add_argument(
        '--use-synthesized-data', action='store_true',
        help='Use synthesized data instead of CSV files.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):

    try:
        import pandas as pd
    except ModuleNotFoundError:
        print('This task requires pandas,'
              'run "pip install -U pandas" to install.')
        sys.exit(1)

    log_dir = './log/mdkp'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='MDKP', log_dir=log_dir, debug=config.debug)

    logger.info('MDKP Demo')
    logger.info('=' * 30)

    train_task = MDKP(
        items_csv=config.item_csv,
        capacity_csv=config.cap_csv,
        use_synthesized_data=config.use_synthesized_data,
        test=False)
    test_task = MDKP(
        items_csv=config.item_csv,
        capacity_csv=config.cap_csv,
        use_synthesized_data=config.use_synthesized_data,
        test=True)
    policy = MDKPPolicy(
        num_bins=train_task.act_shape[0],
        num_items=train_task.act_shape[1],
    )
    solver = SimpleGA(
        pop_size=config.pop_size,
        param_size=policy.num_params,
        logger=logger,
        seed=config.seed,
    )

    # Train.
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        validation_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=1,
        n_evaluations=1,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Summarize the final result.
    logger.info('')
    logger.info('Summary of results')
    logger.info('=' * 30)
    logger.info('Number of attributes: {}'.format(test_task.num_attrs))
    logger.info('Number of items: {}'.format(test_task.num_items))
    logger.info('Number of bins: {}'.format(test_task.num_bins))
    logger.info('Caps: {}'.format(test_task.caps))
    best_params = trainer.solver.best_params[None, :]
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    act_fn = jax.jit(policy.get_actions)
    rollout_key = jax.random.PRNGKey(seed=0)[None, :]
    task_s = task_reset_fn(rollout_key)
    policy_s = policy_reset_fn(task_s)
    act, policy_s = act_fn(task_s, best_params, policy_s)
    task_s, reward, _ = step_fn(task_s, act)
    selections = np.array(act)[0]
    bin_assignment = (
            selections *
            (np.arange(test_task.num_bins)[:, None] + 1)).sum(axis=0)
    results = np.hstack([test_task.items, bin_assignment[:, None]])
    num_selected_items = int(selections.sum())
    if reward == 0:
        logger.info('No valid solution is found.')
    else:
        result_csv = os.path.join(log_dir, 'results.csv')
        pd.DataFrame(results).to_csv(result_csv, header=None)
        logger.info('Number of selected items: {}'.format(num_selected_items))
        logger.info('Total value: {}'.format(float(reward)))
        logger.info('Results saved to {}'.format(result_csv))


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)
