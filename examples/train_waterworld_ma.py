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

"""Train a population of agents to solve the WaterWorld task.

In this task, agents (yellow) tries to catch as much food (green) as possible
while avoiding poisons (red). We wish to feature that it is possible to train
multiple agents in a single task in EvoJAX.
This task is based on:
https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html

Example command to run this script:
`python train_waterworld_ma.py --gpu-id=0 --max-iter=3000`
"""

import argparse
import os
import shutil
import jax
import jax.numpy as jnp

from evojax.task.ma_waterworld import MultiAgentWaterWorld
from evojax.policy.mlp import MLPPolicy
from evojax.algo import PGPE
from evojax import Trainer
from evojax import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hidden-size', type=int, default=100, help='Policy hidden size.')
    parser.add_argument(
        '--num-tests', type=int, default=100, help='Number of test rollouts.')
    parser.add_argument(
        '--n-repeats', type=int, default=64, help='Training repetitions.')
    parser.add_argument(
        '--max-iter', type=int, default=1000, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=100, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=10, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--center-lr', type=float, default=0.011, help='Center learning rate.')
    parser.add_argument(
        '--std-lr', type=float, default=0.054, help='Std learning rate.')
    parser.add_argument(
        '--init-std', type=float, default=0.095, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = './log/water_world_ma'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='MultiAgentWaterWorld', log_dir=log_dir, debug=config.debug)

    logger.info('EvoJAX MultiAgentWaterWorld')
    logger.info('=' * 30)

    num_agents = 16
    max_steps = 500
    train_task = MultiAgentWaterWorld(
        num_agents=num_agents, test=False, max_steps=max_steps)
    test_task = MultiAgentWaterWorld(
        num_agents=num_agents, test=True, max_steps=max_steps)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[-1],
        hidden_dims=[config.hidden_size, ],
        output_dim=train_task.act_shape[-1],
        output_act_fn='softmax',
    )
    solver = PGPE(
        pop_size=num_agents,
        param_size=policy.num_params,
        optimizer='adam',
        center_learning_rate=config.center_lr,
        stdev_learning_rate=config.std_lr,
        init_stdev=config.init_std,
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
        n_evaluations=num_agents,
        n_repeats=config.n_repeats,
        test_n_repeats=config.num_tests,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    # Visualize the policy.
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    action_fn = jax.jit(policy.get_actions)
    best_params = jnp.repeat(
        trainer.solver.best_params[None, :], num_agents, axis=0)
    key = jax.random.PRNGKey(0)[None, :]

    task_state = task_reset_fn(key)
    policy_state = policy_reset_fn(task_state)
    screens = []
    for _ in range(max_steps):
        num_tasks, num_agents = task_state.obs.shape[:2]
        task_state = task_state.replace(
            obs=task_state.obs.reshape((-1, *task_state.obs.shape[2:])))
        action, policy_state = action_fn(task_state, best_params, policy_state)
        action = action.reshape(num_tasks, num_agents, *action.shape[1:])
        task_state = task_state.replace(
            obs=task_state.obs.reshape(
                num_tasks, num_agents, *task_state.obs.shape[1:]))
        task_state, reward, done = step_fn(task_state, action)
        screens.append(MultiAgentWaterWorld.render(task_state))

    gif_file = os.path.join(log_dir, 'water_world_ma.gif')
    screens[0].save(
        gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0)
    logger.info('GIF saved to {}.'.format(gif_file))


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)
