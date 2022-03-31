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

"""Train an agent to solve the classic CartPole swing up task.

Example command to run this script:
# Train in a harder setup.
python train_cartpole.py --gpu-id=0
# Train in an easy setup.
python train_cartpole.py --gpu-id=0 --easy
# Train a permutation invariant agent in a harder setup.
python train_cartpole.py --gpu-id=0 --pi --max-iter=20000 --pop-size=256 \
--center-lr=0.037 \
--std-lr=0.078 \
--init-std=0.082
# Train a permutation invariant agent in a harder setup with CMA-ES.
python train_cartpole.py --gpu-id=0 --pi --max-iter=20000 --pop-size=256 --cma
"""

import argparse
import os
import shutil
import jax

from evojax import Trainer
from evojax.task.cartpole import CartPoleSwingUp
from evojax.policy import MLPPolicy
from evojax.policy import PermutationInvariantPolicy
from evojax.algo import PGPE
from evojax.algo import CMA
from evojax import util
from evojax.util import get_tensorboard_log_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=64, help='NE population size.')
    parser.add_argument(
        '--hidden-size', type=int, default=64, help='Policy hidden size.')
    parser.add_argument(
        '--num-tests', type=int, default=100, help='Number of test rollouts.')
    parser.add_argument(
        '--n-repeats', type=int, default=16, help='Training repetitions.')
    parser.add_argument(
        '--max-iter', type=int, default=1000, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=100, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=20, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--center-lr', type=float, default=0.05, help='Center learning rate.')
    parser.add_argument(
        '--std-lr', type=float, default=0.1, help='Std learning rate.')
    parser.add_argument(
        '--init-std', type=float, default=0.1, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--easy', action='store_true', help='Easy mode.')
    parser.add_argument(
        '--pi', action='store_true', help='Permutation invariant policy.')
    parser.add_argument(
        '--cma', action='store_true', help='Training with CMA-ES.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    hard = not config.easy
    log_dir = './log/cartpole_{}'.format('hard' if hard else 'easy')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='CartPole', log_dir=log_dir, debug=config.debug)

    logger.info('EvoJAX CartPole ({}) Demo'.format('hard' if hard else 'easy'))
    logger.info('=' * 30)

    train_task = CartPoleSwingUp(test=False, harder=hard)
    test_task = CartPoleSwingUp(test=True, harder=hard)
    if config.pi:
        policy = PermutationInvariantPolicy(
            act_dim=test_task.act_shape[0],
            hidden_dim=config.hidden_size,
        )
    else:
        policy = MLPPolicy(
            input_dim=train_task.obs_shape[0],
            hidden_dims=[config.hidden_size] * 2,
            output_dim=train_task.act_shape[0],
        )
    if config.cma:
        solver = CMA(
            pop_size=config.pop_size,
            param_size=policy.num_params,
            init_stdev=config.init_std,
            seed=config.seed,
            logger=logger,
        )
    else:
        solver = PGPE(
            pop_size=config.pop_size,
            param_size=policy.num_params,
            optimizer='adam',
            center_learning_rate=config.center_lr,
            stdev_learning_rate=config.std_lr,
            init_stdev=config.init_std,
            logger=logger,
            seed=config.seed,
        )

    try:
        log_scores_fn = get_tensorboard_log_fn(log_dir=os.path.join(log_dir, "tb_logs"))
    except ImportError as e:
        logger.warning(e)

        def log_scores_fn(i, scores, stage):  # noqa
            pass

    # Train.
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        n_evaluations=config.num_tests,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
        log_scores_fn=log_scores_fn
    )
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    # Generate a GIF to visualize the policy.
    best_params = trainer.solver.best_params[None, :]
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    act_fn = jax.jit(policy.get_actions)
    rollout_key = jax.random.PRNGKey(seed=0)[None, :]

    images = []
    task_s = task_reset_fn(rollout_key)
    policy_s = policy_reset_fn(task_s)
    images.append(CartPoleSwingUp.render(task_s, 0))
    done = False
    step = 0
    while not done:
        act, policy_s = act_fn(task_s, best_params, policy_s)
        task_s, r, d = step_fn(task_s, act)
        step += 1
        done = bool(d[0])
        if step % 5 == 0:
            images.append(CartPoleSwingUp.render(task_s, 0))

    gif_file = os.path.join(
        log_dir, 'cartpole_{}.gif'.format('hard' if hard else 'easy'))
    images[0].save(
        gif_file, save_all=True, append_images=images[1:], duration=40, loop=0)
    logger.info('GIF saved to {}'.format(gif_file))


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)
