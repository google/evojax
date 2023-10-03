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

"""Train an agent for MNIST classification.

Example command to run this script: `python train_mnist.py --gpu-id=0`
"""

import argparse
import os
import shutil

from evojax import Trainer
from evojax.task.few_shot_omniglot import Omniglot
from evojax.policy import FastLearner
from evojax.algo import PGPE
from evojax import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=32, help='NE population size.')
    parser.add_argument(
        '--batch-size', type=int, default=32, help='Batch size for training. (# of sets in few-shot learning, each member sees the same data)')
    parser.add_argument(
        '--num-grad-steps', type=int, default=3, help='# of gradient steps')
    parser.add_argument(
        '--max-iter', type=int, default=20000, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=1000, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=100, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--center-lr', type=float, default=0.003, help='Center learning rate.')
    parser.add_argument(
        '--std-lr', type=float, default=0.089, help='Std learning rate.')
    parser.add_argument(
        '--init-std', type=float, default=0.039, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = './log/omniglot'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='Omniglot', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX Omniglot Demo')
    logger.info('=' * 30)

    ways = n_classes = 5

    policy = FastLearner(n_classes=n_classes, num_grad_steps=config.num_grad_steps, logger=logger)
    train_task = Omniglot(batch_size=config.batch_size, test=False)
    test_task = Omniglot(batch_size=config.batch_size, test=True)
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

    # Train.
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
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

    # Test the final model.
    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)
