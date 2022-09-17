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

"""Train an agent to solve the simple addition problem in a seq2seq setting.

In each rollout, the agent receives a token representing an addition problem,
and is required to output a token that represents the answer.
E.g., the agent may see '012+345', and then upon receiving the prompt '=', it
outputs '357'.
This task is based on:
https://github.com/google/flax/tree/main/examples/seq2seq

Example command to run this script: `python train_seq2seq.py --gpu-id=0`
"""

import argparse
import os
import shutil

from evojax import Trainer
from evojax.task.seq2seq import Seq2seqTask
from evojax.policy.seq2seq import Seq2seqPolicy
from evojax.algo import PGPE
from evojax import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=1024, help='ES population size.')
    parser.add_argument(
        '--batch-size', type=int, default=128, help='Batch size for training.')
    parser.add_argument(
        '--hidden-size', type=int, default=128, help='Policy hidden size.')
    parser.add_argument(
        '--max-iter', type=int, default=50000, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=1000, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=100, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--center-lr', type=float, default=0.01, help='Center learning rate.')
    parser.add_argument(
        '--std-lr', type=float, default=0.03, help='Std learning rate.')
    parser.add_argument(
        '--init-std', type=float, default=0.05, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = './log/seq2seq'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='Seq2seq', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX Seq2seq Demo')
    logger.info('=' * 30)

    policy = Seq2seqPolicy(
        hidden_size=config.hidden_size,
        logger=logger,
    )
    train_task = Seq2seqTask(batch_size=config.batch_size, test=False)
    test_task = Seq2seqTask(batch_size=config.batch_size, test=True)
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
        validation_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=1,
        n_evaluations=100,
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
