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

"""Train an agent for masking MNIST classification.

Example command to run this script: `python train_masking.py`
"""

import os
import shutil
import argparse

from jax import tree_util
import jax.numpy as jnp

from evojax import Trainer
from evojax.task.masking import Masking
from evojax.policy.mask import MaskPolicy
from evojax.algo import PGPE
from evojax import util

from evojax.train_mnist_cnn import run_mnist_training, linear_layer_name
from evojax.datasets import DATASET_LABELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=64, help='NE population size.')
    parser.add_argument(
        '--batch-size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument(
        '--max-iter', type=int, default=5000, help='Max training iterations.')
    parser.add_argument(
        '--cnn-epochs', type=int, default=20, help='Number of epochs to train the CNN for.')
    parser.add_argument(
        '--test-interval', type=int, default=1000, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=100, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--center-lr', type=float, default=0.006, help='Center learning rate.')
    parser.add_argument(
        '--std-lr', type=float, default=0.089, help='Std learning rate.')
    parser.add_argument(
        '--init-std', type=float, default=0.039, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    parser.add_argument(
        '--test-no-mask', action='store_true', help='Whether to test a mask of all ones.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = './log/masking'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='MASK', log_dir=log_dir, debug=config.debug)
    logger.info('\nEvoJAX Masking Tests\n')
    logger.info('=' * 30)

    cnn_params = run_mnist_training(logger=logger, return_model=True, num_epochs=config.cnn_epochs)
    linear_weights = cnn_params[linear_layer_name]["kernel"]
    # mask_size = np.prod(linear_weights.shape)
    # TODO currently just masking the input features to the linear layer
    mask_size = linear_weights.shape[0]

    policy = MaskPolicy(logger=logger, mask_size=mask_size, batch_size=config.batch_size,
                        test_no_mask=config.test_no_mask)
    train_task = Masking(batch_size=config.batch_size, test=False, mnist_params=cnn_params, mask_size=mask_size)
    test_task = Masking(batch_size=config.batch_size, test=True, mnist_params=cnn_params, mask_size=mask_size)

    # Need to initialise the solver with the right parameters
    flat, tree = tree_util.tree_flatten(policy.initial_params)
    processed_params = jnp.concatenate([i.ravel() for i in flat])

    solver = PGPE(
        init_params=processed_params,
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
        dataset_labels=DATASET_LABELS
    )
    best_score = trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)


if __name__ == '__main__':
    configs = parse_args()
    main(configs)
