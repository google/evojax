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
import time
import shutil
import argparse
import wandb

from jax import random
import jax.numpy as jnp

from evojax import Trainer
from evojax.task.masking_task import Masking
from evojax.policy.mask_policy import MaskPolicy
from evojax.algo import PGPE, OpenES, CMA_ES_JAX
from evojax import util

from evojax.train_mnist_cnn import run_mnist_training, full_data_loader, eval_model
from evojax.models import cnn_final_layer_name, CNN
from evojax.datasets import DATASET_LABELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=64, help='NE population size.')
    parser.add_argument(
        '--batch-size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument(
        '--max-iter', type=int, default=1000, help='Max training iterations.')
    parser.add_argument(
        '--max-steps', type=int, default=100, help='Max steps for the tasks.')
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
    parser.add_argument('--algo', type=str, default='', help='Evolutionary algorithm to use.',
                        choices=['PGPE', 'CMA', 'OpenES'])
    parser.add_argument('--pixel-input', action='store_true', help='Input the pixel values to the masking model.')
    config, _ = parser.parse_known_args()
    return config


def main(config):

    log_dir = './log/masking'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='MASK', log_dir=log_dir, debug=config.debug)

    time_str = time.strftime("%m%d_%H%M")
    run_name = f'evojax_masking_{config.algo}_{time_str}'
    wandb.init(name=run_name,
               project="evojax-masking",
               entity="ucl-dark",
               dir=log_dir,
               reinit=True,
               config=config)

    start_time = time.time()
    logger.info('\n\nEvoJAX Masking Tests\n')
    logger.info(f'Start Time - {time.strftime("%H:%M")}')
    logger.info('=' * 50)

    best_cnn_state, best_cnn_acc = run_mnist_training(logger, num_epochs=20, early_stopping=True,
                                                      return_model=True)

    policy = MaskPolicy(logger=logger, pretrained_cnn_state=best_cnn_state)

    train_task = Masking(batch_size=config.batch_size, test=False)
    test_task = Masking(batch_size=config.batch_size, test=True)

    if config.algo == 'PGPE':
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
    elif config.algo == 'CMA':
        solver = CMA_ES_JAX(
            pop_size=config.pop_size,
            param_size=policy.num_params,
            init_stdev=config.init_std,
            logger=logger,
            seed=config.seed,
        )
    elif config.algo == 'OpenES':
        solver = OpenES(
            pop_size=config.pop_size,
            param_size=policy.num_params,
            optimizer='adam',
            init_stdev=config.init_std,
            logger=logger,
            seed=config.seed,
        )
    else:
        raise NotImplementedError

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
        use_for_loop=False
    )

    best_score = trainer.run(demo_mode=False)
    # logger.info(f'Best score was: {best_score}')

    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    # # Run a final evaluation with the mask params found
    # _ = eval_model(cnn_params, datasets_tuple[-1], config.batch_size,
    #                mask_params=mask_params, pixel_input=config.pixel_input, cnn_labels=config.cnn_labels)

    end_time = time.time()
    logger.info(f'Total time taken: {end_time-start_time:.2f}s')
    logger.info('RUN COMPLETE\n')
    logger.info('=' * 50)

    wandb.finish()


if __name__ == '__main__':
    configs = parse_args()
    main(configs)
