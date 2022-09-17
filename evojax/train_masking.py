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

Example command to run this script: `python old_train_masking.py`
"""

import os
import time
import argparse
import wandb

from evojax import Trainer
from evojax.task.masking_task import Masking
from evojax.policy.mask_policy import MaskPolicy
from evojax.algo import PGPE, OpenES
from evojax import util
from evojax.mnist_cnn import run_mnist_training, full_data_loader


def parse_args():
    parser = argparse.ArgumentParser()

    # Evo params
    parser.add_argument('--algo', type=str, default=None, help='Evolutionary algorithm to use.',
                        choices=['PGPE', 'OpenES'])
    parser.add_argument('--pop-size', type=int, default=8, help='NE population size.')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--mask-threshold', type=float, default=0.5, help='Threshold for setting binary mask.')
    parser.add_argument('--max-iter', type=int, default=1000, help='Max training iterations.')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps for the tasks.')
    parser.add_argument('--evo-epochs', type=int, default=1, help='Number of epochs for evo process.')
    parser.add_argument('--test-interval', type=int, default=1000, help='Test interval.')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval.')
    parser.add_argument('--center-lr', type=float, default=0.006, help='Center learning rate.')
    parser.add_argument('--std-lr', type=float, default=0.089, help='Std learning rate.')
    parser.add_argument('--init-std', type=float, default=0.039, help='Initial std.')

    # Params for the CNN
    parser.add_argument('--cnn-epochs', type=int, default=20, help='Number of epochs for cnn pretraining.')
    parser.add_argument('--cnn-lr', type=float, default=0.001, help='Learning rate for the CNN model.')
    parser.add_argument('--use-task-labels', action='store_true', help='Input the task labels to the CNN.')
    parser.add_argument('--l1-pruning-proportion', type=float,
                        help='The proportion of weights to remove with L1 pruning.')
    parser.add_argument('--l1-reg-lambda', type=float, help='The lambda to use with L1 regularisation.')
    parser.add_argument('--dropout-rate', type=float, help='The rate for dropout layers in CNN.')

    # General params
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument('--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')

    config, _ = parser.parse_known_args()
    return config


def run_train_masking(config, logger=None) -> float:

    log_dir = './log/masking'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not logger:
        logger = util.create_logger(name='MASK', log_dir=log_dir, debug=config.debug)

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

    cnn_state = cnn_val_acc = mask_params = best_mask_params = None
    datasets_tuple = full_data_loader()
    for i in range(config.evo_epochs):
        cnn_state, cnn_val_acc = run_mnist_training(logger,
                                                    seed=config.seed,
                                                    num_epochs=config.cnn_epochs,
                                                    evo_epoch=i,
                                                    learning_rate=config.cnn_lr,
                                                    cnn_batch_size=config.batch_size,
                                                    state=cnn_state,
                                                    mask_params=mask_params,
                                                    datasets_tuple=datasets_tuple,
                                                    early_stopping=True,
                                                    # These are the parameters for the other
                                                    # sparsity baseline types
                                                    use_task_labels=config.use_task_labels,
                                                    l1_pruning_proportion=config.l1_pruning_proportion,
                                                    l1_reg_lambda=config.l1_reg_lambda,
                                                    dropout_rate=config.dropout_rate)

        if config.algo:
            policy = MaskPolicy(logger=logger,
                                mask_threshold=config.mask_threshold,
                                pretrained_cnn_state=cnn_state)

            train_task = Masking(batch_size=config.batch_size, validation=False)
            validation_task = Masking(batch_size=config.batch_size, validation=True)

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
                    init_params=best_mask_params
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
                validation_task=validation_task,
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

            best_mask_params, best_score = trainer.run(demo_mode=False)
            mask_params = policy.external_format_params_fn(best_mask_params)
            _, final_test_accuracy = run_mnist_training(logger,
                                                        state=cnn_state,
                                                        eval_only=True,
                                                        mask_params=mask_params,
                                                        cnn_batch_size=config.batch_size,
                                                        )

    # src_file = os.path.join(log_dir, 'best.npz')
    # tar_file = os.path.join(log_dir, 'model.npz')
    # shutil.copy(src_file, tar_file)
    # trainer.model_dir = log_dir
    # trainer.run(demo_mode=True)

    end_time = time.time()
    logger.info(f'Total time taken: {end_time-start_time:.2f}s')
    logger.info('RUN COMPLETE\n')
    logger.info('=' * 50)

    wandb.finish()

    return cnn_val_acc


if __name__ == '__main__':
    configs = parse_args()
    run_train_masking(configs)
