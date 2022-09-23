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
import numpy as np
import jax

from evojax import util
from evojax import Trainer
from evojax.task.masking_task import Masking
from evojax.policy.mask_policy import MaskPolicy
from evojax.algo import PGPE, OpenES, CMA_ES_JAX, CMA_ES
from evojax.mnist_cnn import run_mnist_training
from evojax.datasets import DATASET_LABELS, full_data_loader
from evojax.models import create_train_state


def parse_cnn_args(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument('--cnn-epochs', type=int, default=20, help='Number of epochs for cnn pretraining.')
    arg_parser.add_argument('--cnn-lr', type=float, default=0.001, help='Learning rate for the CNN model.')
    arg_parser.add_argument('--use-task-labels', action='store_true', help='Input the task labels to the CNN.')
    arg_parser.add_argument('--l1-pruning-proportion', type=float,
                            help='The proportion of weights to remove with L1 pruning.')
    arg_parser.add_argument('--l1-reg-lambda', type=float, help='The lambda to use with L1 regularisation.')
    arg_parser.add_argument('--weight-decay', type=float, help='The lambda to use with weight decay.')
    arg_parser.add_argument('--dropout-rate', type=float, help='The rate for dropout layers in CNN.')
    arg_parser.add_argument('--early-stopping', action='store_true', help='Stop on decrease in val accuracy.')


def parse_args():
    parser = argparse.ArgumentParser()

    # Evo params
    parser.add_argument('--algo', type=str, default=None, help='Evolutionary algorithm to use.',
                        choices=['PGPE', 'OpenES', 'CMA'])

    # Different types of masking
    parser.add_argument('--pixel-input', action='store_true',
                        help='Generate the mask based on pixels rather than task labels.')
    parser.add_argument('--image-mask', action='store_true', help='Mask the image rather than the internal features.')
    parser.add_argument('--meta-learning', action='store_true', help='Perform meta learning.')

    parser.add_argument('--pop-size', type=int, default=8, help='NE population size.')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--mask-threshold', type=float, help='Threshold for setting binary mask.')
    parser.add_argument('--max-iter', type=int, default=1000, help='Max training iterations.')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps for the tasks.')
    parser.add_argument('--evo-epochs', type=int, default=0, help='Number of epochs for evo process.')
    parser.add_argument('--test-interval', type=int, default=100, help='Test interval.')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval.')
    parser.add_argument('--center-lr', type=float, default=0.006, help='Center learning rate.')
    parser.add_argument('--std-lr', type=float, default=0.089, help='Std learning rate.')
    parser.add_argument('--init-std', type=float, default=0.039, help='Initial std.')

    # Params for the CNN
    parse_cnn_args(parser)

    # General params
    parser.add_argument('--datasets', nargs='*', default=list(DATASET_LABELS.keys()),
                        help='Which datasets to use.')
    parser.add_argument('--val-fraction', type=float, default=0.2, help='Fraction of data to use for validation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument('--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')

    parsed_config, _ = parser.parse_known_args()
    return parsed_config


def run_train_masking(dataset_names: list,
                      algo=None,
                      # Different masking setups
                      pixel_input=False,
                      image_mask=False,
                      meta_learning=False,
                      # General params
                      seed=0,
                      pop_size=8,
                      batch_size=1024,
                      mask_threshold=None,
                      max_iter=100,
                      max_steps=1,
                      evo_epochs=1,
                      test_interval=10,
                      log_interval=10,
                      center_lr=0.006,
                      std_lr=0.089,
                      init_std=0.039,
                      # Config args
                      debug=False,
                      logger=None,
                      config_dict=None,
                      datasets_tuple=None,
                      val_fraction=0.2,
                      # Cnn args
                      cnn_epochs=20,
                      cnn_lr=1e-3,
                      early_stopping=False,
                      use_task_labels=False,
                      l1_pruning_proportion=None,
                      l1_reg_lambda=None,
                      dropout_rate=None,
                      weight_decay=None
                      ) -> dict:

    assert set(dataset_names).issubset(set(DATASET_LABELS.keys()))

    log_dir = './log/masking'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not logger:
        logger = util.create_logger(name='MASK', log_dir=log_dir, debug=debug)

    time_str = time.strftime("%m%d_%H%M")
    run_name = f'{algo}_{"dropout_" if dropout_rate else ""}{"".join([d[0] for d in dataset_names])}_{time_str}'
    wandb.init(name=run_name,
               project="evojax-masking",
               entity="ucl-dark",
               dir=log_dir,
               reinit=True,
               config=config_dict)

    start_time = time.time()
    logger.info('\n\nEvoJAX Masking Tests\n')
    logger.info(f'Training on the following datasets: {" ".join(dataset_names)}')

    logger.info(f'Start Time - {time.strftime("%H:%M")}')
    logger.info('=' * 50)

    if not datasets_tuple:
        datasets_tuple = full_data_loader(dataset_names=dataset_names, val_fraction=val_fraction)

    cnn_state = mask_params = None
    # full_accuracy_dict = {}
    # if not meta_learning:
    # cnn_state, full_accuracy_dict = run_mnist_training(logger,
    #                                                    seed=seed,
    #                                                    num_epochs=cnn_epochs,
    #                                                    evo_epoch=0,
    #                                                    learning_rate=cnn_lr,
    #                                                    cnn_batch_size=batch_size,
    #                                                    state=cnn_state,
    #                                                    mask_params=mask_params,
    #                                                    datasets_tuple=datasets_tuple,
    #                                                    early_stopping=early_stopping,
    #                                                    # These are the parameters for the other
    #                                                    # sparsity baseline types
    #                                                    use_task_labels=use_task_labels,
    #                                                    l1_pruning_proportion=l1_pruning_proportion,
    #                                                    l1_reg_lambda=l1_reg_lambda,
    #                                                    dropout_rate=dropout_rate,
    #                                                    weight_decay=weight_decay)

    policy = MaskPolicy(logger=logger,
                        mask_threshold=mask_threshold,
                        pixel_input=pixel_input,
                        image_mask=image_mask,
                        pretrained_cnn_state=cnn_state)

    train_task = Masking(batch_size=batch_size, validation=False, pixel_input=pixel_input, dropout_rate=dropout_rate,
                         datasets_tuple=datasets_tuple, max_steps=max_steps)
    validation_task = Masking(batch_size=batch_size, test=False, validation=True, pixel_input=pixel_input,
                              dropout_rate=dropout_rate,
                              datasets_tuple=datasets_tuple, max_steps=max_steps)
    test_task = Masking(batch_size=batch_size, test=True, validation=False, pixel_input=pixel_input,
                        dropout_rate=dropout_rate,
                        datasets_tuple=datasets_tuple, max_steps=max_steps)

    if algo == 'PGPE':
        solver = PGPE(
            pop_size=pop_size,
            param_size=policy.num_params,
            optimizer='adam',
            center_learning_rate=center_lr,
            stdev_learning_rate=std_lr,
            init_stdev=init_std,
            logger=logger,
            seed=seed,
            init_params=None
        )
    elif algo == 'OpenES':
        solver = OpenES(
            pop_size=pop_size,
            param_size=policy.num_params,
            optimizer='adam',
            init_stdev=init_std,
            logger=logger,
            seed=seed,
        )
    elif algo == 'CMA':
        solver = CMA_ES_JAX(
            param_size=policy.num_params,
            pop_size=pop_size,
            logger=logger,
            seed=seed,
            init_stdev=init_std
        )
    elif not max_iter:
        solver = None
        pass
    else:
        raise NotImplementedError

    # for i in range(evo_epochs):
    #     if meta_learning:
    #         init_seed = jax.random.PRNGKey(seed=seed+i)
    #         cnn_state = create_train_state(rng=init_seed, learning_rate=cnn_lr,
    #                                        dropout_rate=dropout_rate, weight_decay=weight_decay)
    #
    #     if i:
    #         cnn_state, accuracy_dict = run_mnist_training(logger,
    #                                                       seed=seed,
    #                                                       num_epochs=cnn_epochs,
    #                                                       evo_epoch=i+1,
    #                                                       learning_rate=cnn_lr,
    #                                                       cnn_batch_size=batch_size,
    #                                                       state=cnn_state,
    #                                                       mask_params=mask_params,
    #                                                       datasets_tuple=datasets_tuple,
    #                                                       early_stopping=early_stopping,
    #                                                       # These are the parameters for the other
    #                                                       # sparsity baseline types
    #                                                       use_task_labels=use_task_labels,
    #                                                       l1_pruning_proportion=l1_pruning_proportion,
    #                                                       l1_reg_lambda=l1_reg_lambda,
    #                                                       dropout_rate=dropout_rate)
    #
    #         # Update the full accuracy dict for that run
    #         full_accuracy_dict = {k: v+accuracy_dict[k] for k, v in full_accuracy_dict.items()}
    #
    #     policy.cnn_state = cnn_state

    # Train.

    if max_iter:
        trainer = Trainer(
            policy=policy,
            solver=solver,
            # train_task=test_task,
            train_task=validation_task,
            test_task=test_task,
            max_iter=max_iter,
            log_interval=log_interval,
            test_interval=test_interval,
            n_repeats=1,
            n_evaluations=1,
            seed=seed,
            log_dir=log_dir,
            logger=logger,
            use_for_loop=False,
        )
        best_score = trainer.run(demo_mode=False)
        best_mask_params = solver.best_params
        mask_params = policy.external_format_params_fn(best_mask_params)
    else:
        mask_params = None

    cnn_state, accuracy_dict = run_mnist_training(logger,
                                                  seed=seed,
                                                  num_epochs=cnn_epochs,
                                                  evo_epoch=0,
                                                  learning_rate=cnn_lr,
                                                  cnn_batch_size=batch_size,
                                                  state=cnn_state,
                                                  mask_params=mask_params,
                                                  datasets_tuple=datasets_tuple,
                                                  early_stopping=early_stopping,
                                                  # These are the parameters for the other
                                                  # sparsity baseline types
                                                  use_task_labels=use_task_labels,
                                                  l1_pruning_proportion=l1_pruning_proportion,
                                                  l1_reg_lambda=l1_reg_lambda,
                                                  dropout_rate=dropout_rate)

    if not pixel_input and max_iter:
        masks = policy.get_task_label_masks(mask_params)
        mean_masks = np.mean(masks, axis=1)
        for k, v in DATASET_LABELS.items():
            logger.info(f'Mean mask for {k}: {mean_masks[v]}')

    _, eval_accuracy_dict = run_mnist_training(logger,
                                               state=cnn_state,
                                               eval_only=True,
                                               mask_params=mask_params,
                                               cnn_batch_size=batch_size,
                                               datasets_tuple=datasets_tuple
                                               )

    end_time = time.time()
    logger.info(f'Total time taken: {end_time-start_time:.2f}s')
    logger.info('RUN COMPLETE\n')
    logger.info('=' * 50)

    wandb.finish()

    del train_task
    del validation_task
    del test_task

    return accuracy_dict


if __name__ == '__main__':
    config = parse_args()
    _ = run_train_masking(dataset_names=config.datasets,
                          algo=config.algo,
                          # Masking types
                          pixel_input=config.pixel_input,
                          image_mask=config.image_mask,
                          meta_learning=config.meta_learning,
                          # General params
                          pop_size=config.pop_size,
                          batch_size=config.batch_size,
                          mask_threshold=config.mask_threshold,
                          max_iter=config.max_iter,
                          max_steps=config.max_steps,
                          evo_epochs=config.evo_epochs,
                          test_interval=config.test_interval,
                          log_interval=config.log_interval,
                          center_lr=config.center_lr,
                          std_lr=config.std_lr,
                          init_std=config.init_std,
                          cnn_epochs=config.cnn_epochs,
                          cnn_lr=config.cnn_lr,
                          # These are the parameters for the other
                          # sparsity baseline types
                          early_stopping=config.early_stopping,
                          use_task_labels=config.use_task_labels,
                          l1_pruning_proportion=config.l1_pruning_proportion,
                          l1_reg_lambda=config.l1_reg_lambda,
                          dropout_rate=config.dropout_rate,
                          weight_decay=config.weight_decay,
                          # Config to pass to wandb
                          val_fraction=config.val_fraction,
                          config_dict=config)
