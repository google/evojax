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

import os
import logging
import time
from typing import Optional, Callable, Tuple

import jax.numpy as jnp
import numpy as np

import wandb

from evojax.task.base import VectorizedTask
from evojax.policy import PolicyNetwork
from evojax.algo import NEAlgorithm
from evojax.algo import QualityDiversityMethod
from evojax.sim_mgr import SimManager
from evojax.obs_norm import ObsNormalizer
from evojax.util import create_logger
from evojax.util import load_model
from evojax.util import save_model
from evojax.util import save_lattices


class Trainer(object):
    """A trainer that organizes the training logistics."""

    def __init__(self,
                 policy: PolicyNetwork,
                 solver: NEAlgorithm,
                 train_task: VectorizedTask,
                 test_task: VectorizedTask,
                 max_iter: int = 1000,
                 log_interval: int = 20,
                 val_interval: int = 10,
                 test_interval: int = 100,
                 n_repeats: int = 1,
                 test_n_repeats: int = 1,
                 n_evaluations: int = 100,
                 seed: int = 42,
                 debug: bool = False,
                 use_for_loop: bool = False,
                 normalize_obs: bool = False,
                 model_dir: str = None,
                 log_dir: str = None,
                 logger: logging.Logger = None,
                 log_scores_fn: Optional[Callable[[int, jnp.ndarray, str], None]] = None):
        """Initialization.

        Args:
            policy - The policy network to use.
            solver - The ES algorithm for optimization.
            train_task - The task for training.
            test_task - The task for evaluation.
            max_iter - Maximum number of training iterations.
            log_interval - Interval for logging.
            test_interval - Interval for tests.
            n_repeats - Number of rollout repetitions.
            n_evaluations - Number of tests to conduct.
            seed - Random seed to use.
            debug - Whether to turn on the debug flag.
            use_for_loop - Use for loop for rollouts.
            normalize_obs - Whether to use an observation normalizer.
            model_dir - Directory to save/load model.
            log_dir - Directory to dump logs.
            logger - Logger.
            log_scores_fn - custom function to log the scores array. Expects input:
                `current_iter`: int, `scores`: jnp.ndarray, 'stage': str = "train" | "test"
        """

        if logger is None:
            self._logger = create_logger(
                name='Trainer', log_dir=log_dir, debug=debug)
        else:
            self._logger = logger

        self._log_interval = log_interval
        self._val_interval = val_interval
        self._test_interval = test_interval
        self._max_iter = max_iter
        self.model_dir = model_dir
        self._log_dir = log_dir

        self._log_scores_fn = log_scores_fn or (lambda x, y, z: None)

        self._obs_normalizer = ObsNormalizer(
            obs_shape=train_task.obs_shape,
            dummy=not normalize_obs,
        )

        self.solver = solver
        self.sim_mgr = SimManager(
            n_repeats=n_repeats,
            test_n_repeats=test_n_repeats,
            pop_size=solver.pop_size,
            n_evaluations=n_evaluations,
            policy_net=policy,
            train_vec_task=train_task,
            test_vec_task=test_task,
            seed=seed,
            obs_normalizer=self._obs_normalizer,
            use_for_loop=use_for_loop,
            logger=self._logger,
        )

        # # This will store the masking network, so masks can be checked throughout training
        # self.policy_network = policy
        # self.dataset_labels = dataset_labels
        # self.masks_array = []

    def wand_log_scores(self, score_array: jnp.ndarray, split: str):
        best_score = score_array.max()
        mean_score = score_array.mean()
        worst_score = score_array.min()
        std_score = score_array.std()

        self._logger.info(
            f'[{split.upper()}] #tests={score_array.size}, max={best_score:.4f}, '
            f'avg={mean_score:.4f}, min={worst_score:.4f}, std={std_score:.4f}')

        wandb.log({f'Evo Best {split.capitalize()} accuracy': best_score,
                   f'Evo Mean {split.capitalize()} accuracy': mean_score,
                   f'Evo Worst {split.capitalize()} accuracy': worst_score,
                   f'Evo {split.capitalize()} STD': std_score})

    def run(self, demo_mode: bool = False) -> Tuple[float, jnp.ndarray]:
        """Start the training / test process."""

        if self.model_dir is not None:
            params, obs_params = load_model(model_dir=self.model_dir)
            self.sim_mgr.obs_params = obs_params
            self._logger.info(
                f'Loaded model parameters from {self.model_dir}.')
        else:
            params = None

        if demo_mode:
            if params is None:
                raise ValueError('No policy parameters to evaluate.')
            self._logger.info('Start to test the parameters.')
            scores = np.array(
                self.sim_mgr.eval_params(params=params, test=True)[0])

            self.wand_log_scores(scores, split='test')

            return scores.mean()
        else:
            self._logger.info(
                f'Start to train for {self._max_iter} iterations.')

            if params is not None:
                # Continue training from the breakpoint.
                self.solver.best_params = params

            best_score = -float('Inf')

            for i in range(self._max_iter):
                start_time = time.perf_counter()
                params = self.solver.ask()
                self._logger.debug(f'solver.ask time: {time.perf_counter() - start_time:.4f}s')

                start_time = time.perf_counter()
                scores, bds = self.sim_mgr.eval_params(params=params, test=False)
                self._logger.debug(f'sim_mgr.eval_params time: {time.perf_counter() - start_time:.4f}s')

                start_time = time.perf_counter()
                if isinstance(self.solver, QualityDiversityMethod):
                    self.solver.observe_bd(bds)
                self.solver.tell(fitness=scores)
                self._logger.debug(f'solver.tell time: {time.perf_counter() - start_time:.4f}s')

                if i > 0 and i % self._log_interval == 0:
                    scores = np.array(scores)
                    self.wand_log_scores(scores, split='train')

                    self._log_scores_fn(i, scores, "train")

                if i > 0 and i % self._test_interval == 0:
                    best_params = self.solver.best_params

                    # # Test and save the mask used for each dataset
                    # current_masks, _ = self.policy_network.get_actions(None, best_params, None)
                    # self.masks_array.append(current_masks)
                    #
                    # mean_mask = jnp.mean(current_masks, axis=1)
                    # for k, v in self.dataset_labels.items():
                    #     self._logger.info(f'[MASK] Mean mask value for {k}: {mean_mask[v]}')
                    #     wandb.log({f'Mean mask {k}': mean_mask[v]})

                    test_scores, _ = self.sim_mgr.eval_params(
                        params=best_params, test=True)

                    self.wand_log_scores(test_scores, split='test')

                    self._log_scores_fn(i, test_scores, "test")
                    mean_test_score = test_scores.mean()
                    save_model(
                        model_dir=self._log_dir,
                        model_name=f'iter_{i}',
                        params=best_params,
                        obs_params=self.sim_mgr.obs_params,
                        best=mean_test_score > best_score,
                    )
                    best_score = max(best_score, mean_test_score)

            # Test and save the final model.
            best_params = self.solver.best_params
            test_scores, _ = self.sim_mgr.eval_params(
                params=best_params, test=True)
            self._logger.info(
                f'[TEST] Iter={self._max_iter}, #tests={test_scores.size}, max={test_scores.max():.4f}, '
                f'avg={test_scores.mean():.4f}, min={test_scores.min():.4f}, std={test_scores.std():.4f}')
            mean_test_score = test_scores.mean()
            save_model(
                model_dir=self._log_dir,
                model_name='final',
                params=best_params,
                obs_params=self.sim_mgr.obs_params,
                best=mean_test_score > best_score,
            )
            best_score = max(best_score, mean_test_score)
            if isinstance(self.solver, QualityDiversityMethod):
                save_lattices(
                    log_dir=self._log_dir,
                    file_name='qd_lattices',
                    fitness_lattice=self.solver.fitness_lattice,
                    params_lattice=self.solver.params_lattice,
                    occupancy_lattice=self.solver.occupancy_lattice,
                )
            self._logger.info(
                f'Training done, best_score={best_score:.4f}')

            # Save all the masks for the run
            # time_str = time.strftime("%Y%m%d_%H%M%S")
            # save_path = os.path.join(self._log_dir, f'masks_for_run_{time_str}')
            # stacked_masks = np.stack(self.masks_array).astype('b')
            # np.savez_compressed(save_path, masks=stacked_masks)

            return best_score
