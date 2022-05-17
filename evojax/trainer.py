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

import logging
import time
from typing import Optional, Callable

import jax.numpy as jnp
import numpy as np

from evojax.task.base import VectorizedTask
from evojax.policy import PolicyNetwork
from evojax.algo import NEAlgorithm
from evojax.sim_mgr import SimManager
from evojax.obs_norm import ObsNormalizer
from evojax.util import create_logger
from evojax.util import load_model
from evojax.util import save_model


class Trainer(object):
    """A trainer that organizes the training logistics."""

    def __init__(self,
                 policy: PolicyNetwork,
                 solver: NEAlgorithm,
                 train_task: VectorizedTask,
                 test_task: VectorizedTask,
                 max_iter: int = 1000,
                 log_interval: int = 20,
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
            valid_vec_task=test_task,
            seed=seed,
            obs_normalizer=self._obs_normalizer,
            use_for_loop=use_for_loop,
            logger=self._logger,
        )

    def run(self, demo_mode: bool = False) -> float:
        """Start the training / test process."""

        if self.model_dir is not None:
            params, obs_params = load_model(model_dir=self.model_dir)
            self.sim_mgr.obs_params = obs_params
            self._logger.info(
                'Loaded model parameters from {}.'.format(self.model_dir))
        else:
            params = None

        if demo_mode:
            if params is None:
                raise ValueError('No policy parameters to evaluate.')
            self._logger.info('Start to test the parameters.')
            scores = np.array(
                self.sim_mgr.eval_params(params=params, test=True))
            self._logger.info(
                '[TEST] #tests={0}, max={1:.4f}, avg={2:.4f}, min={3:.4f}, '
                'std={4:.4f}'.format(scores.size, scores.max(), scores.mean(),
                                     scores.min(), scores.std()))
            return scores.mean()
        else:
            self._logger.info(
                'Start to train for {} iterations.'.format(self._max_iter))

            if params is not None:
                # Continue training from the breakpoint.
                self.solver.best_params = params

            best_score = -float('Inf')

            for i in range(self._max_iter):
                start_time = time.perf_counter()
                params = self.solver.ask()
                self._logger.debug('solver.ask time: {0:.4f}s'.format(
                    time.perf_counter() - start_time))

                start_time = time.perf_counter()
                scores = self.sim_mgr.eval_params(params=params, test=False)
                self._logger.debug('sim_mgr.eval_params time: {0:.4f}s'.format(
                    time.perf_counter() - start_time))

                start_time = time.perf_counter()
                self.solver.tell(fitness=scores)
                self._logger.debug('solver.tell time: {0:.4f}s'.format(
                    time.perf_counter() - start_time))

                if i > 0 and i % self._log_interval == 0:
                    scores = np.array(scores)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores.size, scores.max(), scores.mean(),
                            scores.min(), scores.std()))
                    self._log_scores_fn(i, scores, "train")

                if i > 0 and i % self._test_interval == 0:
                    best_params = self.solver.best_params
                    test_scores = self.sim_mgr.eval_params(
                        params=best_params, test=True)
                    self._logger.info(
                        '[TEST] Iter={0}, #tests={1}, max={2:.4f} avg={3:.4f}, '
                        'min={4:.4f}, std={5:.4f}'.format(
                            i, test_scores.size, test_scores.max(),
                            test_scores.mean(), test_scores.min(),
                            test_scores.std()))
                    self._log_scores_fn(i, test_scores, "test")
                    mean_test_score = test_scores.mean()
                    save_model(
                        model_dir=self._log_dir,
                        model_name='iter_{}'.format(i),
                        params=best_params,
                        obs_params=self.sim_mgr.obs_params,
                        best=mean_test_score > best_score,
                    )
                    best_score = max(best_score, mean_test_score)

            # Test and save the final model.
            best_params = self.solver.best_params
            test_scores = self.sim_mgr.eval_params(
                params=best_params, test=True)
            self._logger.info(
                '[TEST] Iter={0}, #tests={1}, max={2:.4f}, avg={3:.4f}, '
                'min={4:.4f}, std={5:.4f}'.format(
                    self._max_iter, test_scores.size, test_scores.max(),
                    test_scores.mean(), test_scores.min(), test_scores.std()))
            mean_test_score = test_scores.mean()
            save_model(
                model_dir=self._log_dir,
                model_name='final',
                params=best_params,
                obs_params=self.sim_mgr.obs_params,
                best=mean_test_score > best_score,
            )
            best_score = max(best_score, mean_test_score)
            self._logger.info(
                'Training done, best_score={0:.4f}'.format(best_score))

            return best_score
