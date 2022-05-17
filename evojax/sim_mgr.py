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
from functools import partial
from typing import Tuple
from typing import Union
import time

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map

from evojax.obs_norm import ObsNormalizer
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask
from evojax.policy.base import PolicyState
from evojax.policy.base import PolicyNetwork
from evojax.util import create_logger


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def get_task_reset_keys(key: jnp.ndarray,
                        test: bool,
                        pop_size: int,
                        n_tests: int,
                        n_repeats: int,
                        ma_training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key, subkey = random.split(key=key)
    if ma_training:
        reset_keys = random.split(subkey, n_repeats)
    else:
        if test:
            reset_keys = random.split(subkey, n_tests * n_repeats)
        else:
            reset_keys = random.split(subkey, n_repeats)
            reset_keys = jnp.tile(reset_keys, (pop_size, 1))
    return key, reset_keys


@jax.jit
def split_params_for_pmap(param: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(jnp.split(param, jax.local_device_count()))


@jax.jit
def split_states_for_pmap(
        state: Union[TaskState, PolicyState]) -> Union[TaskState, PolicyState]:
    return tree_map(split_params_for_pmap, state)


@jax.jit
def reshape_data_from_pmap(data: jnp.ndarray) -> jnp.ndarray:
    # data.shape = (#device, steps, #jobs/device, *)
    data = data.transpose([1, 0] + [i for i in range(2, data.ndim)])
    return jnp.reshape(data, (data.shape[0], data.shape[1] * data.shape[2], -1))


@partial(jax.jit, static_argnums=(1, 2))
def duplicate_params(params: jnp.ndarray,
                     repeats: int,
                     ma_training: bool) -> jnp.ndarray:
    if ma_training:
        return jnp.tile(params, (repeats, ) + (1,) * (params.ndim - 1))
    else:
        return jnp.repeat(params, repeats=repeats, axis=0)


@jax.jit
def update_score_and_mask(score, reward, mask, done):
    new_score = score + reward * mask
    new_mask = mask * (1 - done.ravel())
    return new_score, new_mask


@partial(jax.jit, static_argnums=(1,))
def report_score(scores, n_repeats):
    return jnp.mean(scores.ravel().reshape((-1, n_repeats)), axis=-1)


@jax.jit
def all_done(masks):
    return masks.sum() == 0


class SimManager(object):
    """Simulation manager."""

    def __init__(self,
                 n_repeats: int,
                 test_n_repeats: int,
                 pop_size: int,
                 n_evaluations: int,
                 policy_net: PolicyNetwork,
                 train_vec_task: VectorizedTask,
                 valid_vec_task: VectorizedTask,
                 seed: int = 0,
                 obs_normalizer: ObsNormalizer = None,
                 use_for_loop: bool = False,
                 logger: logging.Logger = None):
        """Initialization function.

        Args:
            n_repeats - Number of repeated parameter evaluations.
            pop_size - Population size.
            n_evaluations - Number of evaluations of the best parameter.
            policy_net - Policy network.
            train_vec_task - Vectorized tasks for training.
            valid_vec_task - Vectorized tasks for validation.
            seed - Random seed.
            obs_normalizer - Observation normalization helper.
            use_for_loop - Use for loop for rollout instead of jax.lax.scan.
            logger - Logger.
        """

        if logger is None:
            self._logger = create_logger(name='SimManager')
        else:
            self._logger = logger

        self._use_for_loop = use_for_loop
        self._logger.info('use_for_loop={}'.format(self._use_for_loop))
        self._key = random.PRNGKey(seed=seed)
        self._n_repeats = n_repeats
        self._test_n_repeats = test_n_repeats
        self._pop_size = pop_size
        self._n_evaluations = max(n_evaluations, jax.local_device_count())
        self._ma_training = train_vec_task.multi_agent_training

        self.obs_normalizer = obs_normalizer
        if self.obs_normalizer is None:
            self.obs_normalizer = ObsNormalizer(
                obs_shape=train_vec_task.obs_shape,
                dummy=True,
            )
        self.obs_params = self.obs_normalizer.get_init_params()

        self._num_device = jax.local_device_count()
        if self._pop_size % self._num_device != 0:
            raise ValueError(
                'pop_size must be multiples of GPU/TPUs: '
                'pop_size={}, #devices={}'.format(
                    self._pop_size, self._num_device))
        if self._n_evaluations % self._num_device != 0:
            raise ValueError(
                'n_evaluations must be multiples of GPU/TPUs: '
                'n_evaluations={}, #devices={}'.format(
                    self._n_evaluations, self._num_device))

        def step_once(carry, input_data, task):
            (task_state, policy_state, params, obs_params,
             accumulated_reward, valid_mask) = carry
            if task.multi_agent_training:
                num_tasks, num_agents = task_state.obs.shape[:2]
                task_state = task_state.replace(
                    obs=task_state.obs.reshape((-1, *task_state.obs.shape[2:])))
            org_obs = task_state.obs
            normed_obs = self.obs_normalizer.normalize_obs(org_obs, obs_params)
            task_state = task_state.replace(obs=normed_obs)
            actions, policy_state = policy_net.get_actions(
                task_state, params, policy_state)
            if task.multi_agent_training:
                task_state = task_state.replace(
                    obs=task_state.obs.reshape(
                        (num_tasks, num_agents, *task_state.obs.shape[1:])))
                actions = actions.reshape(
                    (num_tasks, num_agents, *actions.shape[1:]))
            task_state, reward, done = task.step(task_state, actions)
            if task.multi_agent_training:
                reward = reward.ravel()
                done = jnp.repeat(done, num_agents, axis=0)
            accumulated_reward = accumulated_reward + reward * valid_mask
            valid_mask = valid_mask * (1 - done.ravel())
            return ((task_state, policy_state, params, obs_params,
                     accumulated_reward, valid_mask),
                    (org_obs, valid_mask))

        def rollout(task_states, policy_states, params, obs_params,
                    step_once_fn, max_steps):
            accumulated_rewards = jnp.zeros(params.shape[0])
            valid_masks = jnp.ones(params.shape[0])
            ((task_states, policy_states, params, obs_params,
              accumulated_rewards, valid_masks),
             (obs_set, obs_mask)) = jax.lax.scan(
                step_once_fn,
                (task_states, policy_states, params, obs_params,
                 accumulated_rewards, valid_masks), (), max_steps)
            return accumulated_rewards, obs_set, obs_mask

        self._policy_reset_fn = jax.jit(policy_net.reset)
        self._policy_act_fn = jax.jit(policy_net.get_actions)

        # Set up training functions.
        self._train_reset_fn = train_vec_task.reset
        self._train_step_fn = train_vec_task.step
        self._train_max_steps = train_vec_task.max_steps
        self._train_rollout_fn = partial(
            rollout,
            step_once_fn=partial(step_once, task=train_vec_task),
            max_steps=train_vec_task.max_steps)
        if self._num_device > 1:
            self._train_rollout_fn = jax.jit(jax.pmap(
                self._train_rollout_fn, in_axes=(0, 0, 0, None)))

        # Set up validation functions.
        self._valid_reset_fn = valid_vec_task.reset
        self._valid_step_fn = valid_vec_task.step
        self._valid_max_steps = valid_vec_task.max_steps
        self._valid_rollout_fn = partial(
            rollout,
            step_once_fn=partial(step_once, task=valid_vec_task),
            max_steps=valid_vec_task.max_steps)
        if self._num_device > 1:
            self._valid_rollout_fn = jax.jit(jax.pmap(
                self._valid_rollout_fn, in_axes=(0, 0, 0, None)))

    def eval_params(self, params: jnp.ndarray, test: bool) -> jnp.ndarray:
        """Evaluate population parameters or test the best parameter.

        Args:
            params - Parameters to be evaluated.
            test - Whether we are testing the best parameter
        Returns:
            An array of fitness scores.
        """
        if self._use_for_loop:
            return self._for_loop_eval(params, test)
        else:
            return self._scan_loop_eval(params, test)

    def _for_loop_eval(self, params: jnp.ndarray, test: bool) -> jnp.ndarray:
        """Rollout using for loop (no multi-device or ma_training yet)."""
        policy_reset_func = self._policy_reset_fn
        policy_act_func = self._policy_act_fn
        if test:
            n_repeats = self._test_n_repeats
            task_reset_func = self._valid_reset_fn
            task_step_func = self._valid_step_fn
            task_max_steps = self._valid_max_steps
            params = duplicate_params(
                params[None, :], self._n_evaluations, False)
        else:
            n_repeats = self._n_repeats
            task_reset_func = self._train_reset_fn
            task_step_func = self._train_step_fn
            task_max_steps = self._train_max_steps

        params = duplicate_params(params, n_repeats, self._ma_training)

        # Start rollout.
        self._key, reset_keys = get_task_reset_keys(
            self._key, test, self._pop_size, self._n_evaluations, n_repeats,
            self._ma_training)
        task_state = task_reset_func(reset_keys)
        policy_state = policy_reset_func(task_state)
        scores = jnp.zeros(params.shape[0])
        valid_mask = jnp.ones(params.shape[0])
        start_time = time.perf_counter()
        rollout_steps = 0
        sim_steps = 0
        for i in range(task_max_steps):
            actions, policy_state = policy_act_func(
                task_state, params, policy_state)
            task_state, reward, done = task_step_func(task_state, actions)
            scores, valid_mask = update_score_and_mask(
                scores, reward, valid_mask, done)
            rollout_steps += 1
            sim_steps = sim_steps + valid_mask
            if all_done(valid_mask):
                break
        time_cost = time.perf_counter() - start_time
        self._logger.debug('{} steps/s, mean.steps={}'.format(
            int(rollout_steps * task_state.obs.shape[0] / time_cost),
            sim_steps.sum() / task_state.obs.shape[0]))

        return report_score(scores, n_repeats)

    def _scan_loop_eval(self, params: jnp.ndarray, test: bool) -> jnp.ndarray:
        """Rollout using jax.lax.scan."""
        policy_reset_func = self._policy_reset_fn
        if test:
            n_repeats = self._test_n_repeats
            task_reset_func = self._valid_reset_fn
            rollout_func = self._valid_rollout_fn
            params = duplicate_params(
                params[None, :], self._n_evaluations, False)
        else:
            n_repeats = self._n_repeats
            task_reset_func = self._train_reset_fn
            rollout_func = self._train_rollout_fn

        # Suppose pop_size=2 and n_repeats=3.
        # For multi-agents training, params become
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        # For non-ma training, params become
        #   a1, a2, ..., an  (individual 1 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   b1, b2, ..., bn  (individual 2 params)
        params = duplicate_params(params, n_repeats, self._ma_training)

        self._key, reset_keys = get_task_reset_keys(
            self._key, test, self._pop_size, self._n_evaluations, n_repeats,
            self._ma_training)

        # Reset the tasks and the policy.
        task_state = task_reset_func(reset_keys)
        policy_state = policy_reset_func(task_state)
        if self._num_device > 1:
            params = split_params_for_pmap(params)
            task_state = split_states_for_pmap(task_state)
            policy_state = split_states_for_pmap(policy_state)

        # Do the rollouts.
        scores, all_obs, masks = rollout_func(
            task_state, policy_state, params, self.obs_params)
        if self._num_device > 1:
            all_obs = reshape_data_from_pmap(all_obs)
            masks = reshape_data_from_pmap(masks)

        if not test and not self.obs_normalizer.is_dummy:
            self.obs_params = self.obs_normalizer.update_normalization_params(
                obs_buffer=all_obs, obs_mask=masks, obs_params=self.obs_params)

        if self._ma_training:
            if not test:
                # In training, each agent has different parameters.
                return jnp.mean(scores.ravel().reshape((n_repeats, -1)), axis=0)
            else:
                # In tests, they share the same parameters.
                return jnp.mean(scores.ravel().reshape((n_repeats, -1)), axis=1)
        else:
            return jnp.mean(scores.ravel().reshape((-1, n_repeats)), axis=-1)
