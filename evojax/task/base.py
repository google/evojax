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

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Tuple
from typing import TypeVar
import dataclasses

import jax.numpy as jnp
from flax.struct import dataclass


class TaskState(ABC):
    """A template of the task state."""
    obs: jnp.ndarray


class VectorizedTask(ABC):
    """Interface for all the EvoJAX tasks."""

    max_steps: int
    obs_shape: Tuple
    act_shape: Tuple
    test: bool
    multi_agent_training: bool = False

    @abstractmethod
    def reset(self, key: jnp.array) -> TaskState:
        """This resets the vectorized task.

        Args:
            key - A jax random key.
        Returns:
            TaskState. Initial task state.
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        """This steps once the simulation.

        Args:
            state - System internal states of shape (num_tasks, *).
            action - Vectorized actions of shape (num_tasks, action_size).
        Returns:
            TaskState. Task states.
            jnp.ndarray. Reward.
            jnp.ndarray. Task termination flag: 1 for done, 0 otherwise.
        """
        raise NotImplementedError()


T = TypeVar('T')


class BDExtractor(object):
    """Behavior descriptor extractor."""

    def __init__(self,
                 bd_spec: List[Tuple[str, int]],
                 bd_state_spec: List[Tuple[str, T]],
                 task_state_def: T):
        """Initialization of a behavior descriptor extractor.

        Args:
            bd_spec - A list of behavior descriptors, each of which gives the
                      name and the number of bins. E.g. [('bd1', 10), ...]
            bd_state_spec - A list of states that record rollout statistics to
                            help calculate the behavior descriptors.
                            E.g. [('bd1_stat1', jnp.int32), ...]
            task_state_def - The original task definition. BDExtractor extends
                             that definition to record data in rollouts.
        """
        self.bd_spec = bd_spec
        if bd_state_spec is None:
            self.bd_state_spec = []
        else:
            self.bd_state_spec = bd_state_spec
        self.extended_task_state = self.get_extended_task_state_def(
            task_state_def)

    def get_extended_task_state_def(self, task_state_def: T) -> T:
        """Augment the original task state definition with more entries.

        Args:
            task_state_def - This should be a flax.struct.dataclass instance.
        Returns:
            A flax.struct.dataclass type of class definition with extra entries.
        """
        if self.bd_spec is None:
            return task_state_def
        else:
            fields = []
            # Keep what is in the original task_state.
            data_fields = task_state_def.__dict__['__annotations__']
            for name, field in data_fields.items():
                fields.append((name, field))
            # Add bd fields.
            fields.extend([(x[0], jnp.int32) for x in self.bd_spec])
            # Add extra bd state fields to help the calculation.
            fields.extend(self.bd_state_spec)
            return dataclass(dataclasses.make_dataclass(
                type(task_state_def).__name__,
                fields=fields, bases=task_state_def.__bases__, init=False))

    def init_extended_state(self, task_state: TaskState) -> T:
        """Return an extended task_state that includes bd_state fields.

        Args:
            task_state - Original task state.
        Returns:
            An instance of the extended task state.
        """
        bd_fields = {x[0]: jnp.zeros((), dtype=jnp.int32) for x in self.bd_spec}
        bd_state = self.init_state(task_state)
        return self.extended_task_state(
            **bd_state,    # These keep track of the stats we need.
            **bd_fields,   # These will be updated in self.summarize.
            **task_state.__dict__)

    def init_state(self, extended_task_state: T) -> Dict[str, T]:
        """A task initializes some behavior descriptor related states here.

        Args:
            extended_task_state - An instance of the extended task state, with
                                  dummy behavior descriptor related states.
        Returns:
            A dictionary that contains the initial values for each of the
            behavior descriptor related states.
        """
        raise NotImplementedError()

    def update(self,
               extended_task_state: T,
               action: jnp.ndarray,
               reward: jnp.float32,
               done: jnp.int32) -> T:
        """Update behavior descriptor calculation states.

        Args:
            extended_task_state - An instance of extended task state.
            action - The action taken at this step.
            reward - The reward acquired at this step.
            done - The termination flag from this step.
        Returns:
            The same instance of the extended task state, but with behavior
            descriptor related states updated.
        """
        raise NotImplementedError()

    def summarize(self, extended_task_state: T) -> T:
        """Summarize the behavior descriptor related states to calculate BDs.

        Args:
            extended_task_state - An instance of the extended task state.
        Returns:
            The same instance, but with behavior descriptions calculated within.
        """
        raise NotImplementedError()
