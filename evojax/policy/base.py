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
from typing import Tuple
import jax.numpy as jnp
import jax.random
from flax.struct import dataclass

from evojax.task.base import TaskState


@dataclass
class PolicyState(object):
    """Policy internal states."""

    keys: jnp.ndarray


class PolicyNetwork(ABC):
    """Interface for all policy networks in EvoJAX."""

    num_params: int

    def reset(self, states: TaskState) -> PolicyState:
        """Reset the policy.

        Args:
            TaskState - Initial observations.
        Returns:
            PolicyState. Policy internal states.
        """
        keys = jax.random.split(jax.random.PRNGKey(0), states.obs.shape[0])
        return PolicyState(keys=keys)

    @abstractmethod
    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        """Get vectorized actions.

        Args:
            t_states - Task states.
            params - A batch of parameters, shape is (num_envs, param_size).
            p_states - Policy internal states.
        Returns:
            jnp.ndarray. Vectorized actions.
            PolicyState. Internal states.
        """
        raise NotImplementedError()
