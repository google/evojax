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

from typing import Tuple
from PIL import Image
from PIL import ImageDraw
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask


GRAVITY = 9.82
CART_MASS = 0.5
POLE_MASS = 0.5
POLE_LEN = 0.6
FRICTION = 0.1
FORCE_SCALING = 10.0
DELTA_T = 0.01
CART_X_LIMIT = 2.4

SCREEN_W = 600
SCREEN_H = 600
CART_W = 40
CART_H = 20
VIZ_SCALE = 100
WHEEL_RAD = 5


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    state: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


def get_init_state_easy(key: jnp.ndarray) -> jnp.ndarray:
    return (random.normal(key, shape=(4,)) * 0.2 +
            jnp.array([0, 0, jnp.pi, 0]))


def get_init_state_hard(key: jnp.ndarray) -> jnp.ndarray:
    return (jnp.multiply(random.uniform(key, shape=(4,)) * 2 - 1,
                         jnp.array([CART_X_LIMIT, 10., jnp.pi / 2., 10.])) +
            jnp.array([0, 0, jnp.pi, 0]))


def get_obs(state: jnp.ndarray) -> jnp.ndarray:
    x, x_dot, theta, theta_dot = state
    return jnp.array([x, x_dot, jnp.cos(theta), jnp.sin(theta), theta_dot])


def get_reward(state: jnp.ndarray) -> jnp.float32:
    x, x_dot, theta, theta_dot = state
    reward_theta = (jnp.cos(theta) + 1.0) / 2.0
    reward_x = jnp.cos((x / CART_X_LIMIT) * (jnp.pi / 2.0))
    return reward_theta * reward_x


def update_state(action: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
    action = jnp.clip(action, -1.0, 1.0)[0] * FORCE_SCALING
    x, x_dot, theta, theta_dot = state
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    total_m = CART_MASS + POLE_MASS
    m_p_l = POLE_MASS * POLE_LEN
    x_dot_update = (
            (-2 * m_p_l * (theta_dot ** 2) * s +
             3 * POLE_MASS * GRAVITY * s * c +
             4 * action - 4 * FRICTION * x_dot) /
            (4 * total_m - 3 * POLE_MASS * c ** 2)
    )
    theta_dot_update = (
            (-3 * m_p_l * (theta_dot ** 2) * s * c +
             6 * total_m * GRAVITY * s +
             6 * (action - FRICTION * x_dot) * c) /
            (4 * POLE_LEN * total_m - 3 * m_p_l * c ** 2)
    )
    x = x + x_dot * DELTA_T
    theta = theta + theta_dot * DELTA_T
    x_dot = x_dot + x_dot_update * DELTA_T
    theta_dot = theta_dot + theta_dot_update * DELTA_T
    return jnp.array([x, x_dot, theta, theta_dot])


def out_of_screen(state: jnp.ndarray) -> jnp.float32:
    x = state[0]
    beyond_boundary_l = jnp.where(x < -CART_X_LIMIT, 1, 0)
    beyond_boundary_r = jnp.where(x > CART_X_LIMIT, 1, 0)
    return jnp.bitwise_or(beyond_boundary_l, beyond_boundary_r)


class CartPoleSwingUp(VectorizedTask):
    """Cart-pole swing up task."""

    def __init__(self,
                 max_steps: int = 1000,
                 harder: bool = False,
                 test: bool = False):

        self.max_steps = max_steps
        self.obs_shape = tuple([5, ])
        self.act_shape = tuple([1, ])
        self.test = test
        if harder:
            get_init_state_fn = get_init_state_hard
        else:
            get_init_state_fn = get_init_state_easy

        def reset_fn(key):
            next_key, key = random.split(key)
            state = get_init_state_fn(key)
            return State(state=state, obs=get_obs(state),
                         steps=jnp.zeros((), dtype=int), key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            cur_state = update_state(action=action, state=state.state)
            reward = get_reward(state=cur_state)
            steps = state.steps + 1
            done = jnp.bitwise_or(out_of_screen(cur_state), steps >= max_steps)
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            next_key, key = random.split(state.key)
            cur_state = jax.lax.cond(
                done, lambda x: get_init_state_fn(key), lambda x: x, cur_state)
            return State(state=cur_state, obs=get_obs(state=cur_state),
                         steps=steps, key=next_key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

    @staticmethod
    def render(state: State, task_id: int) -> Image:
        """Render a specified task."""
        img = Image.new('RGB', (SCREEN_W, SCREEN_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        x, _, theta, _ = np.array(state.state[task_id])
        cart_y = SCREEN_H // 2 + 100
        cart_x = x * VIZ_SCALE + SCREEN_W // 2
        # Draw the horizon.
        draw.line(
            (0, cart_y + CART_H // 2 + WHEEL_RAD,
             SCREEN_W, cart_y + CART_H // 2 + WHEEL_RAD),
            fill=(0, 0, 0), width=1)
        # Draw the cart.
        draw.rectangle(
            (cart_x - CART_W // 2, cart_y - CART_H // 2,
             cart_x + CART_W // 2, cart_y + CART_H // 2),
            fill=(255, 0, 0), outline=(0, 0, 0))
        # Draw the wheels.
        draw.ellipse(
            (cart_x - CART_W // 2 - WHEEL_RAD,
             cart_y + CART_H // 2 - WHEEL_RAD,
             cart_x - CART_W // 2 + WHEEL_RAD,
             cart_y + CART_H // 2 + WHEEL_RAD),
            fill=(220, 220, 220), outline=(0, 0, 0))
        draw.ellipse(
            (cart_x + CART_W // 2 - WHEEL_RAD,
             cart_y + CART_H // 2 - WHEEL_RAD,
             cart_x + CART_W // 2 + WHEEL_RAD,
             cart_y + CART_H // 2 + WHEEL_RAD),
            fill=(220, 220, 220), outline=(0, 0, 0))
        # Draw the pole.
        draw.line(
            (cart_x, cart_y,
             cart_x + POLE_LEN * VIZ_SCALE * np.cos(theta - np.pi / 2),
             cart_y + POLE_LEN * VIZ_SCALE * np.sin(theta - np.pi / 2)),
            fill=(0, 0, 255), width=6)
        return img
