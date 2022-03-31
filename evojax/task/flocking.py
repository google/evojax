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

"""
This flocking task is based on the following colab notebook:
https://github.com/google/jax-md/blob/main/notebooks/flocking.ipynb
"""

from PIL import Image, ImageDraw
from functools import partial
from typing import Tuple

import jax
from jax import vmap
import jax.numpy as jnp
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

DT = 0.2
SPEED = 0.12
J_ALIGN = 0.2
D_ALIGN = 0.15
J_AVOID = 0.1
D_AVOID = 0.12
J_COHESION = 0.3
D_COHESION = 0.3
ALPHA = 3

SCREEN_W = 400
SCREEN_H = 300
FISH_SIZE = 20.
BOIDS_NUM = 30
NEIGHBOR_NUM = 5


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    state: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


def sample_position(key: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    return jax.random.uniform(key, shape=(n, 2), minval=0.0, maxval=1.0)


def sample_theta(key: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    return jax.random.uniform(key, shape=(n, 1), maxval=2. * jnp.pi)


def unpack_obs(obs: jnp.ndarray) -> jnp.ndarray:
    position, theta = obs[..., :2], obs[..., 2:]
    return position, theta


def unpack_act(action: jnp.ndarray) -> jnp.ndarray:
    d_theta, d_speed = action[..., :1], action[..., 1:2]
    return d_theta, d_speed


def pack_obs(position: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([position, theta], axis=-1)


def displacement(p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    dR = p1 - p2
    return jnp.mod(dR + 0.5, 1) - 0.5


def map_product(displacement):
    return vmap(vmap(displacement, (0, None), 0), (None, 0), 0)


def calc_distance(dR: jnp.ndarray) -> jnp.ndarray:
    dr = jnp.sqrt(jnp.sum(dR**2, axis=-1))
    return dr


def select_xy(xy: jnp.ndarray, ix: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(xy, ix, axis=0)


_select_xy = jax.vmap(select_xy, in_axes=(None, 0))


def choose_neighbor(state: jnp.ndarray) -> jnp.ndarray:
    position, _ = unpack_obs(state)
    dR = map_product(displacement)(position, position)
    dr = calc_distance(dR)
    sort_idx = jnp.argsort(dr, axis=1)
    neighbor_ix = sort_idx[:, :NEIGHBOR_NUM]
    neighbors = _select_xy(state, neighbor_ix)
    return jnp.reshape(neighbors, (len(state), -1))


def normal(theta: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([jnp.cos(theta), jnp.sin(theta)], axis=-1)


def field_of_view_mask(dR, N, theta_min, theta_max):
    dr = calc_distance(dR)
    dR_hat = dR / dr
    ctheta = jnp.dot(dR_hat, N)
    return jnp.logical_and(ctheta > jnp.cos(theta_max),
                           ctheta < jnp.cos(theta_min))


def align_fn(dR, N_1, N_2, J_align, D_align, alpha):
    dr = calc_distance(dR) / D_align
    energy = J_align / alpha * (1. - dr)**alpha * (1 - jnp.dot(N_1, N_2))**2
    return jnp.where(dr < 1.0, energy, 0.)


def avoid_fn(dR, J_avoid, D_avoid, alpha):
    dr = calc_distance(dR) / D_avoid
    energy = J_avoid / alpha * (1. - dr)**alpha
    return jnp.where(dr < 1.0, energy, 0.)


def cohesion_fn(dR, N, mask, J_cohesion, D_cohesion, eps=1e-7):
    dr = calc_distance(dR)

    mask = jnp.reshape(mask, mask.shape + (1, ))
    dr = jnp.reshape(dr, dr.shape + (1, ))
    mask = jnp.logical_and(dr < D_cohesion, mask)

    N_com = jnp.where(mask, 1.0, 0)
    dR_com = jnp.where(mask, dR, 0)
    dR_com = jnp.sum(dR_com, axis=1) / (jnp.sum(N_com, axis=1) + eps)
    dR_com = dR_com / jnp.linalg.norm(dR_com + eps, axis=1, keepdims=True)
    return 0.5 * J_cohesion * (1 - jnp.sum(dR_com * N, axis=1))**2


def calc_energy(position, theta):
    E_align = partial(align_fn, J_align=J_ALIGN, D_align=D_ALIGN, alpha=ALPHA)
    E_align = vmap(vmap(E_align, (0, None, 0)), (0, 0, None))

    E_avoid = partial(avoid_fn, J_avoid=J_AVOID, D_avoid=D_AVOID, alpha=ALPHA)
    E_avoid = vmap(vmap(E_avoid))

    E_cohesion = partial(cohesion_fn,
                         J_cohesion=J_COHESION,
                         D_cohesion=D_COHESION)

    dR = map_product(displacement)(position, position)
    N = normal(theta)

    fov = partial(field_of_view_mask, theta_min=0., theta_max=jnp.pi / 3.)
    fov = vmap(vmap(fov, (0, None)))
    mask = fov(dR, N)

    return 0.5 * jnp.sum(E_align(dR, N, N) + E_avoid(dR)) + jnp.sum(
        E_cohesion(dR, N, mask))


def update_state(state, action, action_type):
    position, theta = unpack_obs(state.state)
    action = jnp.concatenate([action, jnp.ones_like(action)], axis=1)
    d_theta, d_speed = unpack_act(action)
    N = normal(theta)
    d_speed = jax.lax.cond(
        action_type,
        lambda x: (x + 1) / 2 * 0.4 + 0.8,
        lambda x: x,
        d_speed
        )
    new_obs = pack_obs(jnp.mod(position + DT * SPEED * N * d_speed, 1),
                       theta + DT * d_theta)
    return new_obs


def get_reward(state: State, max_steps: jnp.int32, reward_type: jnp.int32):
    position, theta = unpack_obs(state.state)
    reward = calc_energy(position, theta)
    reward = jax.lax.cond(
        reward_type == 0,
        lambda x: -x,
        lambda x: -x * (state.steps / max_steps) ** 2,
        reward)
    return reward


def to_pillow_coordinate(position, width, height):
    # Fix the format for drawing with pillow.
    return jnp.stack([position[:, 0] * width,
                      (1.0 - position[:, 1]) * height], axis=1)


def rotate(px, py, cx, cy, angle):
    R = jnp.array([[jnp.cos(angle), jnp.sin(angle)],
                   [-jnp.sin(angle), jnp.cos(angle)]])
    u = jnp.array([px - cx, py - cy])
    x, y = jnp.dot(R, u) + jnp.array([cx, cy])
    return x, y


def render_fishes(position, theta, width, height):
    position = to_pillow_coordinate(position, width, height)
    colors = [(88, 167, 210), (0, 153, 255), (56, 96, 123)]
    color_map = [i % len(colors) for i in range(BOIDS_NUM)]

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    fish_size = FISH_SIZE
    fish_size_half = (fish_size + 1) // 2

    for i in range(BOIDS_NUM):
        x, y = position[i]
        angle = theta[i][0]

        polygon_xy = [
            (x - fish_size_half, y - fish_size_half / 3),
            (x - fish_size_half / 5 * 3, y - fish_size_half / 4),
            (x, y - fish_size_half / 2),
            (x + fish_size_half / 3 * 2, y - fish_size_half / 3),
            (x + fish_size_half, y),
            (x + fish_size_half / 3 * 2, y + fish_size_half / 3),
            (x, y + fish_size_half / 2),
            (x - fish_size_half / 5 * 3, y + fish_size_half / 4),
            (x - fish_size_half, y + fish_size_half / 3),
        ]

        polygon_xy = [rotate(px, py, x, y, angle) for px, py in polygon_xy]
        draw.polygon(xy=polygon_xy, fill=colors[color_map[i]])
    return image


def render_single(obs_single):
    position, theta = unpack_obs(obs_single)
    image = render_fishes(position, theta, SCREEN_W * 2, SCREEN_H * 2)  # anti-aliasing
    image = image.resize((SCREEN_W, SCREEN_H), resample=Image.LANCZOS)
    return image


class FlockingTask(VectorizedTask):

    def __init__(
            self,
            max_steps: int = 150,
            reward_type: int = 0,  # (0: as it is, 1: increase rewards for late step)
            action_type: int = 0   # (0: theta, 1: theta/speed)
            ):
        self.max_steps = max_steps
        self.obs_shape = tuple([NEIGHBOR_NUM * 3, BOIDS_NUM])
        self.act_shape = tuple([action_type + 1, ])

        def reset_fn(key):
            next_key, key = jax.random.split(key)
            position = sample_position(key, BOIDS_NUM)
            theta = sample_theta(key, BOIDS_NUM)
            state = pack_obs(position, theta)
            return State(obs=choose_neighbor(state),
                         state=state,
                         steps=jnp.zeros((), dtype=jnp.int32),
                         key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            new_state = update_state(state, action, action_type)
            new_obs = choose_neighbor(new_state)
            new_steps = jnp.int32(state.steps + 1)
            next_key, _ = jax.random.split(state.key)
            reward = get_reward(state, max_steps, reward_type)
            done = jnp.where(max_steps <= new_steps, True, False)
            return State(obs=new_obs,
                         state=new_state,
                         steps=new_steps,
                         key=next_key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def step(self, state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

    def reset(self, key: jnp.ndarray):
        return self._reset_fn(key)

    @staticmethod
    def render(state: State, task_id: int) -> Image:
        return render_single(state.state[task_id])
