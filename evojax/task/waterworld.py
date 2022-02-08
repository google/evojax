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

"""Implementation of the WaterWorld task in JAX.

Ref: https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html
"""

from typing import Tuple
from functools import partial
from PIL import Image
from PIL import ImageDraw
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask


SCREEN_W = 600
SCREEN_H = 600
BUBBLE_RADIUS = 10
MIN_DIST = 2 * BUBBLE_RADIUS
MAX_RANGE = 120
NUM_RANGE_SENSORS = 30
DELTA_ANG = 2 * 3.14 / NUM_RANGE_SENSORS

TYPE_VOID = 0
TYPE_WALL = 1
TYPE_FOOD = 2
TYPE_POISON = 3
TYPE_AGENT = 4
SENSOR_DATA_DIM = 5  # dist_wall, dist_food, dist_poison, item_vel_x, item_vel_y

ACT_UP = 0
ACT_DOWN = 1
ACT_LEFT = 2
ACT_RIGHT = 3


@dataclass
class BubbleStatus(object):
    pos_x: jnp.float32
    pos_y: jnp.float32
    vel_x: jnp.float32
    vel_y: jnp.float32
    bubble_type: jnp.int32
    valid: jnp.int32
    poison_cnt: jnp.int32


@dataclass
class State(TaskState):
    agent_state: BubbleStatus
    item_state: BubbleStatus
    obs: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


@partial(jax.vmap, in_axes=(0, None))
def create_bubbles(key: jnp.ndarray, is_agent: bool) -> BubbleStatus:
    k_pos_x, k_pos_y, k_vel, k_bubble_type = random.split(key, 4)
    if is_agent:
        bubble_type = TYPE_AGENT
        vel_x = vel_y = 0.
    else:
        bubble_type = jnp.where(
            random.uniform(k_bubble_type) > 0.5, TYPE_FOOD, TYPE_POISON)
        vel_x, vel_y = random.uniform(
            k_vel, shape=(2,), minval=-2.5, maxval=2.5)
    return BubbleStatus(
        pos_x=random.uniform(
            k_pos_x, shape=(), dtype=jnp.float32,
            minval=MIN_DIST, maxval=SCREEN_W - MIN_DIST),
        pos_y=random.uniform(
            k_pos_y, shape=(), dtype=jnp.float32,
            minval=MIN_DIST, maxval=SCREEN_H - MIN_DIST),
        vel_x=vel_x, vel_y=vel_y, bubble_type=bubble_type,
        valid=1, poison_cnt=0)


def get_reward(agent: BubbleStatus,
               items: BubbleStatus) -> Tuple[BubbleStatus,
                                             BubbleStatus,
                                             jnp.float32]:
    dist = jnp.sqrt(jnp.square(agent.pos_x - items.pos_x) +
                    jnp.square(agent.pos_y - items.pos_y))
    rewards = (jnp.where(items.bubble_type == TYPE_FOOD, 1., -1.) *
               items.valid * jnp.where(dist < MIN_DIST, 1, 0))
    poison_cnt = jnp.sum(jnp.where(rewards == -1., 1, 0)) + agent.poison_cnt
    reward = jnp.sum(rewards)
    items_valid = (dist >= MIN_DIST) * items.valid
    agent_state = BubbleStatus(
        pos_x=agent.pos_x, pos_y=agent.pos_y,
        vel_x=agent.vel_x, vel_y=agent.vel_y,
        bubble_type=agent.bubble_type,
        valid=agent.valid, poison_cnt=poison_cnt)
    items_state = BubbleStatus(
        pos_x=items.pos_x, pos_y=items.pos_y,
        vel_x=items.vel_x, vel_y=items.vel_y,
        bubble_type=items.bubble_type,
        valid=items_valid, poison_cnt=items.poison_cnt)
    return agent_state, items_state, reward


@jax.vmap
def update_item_state(item: BubbleStatus) -> BubbleStatus:
    vel_x = item.vel_x
    vel_y = item.vel_y
    pos_x = item.pos_x + vel_x
    pos_y = item.pos_y + vel_y
    # Collide with the west wall.
    vel_x = jnp.where(pos_x < 1, -vel_x, vel_x)
    pos_x = jnp.where(pos_x < 1, 1, pos_x)
    # Collide with the east wall.
    vel_x = jnp.where(pos_x > SCREEN_W - 1, -vel_x, vel_x)
    pos_x = jnp.where(pos_x > SCREEN_W - 1, SCREEN_W - 1, pos_x)
    # Collide with the north wall.
    vel_y = jnp.where(pos_y < 1, -vel_y, vel_y)
    pos_y = jnp.where(pos_y < 1, 1, pos_y)
    # Collide with the south wall.
    vel_y = jnp.where(pos_y > SCREEN_H - 1, -vel_y, vel_y)
    pos_y = jnp.where(pos_y > SCREEN_H - 1, SCREEN_H - 1, pos_y)
    return BubbleStatus(
        pos_x=pos_x, pos_y=pos_y, vel_x=vel_x, vel_y=vel_y,
        bubble_type=item.bubble_type, valid=item.valid,
        poison_cnt=item.poison_cnt)


def update_agent_state(agent: BubbleStatus,
                       direction: jnp.int32) -> BubbleStatus:
    vel_x = agent.vel_x
    vel_x = jnp.where(direction == ACT_RIGHT, vel_x + 1, vel_x)
    vel_x = jnp.where(direction == ACT_LEFT, vel_x - 1, vel_x)
    vel_x = vel_x * 0.95

    vel_y = agent.vel_y
    vel_y = jnp.where(direction == ACT_UP, vel_y - 1, vel_y)
    vel_y = jnp.where(direction == ACT_DOWN, vel_y + 1, vel_y)
    vel_y = vel_y * 0.95

    pos_x = agent.pos_x + vel_x
    pos_y = agent.pos_y + vel_y
    # Collide with the west wall.
    vel_x = jnp.where(pos_x < 1, 0, vel_x)
    vel_y = jnp.where(pos_x < 1, 0, vel_y)
    pos_x = jnp.where(pos_x < 1, 1, pos_x)
    # Collide with the east wall.
    vel_x = jnp.where(pos_x > SCREEN_W - 1, 0, vel_x)
    vel_y = jnp.where(pos_x > SCREEN_W - 1, 0, vel_y)
    pos_x = jnp.where(pos_x > SCREEN_W - 1, SCREEN_W - 1, pos_x)
    # Collide with the north wall.
    vel_x = jnp.where(pos_y < 1, 0, vel_x)
    vel_y = jnp.where(pos_y < 1, 0, vel_y)
    pos_y = jnp.where(pos_y < 1, 1, pos_y)
    # Collide with the south wall.
    vel_x = jnp.where(pos_y > SCREEN_H - 1, 0, vel_x)
    vel_y = jnp.where(pos_y > SCREEN_H - 1, 0, vel_y)
    pos_y = jnp.where(pos_y > SCREEN_H - 1, SCREEN_H - 1, pos_y)

    return BubbleStatus(
        pos_x=pos_x, pos_y=pos_y, vel_x=vel_x, vel_y=vel_y,
        bubble_type=agent.bubble_type, valid=agent.valid,
        poison_cnt=agent.poison_cnt)


@jax.vmap
def get_line_seg_intersection(x1: jnp.float32,
                              y1: jnp.float32,
                              x2: jnp.float32,
                              y2: jnp.float32,
                              x3: jnp.float32,
                              y3: jnp.float32,
                              x4: jnp.float32,
                              y4: jnp.float32) -> Tuple[bool, jnp.ndarray]:
    """Determine if line segment (x1, y1, x2, y2) intersects with line
    segment (x3, y3, x4, y4), and return the intersection coordinate.
    """
    denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    ua = jnp.where(
        jnp.isclose(denominator, 0.0), 0,
        ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator)
    mask1 = jnp.bitwise_and(ua > 0., ua < 1.)
    ub = jnp.where(
        jnp.isclose(denominator, 0.0), 0,
        ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator)
    mask2 = jnp.bitwise_and(ub > 0., ub < 1.)
    intersected = jnp.bitwise_and(mask1, mask2)
    x_intersection = x1 + ua * (x2 - x1)
    y_intersection = y1 + ua * (y2 - y1)
    up = jnp.where(intersected,
                   jnp.array([x_intersection, y_intersection]),
                   jnp.array([SCREEN_W, SCREEN_W]))
    return intersected, up


@jax.vmap
def get_line_dot_intersection(x1: jnp.float32,
                              y1: jnp.float32,
                              x2: jnp.float32,
                              y2: jnp.float32,
                              x3: jnp.float32,
                              y3: jnp.float32) -> Tuple[bool, jnp.ndarray]:
    """Determine if a line segment (x1, y1, x2, y2) intersects with a dot at
    (x3, y3) with radius BUBBLE_RADIUS, if so return the point of intersection.
    """
    point_xy = jnp.array([x3, y3])
    v = jnp.array([y2 - y1, x1 - x2])
    v_len = jnp.linalg.norm(v)
    d = jnp.abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1)) / v_len
    up = point_xy + v / v_len * d
    ua = jnp.where(jnp.abs(x2 - x1) > jnp.abs(y2 - y1),
                   (up[0] - x1) / (x2 - x1),
                   (up[1] - y1) / (y2 - y1))
    ua = jnp.where(d > BUBBLE_RADIUS, 0, ua)
    intersected = jnp.bitwise_and(ua > 0., ua < 1.)
    return intersected, up


def get_obs(agent: BubbleStatus,
            items: BubbleStatus,
            walls: jnp.ndarray) -> jnp.ndarray:
    sensor_obs = []
    agent_xy = jnp.array([agent.pos_x, agent.pos_y]).ravel()
    for i in jnp.arange(NUM_RANGE_SENSORS):
        ang = i * DELTA_ANG
        range_xy = jnp.array([agent.pos_x + MAX_RANGE * jnp.cos(ang),
                              agent.pos_y + MAX_RANGE * jnp.sin(ang)])
        # Check for intersections with the 4 walls.
        intersected_with_walls, wall_intersections = get_line_seg_intersection(
            jnp.ones(4) * agent_xy[0], jnp.ones(4) * agent_xy[1],
            jnp.ones(4) * range_xy[0], jnp.ones(4) * range_xy[1],
            walls[:, 0], walls[:, 1], walls[:, 2], walls[:, 3])
        dist_to_walls = jnp.where(
            intersected_with_walls,
            jnp.sqrt(jnp.square(wall_intersections[:, 1] - agent_xy[1]) +
                     jnp.square(wall_intersections[:, 0] - agent_xy[0])),
            MAX_RANGE,
        )
        ix_walls = jnp.argmin(dist_to_walls)
        # Check for intersections with the items.
        n_items = len(items.valid)
        intersected_with_items, item_intersections = get_line_dot_intersection(
            jnp.ones(n_items) * agent_xy[0], jnp.ones(n_items) * agent_xy[1],
            jnp.ones(n_items) * range_xy[0], jnp.ones(n_items) * range_xy[1],
            items.pos_x, items.pos_y)
        dist_to_items = jnp.where(
            jnp.bitwise_and(items.valid, intersected_with_items),
            jnp.sqrt(jnp.square(item_intersections[:, 1] - agent_xy[1]) +
                     jnp.square(item_intersections[:, 0] - agent_xy[0])),
            MAX_RANGE,
        )
        ix_items = jnp.argmin(dist_to_items)
        # Fill in the sensor data.
        detected_xy = jnp.where(
            jnp.min(dist_to_walls) < jnp.min(dist_to_items),
            jnp.where(
                intersected_with_walls[ix_walls],
                jnp.array([
                    dist_to_walls[ix_walls], MAX_RANGE, MAX_RANGE, 0, 0]),
                jnp.array([
                    MAX_RANGE, MAX_RANGE, MAX_RANGE, 0, 0])),
            jnp.where(
                intersected_with_items[ix_items],
                jnp.where(
                    items.bubble_type[ix_items] == TYPE_FOOD,
                    jnp.array([
                        MAX_RANGE, dist_to_items[ix_items], MAX_RANGE,
                        items.vel_x[ix_items], items.vel_y[ix_items]]),
                    jnp.array([
                        MAX_RANGE, MAX_RANGE, dist_to_items[ix_items],
                        items.vel_x[ix_items], items.vel_y[ix_items]]),
                ),
                jnp.array([
                    MAX_RANGE, MAX_RANGE, MAX_RANGE, 0, 0])),
            )
        sensor_obs.append(detected_xy)

    sensor_obs = jnp.stack(sensor_obs)
    sensor_obs = sensor_obs.at[:, :3].divide(MAX_RANGE)  # Normalized distances.
    vel_xy = jnp.array([agent.vel_x, agent.vel_y]).ravel()
    return jnp.concatenate([sensor_obs.ravel(), vel_xy], axis=0)


class WaterWorld(VectorizedTask):
    """Water world.
    ref: https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html
    """

    def __init__(self,
                 num_items: int = 50,
                 max_steps: int = 1000,
                 test: bool = False):

        self.max_steps = max_steps
        self.test = test
        self.obs_shape = tuple([NUM_RANGE_SENSORS * SENSOR_DATA_DIM + 2, ])
        self.act_shape = tuple([4, ])
        walls = jnp.array([[0, 0, 0, SCREEN_H],
                           [0, SCREEN_H, SCREEN_W, SCREEN_H],
                           [SCREEN_W, SCREEN_H, SCREEN_W, 0],
                           [SCREEN_W, 0, 0, 0]])

        def reset_fn(key):
            next_key, key = random.split(key)
            ks = random.split(key, 1 + num_items)
            agent = create_bubbles(ks[0][None, :], True)
            items = create_bubbles(ks[1:], False)
            obs = get_obs(agent, items, walls)
            return State(agent_state=agent, item_state=items, obs=obs,
                         steps=jnp.zeros((), dtype=jnp.int32), key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            next_key, key = random.split(state.key)
            direction = random.choice(
                key, 4, (), replace=False, p=action.ravel())
            agent = update_agent_state(state.agent_state, direction)
            items = update_item_state(state.item_state)
            agent, items, reward = get_reward(agent, items)
            steps = state.steps + 1
            done = jnp.where(steps >= max_steps, 1, 0)
            obs = get_obs(agent, items, walls)
            return State(agent_state=agent, item_state=items, obs=obs,
                         steps=steps, key=next_key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.array) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

    @staticmethod
    def render(state: State, task_id: int = 0) -> Image:
        img = Image.new('RGB', (SCREEN_W, SCREEN_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        state = tree_util.tree_map(lambda s: s[task_id], state)
        # Draw the agent.
        agent = state.agent_state
        x, y = agent.pos_x, agent.pos_y
        sensor_data = np.array(state.obs)[:-2].reshape(
            NUM_RANGE_SENSORS, SENSOR_DATA_DIM)
        for i, obs in enumerate(sensor_data):
            ang = i * DELTA_ANG
            dist = np.min(obs[:3])
            x_end = x + dist * MAX_RANGE * np.cos(ang)
            y_end = y + dist * MAX_RANGE * np.sin(ang)
            draw.line((x, y, x_end, y_end), fill=(0, 0, 0), width=1)
        draw.ellipse(
            (x - BUBBLE_RADIUS, y - BUBBLE_RADIUS,
             x + BUBBLE_RADIUS, y + BUBBLE_RADIUS),
            fill=(255, 255, 0), outline=(0, 0, 0))
        # Draw the items.
        items = state.item_state
        for v, t, x, y in zip(np.array(items.valid, dtype=bool),
                              np.array(items.bubble_type, dtype=int),
                              np.array(items.pos_x),
                              np.array(items.pos_y)):
            if v:
                color = (0, 255, 0) if t == TYPE_FOOD else (255, 0, 0)
                draw.ellipse(
                    (x - BUBBLE_RADIUS, y - BUBBLE_RADIUS,
                     x + BUBBLE_RADIUS, y + BUBBLE_RADIUS,),
                    fill=color, outline=(0, 0, 0))
        return img
