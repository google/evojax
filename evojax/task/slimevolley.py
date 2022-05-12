# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the Slime Volley environment in EvoJAX

Slime Volleyball is a game created in the early 2000s by unknown author.

The game is very simple: the agent's goal is to get the ball to land on
the ground of its opponent's side, causing its opponent to lose a life.

Each agent starts off with five lives. The episode ends when either agent
loses all five lives, or after 3000 timesteps has passed. An agent receives
a reward of +1 when its opponent loses or -1 when it loses a life.

An agent loses when it loses 5 times in the Test environment, or if it
loses based on score count after 3000 time steps.

During Training, the game is simply played for 3000 time steps, not
terminating even when one player loses 5 times.

This task is based on:
https://otoro.net/slimevolley/

The implementation is based on:
https://github.com/hardmaru/slimevolleygym
"""

import math
import numpy as np

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask

import cv2

from PIL import Image

# game settings:

RENDER_MODE = True

REF_W = 24*2
REF_H = REF_W
REF_U = 1.5  # ground height
REF_WALL_WIDTH = 1.0  # wall width
REF_WALL_HEIGHT = 3.5
PLAYER_SPEED_X = 10*1.75
PLAYER_SPEED_Y = 10*1.35
MAX_BALL_SPEED = 15*1.5
TIMESTEP = 1/30.
NUDGE = 0.1
FRICTION = 1.0  # 1 means no FRICTION, less means FRICTION
GRAVITY = -9.8*2*1.5

MAXLIVES = 5  # game ends when one agent loses this many games

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 500

FACTOR = WINDOW_WIDTH / REF_W

# if set to true, renders using cv2 directly on numpy array
# (otherwise uses pyglet / opengl -> much smoother for human player)
PIXEL_MODE = True
PIXEL_SCALE = 2  # Render at multiple of Pixel Obs resolution, then downscale.

PIXEL_WIDTH = 84*2*2
PIXEL_HEIGHT = 84*2


def setNightColors():
    # night time color:
    global BALL_COLOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR
    global PIXEL_AGENT_LEFT_COLOR, PIXEL_AGENT_RIGHT_COLOR
    global BACKGROUND_COLOR, FENCE_COLOR, COIN_COLOR, GROUND_COLOR
    BALL_COLOR = (217, 79, 0)
    AGENT_LEFT_COLOR = (35, 93, 188)
    AGENT_RIGHT_COLOR = (255, 236, 0)
    PIXEL_AGENT_LEFT_COLOR = (0, 87, 184)  # AZURE BLUE
    PIXEL_AGENT_RIGHT_COLOR = (254, 221, 0)  # YELLOW

    BACKGROUND_COLOR = (11, 16, 19)
    FENCE_COLOR = (102, 56, 35)
    COIN_COLOR = FENCE_COLOR
    GROUND_COLOR = (116, 114, 117)


def setDayColors():
    # day time color:
    # note: do not use day time colors for pixel-obs training.
    global BALL_COLOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR
    global PIXEL_AGENT_LEFT_COLOR, PIXEL_AGENT_RIGHT_COLOR
    global BACKGROUND_COLOR, FENCE_COLOR, COIN_COLOR, GROUND_COLOR
    global PIXEL_SCALE, PIXEL_WIDTH, PIXEL_HEIGHT
    PIXEL_SCALE = int(4*1.0)
    PIXEL_WIDTH = int(84*2*1.0)
    PIXEL_HEIGHT = int(84*1.0)
    BALL_COLOR = (255, 200, 20)
    AGENT_LEFT_COLOR = (240, 75, 0)
    AGENT_RIGHT_COLOR = (0, 150, 255)
    PIXEL_AGENT_LEFT_COLOR = (240, 75, 0)
    PIXEL_AGENT_RIGHT_COLOR = (0, 150, 255)

    BACKGROUND_COLOR = (255, 255, 255)
    FENCE_COLOR = (240, 210, 130)
    COIN_COLOR = FENCE_COLOR
    GROUND_COLOR = (128, 227, 153)


setNightColors()


def setPixelObsMode():
    """
    used for experimental pixel-observation mode
    note: new dim's chosen to be PIXEL_SCALE (2x) as Pixel Obs dims
          (will be downsampled)
    """
    global WINDOW_WIDTH, WINDOW_HEIGHT, FACTOR
    global AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR, PIXEL_MODE
    PIXEL_MODE = True
    WINDOW_WIDTH = PIXEL_WIDTH * PIXEL_SCALE
    WINDOW_HEIGHT = PIXEL_HEIGHT * PIXEL_SCALE
    FACTOR = WINDOW_WIDTH / REF_W
    AGENT_LEFT_COLOR = PIXEL_AGENT_LEFT_COLOR
    AGENT_RIGHT_COLOR = PIXEL_AGENT_RIGHT_COLOR


setPixelObsMode()


def upsize_image(img):
    return cv2.resize(img, (PIXEL_WIDTH*PIXEL_SCALE, PIXEL_HEIGHT*PIXEL_SCALE),
                      interpolation=cv2.INTER_NEAREST)


def downsize_image(img):
    return cv2.resize(img, (PIXEL_WIDTH, PIXEL_HEIGHT),
                      interpolation=cv2.INTER_AREA)


# conversion from space to pixels (allows us to render to diff resolutions)
def toX(x):
    return (x+REF_W/2)*FACTOR


def toP(x):
    return (x)*FACTOR


def toY(y):
    return y*FACTOR


def create_canvas(c):
    result = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    for channel in range(3):
        result[:, :, channel] *= c[channel]
    return result


def rect(canvas, x, y, width, height, color):
    """ Processing style function to make it easy to port p5.js to python """
    return cv2.rectangle(canvas, (round(x), round(WINDOW_HEIGHT-y)),
                         (round(x+width), round(WINDOW_HEIGHT-y+height)),
                         color, thickness=-1, lineType=cv2.LINE_AA)


def half_circle(canvas, x, y, r, color):
    """ Processing style function to make it easy to port p5.js to python """
    return cv2.ellipse(canvas, (round(x), WINDOW_HEIGHT-round(y)),
                       (round(r), round(r)), 0, 0, -180, color, thickness=-1,
                       lineType=cv2.LINE_AA)


def circle(canvas, x, y, r, color):
    """ Processing style function to make it easy to port p5.js to python """
    return cv2.circle(canvas, (round(x), round(WINDOW_HEIGHT-y)), round(r),
                      color, thickness=-1, lineType=cv2.LINE_AA)


@dataclass
class BaselinePolicyParams(object):
    w: jnp.ndarray
    b: jnp.ndarray


def initBaselinePolicyParams():
    nGameInput = 8  # 8 states for agent
    nGameOutput = 3  # 3 buttons (forward, backward, jump)
    nRecurrentState = 4  # extra recurrent states for feedback.
    """See training details:
    https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
    """
    weight = jnp.array(
        [7.5719, 4.4285, 2.2716, -0.3598, -7.8189, -2.5422, -3.2034, 0.3935,
         1.2202, -0.49, -0.0316, 0.5221, 0.7026, 0.4179, -2.1689, 1.646,
         -13.3639, 1.5151, 1.1175, -5.3561, 5.0442, 0.8451, 0.3987, -2.9501,
         -3.7811, -5.8994, 6.4167, 2.5014, 7.338, -2.9887, 2.4586, 13.4191,
         2.7395, -3.9708, 1.6548, -2.7554, -1.5345, -6.4708, 9.2426, -0.7392,
         0.4452, 1.8828, -2.6277, -10.851, -3.2353, -4.4653, -3.1153, -1.3707,
         7.318, 16.0902, 1.4686, 7.0391, 1.7765, -1.155, 2.6697, -8.8877,
         1.1958, -3.2839, -5.4425, 1.6809, 7.6812, -2.4732, 1.738, 0.3781,
         0.8718, 2.5886, 1.6911, 1.2953, -9.0052, -4.6038, -6.7447, -2.5528,
         0.4391, -4.9278, -3.6695, -4.8673, -1.6035, 1.5011, -5.6124, 4.9747,
         1.8998, 3.0359, 6.2983, -4.8568, -2.1888, -4.1143, -3.9874, -0.0459,
         4.7134, 2.8952, -9.3627, -4.685, 0.3601, -1.3699, 9.7294, 11.5596,
         0.1918, 3.0783, 0.0329, -0.1362, -0.1188, -0.7579, 0.3278, -0.977,
         -0.9377])
    weight = weight.reshape(nGameOutput+nRecurrentState,
                            nGameInput+nGameOutput+nRecurrentState)
    bias = jnp.array([2.2935, -2.0353, -1.7786, 5.4567,
                      -3.6368, 3.4996, -0.0685])

    return BaselinePolicyParams(weight, bias)


def initBaselinePolicyState():
    return jnp.zeros(7)


@dataclass
class ParticleState(object):
    x: jnp.float32
    y: jnp.float32
    prev_x: jnp.float32
    prev_y: jnp.float32
    vx: jnp.float32
    vy: jnp.float32
    r: jnp.float32


def initParticleState(x, y, vx, vy, r):
    return ParticleState(jnp.float32(x), jnp.float32(y),
                         jnp.float32(x), jnp.float32(y),
                         jnp.float32(vx), jnp.float32(vy),
                         jnp.float32(r))


@dataclass
class AgentState(object):
    direction: jnp.int32  # -1 means left, 1 means right player.
    x: jnp.float32
    y: jnp.float32
    r: jnp.float32
    vx: jnp.float32
    vy: jnp.float32
    desired_vx: jnp.float32
    desired_vy: jnp.float32
    life: jnp.int32


def initAgentState(direction, x, y):
    return AgentState(direction, x, y, 1.5, 0, 0, 0, 0, MAXLIVES)


@dataclass
class GameState(object):
    ball: ParticleState
    agent_left: AgentState
    agent_right: AgentState
    hidden_left: jnp.ndarray  # rnn hidden state for internal policy
    hidden_right: jnp.ndarray
    action_left_flag: jnp.int32  # if 1, then use the action action_left
    action_left: jnp.ndarray
    action_right_flag: jnp.int32  # same as above
    action_right: jnp.ndarray


@dataclass
class State(TaskState):
    game_state: GameState
    obs: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


@dataclass
class Observation(object):  # is also the "RelativeState" in the original code
    """
    keeps track of the obs.
    Note: the observation is from the perspective of the agent.
    an agent playing either side of the fence must see obs the same way
    """
    x: jnp.float32  # agent
    y: jnp.float32
    vx: jnp.float32
    vy: jnp.float32
    bx: jnp.float32  # ball
    by: jnp.float32
    bvx: jnp.float32
    bvy: jnp.float32
    ox: jnp.float32  # opponent
    oy: jnp.float32
    ovx: jnp.float32
    ovy: jnp.float32


def getZeroObs() -> Observation:
    return Observation(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def getObsArray(rs: Observation):
    # scale inputs to be in the order
    # of magnitude of 10 for neural network. (legacy)
    scaleFactor = 10.0
    result = jnp.array([rs.x, rs.y, rs.vx, rs.vy,
                        rs.bx, rs.by, rs.bvx, rs.bvy,
                        rs.ox, rs.oy, rs.ovx, rs.ovy]) / scaleFactor
    return result


class Particle:
    """ used for the ball, and also for the round stub above the fence """
    def __init__(self, p: ParticleState, c):
        self.p = p
        self.c = c

    def display(self, canvas):
        return circle(canvas, toX(float(self.p.x)), toY(float(self.p.y)),
                      toP(float(self.p.r)), color=self.c)

    def move(self):
        self.p = ParticleState(self.p.x+self.p.vx*TIMESTEP,
                               self.p.y+self.p.vy*TIMESTEP,
                               self.p.x, self.p.y,
                               self.p.vx, self.p.vy, r=self.p.r)

    def applyAcceleration(self, ax, ay):
        self.p = ParticleState(self.p.x, self.p.y,
                               self.p.prev_x, self.p.prev_y,
                               self.p.vx+ax*TIMESTEP, self.p.vy+ay*TIMESTEP,
                               r=self.p.r)

    def checkEdges(self):
        oldp = self.p
        return_sign = jnp.where(oldp.x <= 0, -1, 1)
        newx = oldp.x
        newy = oldp.y
        newpx = oldp.prev_x
        newpy = oldp.prev_y
        newvx = oldp.vx
        newvy = oldp.vy

        newx = jnp.where(oldp.x <= (oldp.r-REF_W/2),
                         oldp.r-REF_W/2+NUDGE*TIMESTEP, newx)
        newvx = jnp.where(oldp.x <= (oldp.r-REF_W/2),
                          oldp.vx*(-FRICTION), newvx)

        newx = jnp.where(oldp.x >= (REF_W/2-oldp.r),
                         REF_W/2-oldp.r-NUDGE*TIMESTEP, newx)
        newvx = jnp.where(oldp.x >= (REF_W/2-oldp.r),
                          oldp.vx*(-FRICTION), newvx)

        return_value = jnp.where(oldp.y <= (oldp.r+REF_U), 1, 0)

        newy = jnp.where(oldp.y <= (oldp.r+REF_U),
                         oldp.r+REF_U+NUDGE*TIMESTEP, newy)
        newvy = jnp.where(oldp.y <= (oldp.r+REF_U),
                          oldp.vy*(-FRICTION), newvy)

        newy = jnp.where(oldp.y >= (REF_H-oldp.r),
                         REF_H-oldp.r-NUDGE*TIMESTEP, newy)

        newvy = jnp.where(oldp.y >= (REF_H-oldp.r),
                          oldp.vy*(-FRICTION), newvy)

        # fence:

        newx = jnp.where((oldp.x <= (REF_WALL_WIDTH/2+oldp.r)) &
                         (oldp.prev_x > (REF_WALL_WIDTH/2+oldp.r)) &
                         (oldp.y <= REF_WALL_HEIGHT),
                         REF_WALL_WIDTH/2+oldp.r+NUDGE*TIMESTEP, newx)
        newvx = jnp.where((oldp.x <= (REF_WALL_WIDTH/2+oldp.r)) &
                          (oldp.prev_x > (REF_WALL_WIDTH/2+oldp.r)) &
                          (oldp.y <= REF_WALL_HEIGHT),
                          oldp.vx*(-FRICTION), newvx)

        newx = jnp.where((oldp.x >= (-REF_WALL_WIDTH/2-oldp.r)) &
                         (oldp.prev_x < (-REF_WALL_WIDTH/2-oldp.r)) &
                         (oldp.y <= REF_WALL_HEIGHT),
                         -REF_WALL_WIDTH/2-oldp.r-NUDGE*TIMESTEP, newx)
        newvx = jnp.where((oldp.x >= (-REF_WALL_WIDTH/2-oldp.r)) &
                          (oldp.prev_x < (-REF_WALL_WIDTH/2-oldp.r)) &
                          (oldp.y <= REF_WALL_HEIGHT),
                          oldp.vx*(-FRICTION), newvx)

        self.p = ParticleState(newx, newy, newpx, newpy, newvx, newvy, oldp.r)
        return return_value*return_sign

    def bounce(self, p):  # bounce two balls that have collided (this and that)
        oldp = self.p
        abx = oldp.x-p.x
        aby = oldp.y-p.y
        abd = jnp.sqrt(abx*abx+aby*aby)
        abx /= abd  # normalize
        aby /= abd
        nx = abx  # reuse calculation
        ny = aby
        abx *= NUDGE
        aby *= NUDGE

        new_y = oldp.y
        new_x = oldp.x
        dy = (new_y - p.x)
        dx = (new_x - p.y)

        total_r = oldp.r+p.r
        total_r2 = total_r*total_r

        # this was a while loop in the orig code, but most cases < 15.
        for i in range(15):
            total_d2 = (dy*dy + dx*dx)
            new_x = jnp.where(total_d2 < total_r2, new_x+abx, new_x)
            new_y = jnp.where(total_d2 < total_r2, new_y+aby, new_y)
            dy = (p.y - new_y)
            dx = (p.x - new_x)

        ux = oldp.vx - p.vx
        uy = oldp.vy - p.vy
        un = ux*nx + uy*ny
        unx = nx*(un*2.)  # added factor of 2
        uny = ny*(un*2.)  # added factor of 2
        ux -= unx
        uy -= uny
        return ParticleState(x=new_x, y=new_y,
                             prev_x=oldp.prev_x, prev_y=oldp.prev_y,
                             vx=ux + p.vx, vy=uy + p.vy, r=oldp.r)

    def bounceIfColliding(self, p):
        dy = p.y - self.p.y
        dx = p.x - self.p.x
        d2 = (dx*dx+dy*dy)
        r = self.p.r+p.r
        r2 = r*r
        newp = self.bounce(p)

        # make if condition work with jax:
        newx = jnp.where(d2 < r2, newp.x, self.p.x)
        newy = jnp.where(d2 < r2, newp.y, self.p.y)
        newprev_x = jnp.where(d2 < r2, newp.prev_x, self.p.prev_x)
        newprev_y = jnp.where(d2 < r2, newp.prev_y, self.p.prev_y)
        newvx = jnp.where(d2 < r2, newp.vx, self.p.vx)
        newvy = jnp.where(d2 < r2, newp.vy, self.p.vy)
        self.p = ParticleState(x=newx, y=newy,
                               prev_x=newprev_x, prev_y=newprev_y,
                               vx=newvx, vy=newvy, r=self.p.r)

    def limitSpeed(self, maxSpeed):
        oldp = self.p
        mag2 = oldp.vx*oldp.vx+oldp.vy*oldp.vy
        mag = jnp.sqrt(mag2)

        newvx = oldp.vx
        newvy = oldp.vy
        newvx = jnp.where(mag2 > (maxSpeed*maxSpeed), newvx/mag, newvx)
        newvy = jnp.where(mag2 > (maxSpeed*maxSpeed), newvy/mag, newvy)
        newvx = jnp.where(mag2 > (maxSpeed*maxSpeed), newvx*maxSpeed, newvx)
        newvy = jnp.where(mag2 > (maxSpeed*maxSpeed), newvy*maxSpeed, newvy)

        self.p = ParticleState(x=oldp.x, y=oldp.y,
                               prev_x=oldp.prev_x, prev_y=oldp.prev_y,
                               vx=newvx, vy=newvy, r=oldp.r)


class Agent:
    """ keeps track of the agent in the game. note: not the policy network """
    def __init__(self, agent, c):
        self.p = agent
        self.state = getZeroObs()
        self.c = c

    def setAction(self, action):
        forward = jnp.int32(0)
        backward = jnp.int32(0)
        jump = jnp.int32(0)

        forward = jnp.where(action[0] > 0, 1, forward)
        backward = jnp.where(action[1] > 0, 1, backward)
        jump = jnp.where(action[2] > 0, 1, jump)

        new_desired_vx = jnp.float32(0.0)
        new_desired_vy = jnp.float32(0.0)

        new_desired_vx = jnp.where(forward & (1-backward),
                                   -PLAYER_SPEED_X, new_desired_vx)
        new_desired_vx = jnp.where(backward & (1-forward),
                                   PLAYER_SPEED_X, new_desired_vx)

        new_desired_vy = jnp.where(jump, PLAYER_SPEED_Y, new_desired_vy)

        p = self.p
        self.p = AgentState(p.direction, p.x, p.y, p.r, p.vx, p.vy,
                            new_desired_vx, new_desired_vy, p.life)

    def move(self):
        p = self.p
        self.p = AgentState(p.direction, p.x+p.vx*TIMESTEP, p.y+p.vy*TIMESTEP,
                            p.r, p.vx, p.vy,
                            p.desired_vx, p.desired_vy,
                            p.life)

    def update(self):
        p = self.p
        new_vy = p.vy + GRAVITY*TIMESTEP

        new_vy = jnp.where(p.y <= REF_U+NUDGE*TIMESTEP, p.desired_vy, new_vy)

        new_vx = p.desired_vx*p.direction

        self.p = AgentState(p.direction, p.x, p.y, p.r, new_vx, new_vy,
                            p.desired_vx, p.desired_vy, p.life)

        self.move()

        p = self.p

        # stay in their own half:

        new_y = p.y
        new_vy = p.vy
        new_y = jnp.where(p.y <= REF_U, REF_U, new_y)
        new_vy = jnp.where(p.y <= REF_U, 0, new_vy)

        # stay in their own half:
        new_vx = p.vx
        new_x = p.x

        new_vx = jnp.where(p.x*p.direction <= (REF_WALL_WIDTH/2+p.r),
                           0, new_vx)
        new_x = jnp.where(p.x*p.direction <= (REF_WALL_WIDTH/2+p.r),
                          p.direction*(REF_WALL_WIDTH/2+p.r), new_x)

        new_vx = jnp.where(p.x*p.direction >= (REF_W/2-p.r), 0, new_vx)
        new_x = jnp.where(p.x*p.direction >= (REF_W/2-p.r),
                          p.direction*(REF_W/2-p.r), new_x)

        self.p = AgentState(p.direction, new_x, new_y, p.r,
                            new_vx, new_vy, p.desired_vx, p.desired_vy,
                            p.life)

    def updateLife(self, result):
        """ updates the life based on result and internal direction """
        p = self.p
        updateAmount = p.direction*result  # only update if this value is -1
        new_life = jnp.where(updateAmount < 0, p.life-1, p.life)
        self.p = AgentState(p.direction, p.x, p.y, p.r, p.vx, p.vy,
                            p.desired_vx, p.desired_vy, new_life)

    def updateState(self, ball: ParticleState, opponent: AgentState):
        """ normalized to side, customized for each agent's perspective"""
        p = self.p
        # agent's self
        x = p.x*p.direction
        y = p.y
        vx = p.vx*p.direction
        vy = p.vy
        # ball
        bx = ball.x*p.direction
        by = ball.y
        bvx = ball.vx*p.direction
        bvy = ball.vy
        # opponent
        ox = opponent.x*(-p.direction)
        oy = opponent.y
        ovx = opponent.vx*(-p.direction)
        ovy = opponent.vy

        self.state = Observation(x, y, vx, vy, bx, by,
                                 bvx, bvy, ox, oy, ovx, ovy)

    def getObservation(self):
        return getObsArray(self.state)

    def display(self, canvas, ball_x, ball_y):
        bx = float(ball_x)
        by = float(ball_y)
        p = self.p
        x = float(p.x)
        y = float(p.y)
        r = float(p.r)
        direction = int(p.direction)

        angle = math.pi * 60 / 180
        if direction == 1:
            angle = math.pi * 120 / 180
        eyeX = 0
        eyeY = 0

        canvas = half_circle(canvas, toX(x), toY(y), toP(r), color=self.c)

        # track ball with eyes (replace with observed info later):
        c = math.cos(angle)
        s = math.sin(angle)
        ballX = bx-(x+(0.6)*r*c)
        ballY = by-(y+(0.6)*r*s)

        dist = math.sqrt(ballX*ballX+ballY*ballY)
        eyeX = ballX/dist
        eyeY = ballY/dist

        canvas = circle(canvas, toX(x+(0.6)*r*c), toY(y+(0.6)*r*s),
                        toP(r)*0.3, color=(255, 255, 255))
        canvas = circle(canvas, toX(x+(0.6)*r*c+eyeX*0.15*r),
                        toY(y+(0.6)*r*s+eyeY*0.15*r), toP(r)*0.1,
                        color=(0, 0, 0))

        # draw coins (lives) left
        num_lives = int(p.life)
        for i in range(1, num_lives):
            canvas = circle(canvas, toX(direction*(REF_W/2+0.5-i*2.)),
                            WINDOW_HEIGHT-toY(1.5), toP(0.5),
                            color=COIN_COLOR)

        return canvas


class Wall:
    """ used for the fence, and also the ground """
    def __init__(self, x, y, w, h, c):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c

    def display(self, canvas):
        return rect(canvas, toX(self.x-self.w/2), toY(self.y+self.h/2),
                    toP(self.w), toP(self.h), color=self.c)


def baselinePolicy(obs: jnp.ndarray, state: jnp.ndarray,
                   params: BaselinePolicyParams):
    """ take obs, prev rnn state, return updated rnn state, action """
    nGameInput = 8  # 8 states that policy cares about (ignores last 4)
    nGameOutput = 3  # 3 buttons (forward, backward, jump)

    weight = params.w
    bias = params.b
    inputState = jnp.concatenate([obs[:nGameInput], state])
    outputState = jnp.tanh(jnp.dot(weight, inputState)+bias)
    action = jnp.zeros(nGameOutput)
    action = jnp.where(outputState[:nGameOutput] > 0.75, 1, action)
    return outputState, action


class Game:
    """
    the main slime volley game.
    can be used in various settings,
    such as ai vs ai, ai vs human, human vs human
    """
    def __init__(self, gameState):
        self.baselineParams = initBaselinePolicyParams()
        self.ground = None
        self.fence = None
        self.fenceStub = None
        self.reset(gameState)

    def reset(self, gameState):
        self.ground = Wall(0, 0.75, REF_W, REF_U, c=GROUND_COLOR)
        self.fence = Wall(0, 0.75 + REF_WALL_HEIGHT/2, REF_WALL_WIDTH,
                          (REF_WALL_HEIGHT-1.5), c=FENCE_COLOR)
        fenceStubParticle = initParticleState(0, REF_WALL_HEIGHT,
                                              0, 0, REF_WALL_WIDTH/2)
        self.fenceStub = Particle(fenceStubParticle, c=FENCE_COLOR)
        self.setGameState(gameState)

    def setGameState(self, gameState):
        self.ball = Particle(gameState.ball, c=BALL_COLOR)
        self.agent_left = Agent(gameState.agent_left, c=AGENT_LEFT_COLOR)
        self.agent_right = Agent(gameState.agent_right, c=AGENT_RIGHT_COLOR)
        self.agent_left.updateState(self.ball.p, self.agent_right.p)
        self.agent_right.updateState(self.ball.p, self.agent_left.p)
        self.hidden_left = gameState.hidden_left
        self.hidden_right = gameState.hidden_right
        self.action_left_flag = gameState.action_left_flag
        self.action_left = gameState.action_left
        self.action_right_flag = gameState.action_right_flag
        self.action_right = gameState.action_right

    def setLeftAction(self, action):
        self.action_left_flag = jnp.int32(1)
        self.action_left = action

    def setRightAction(self, action):
        self.action_right_flag = jnp.int32(1)
        self.action_right = action

    def setAction(self):
        obs_left = self.agent_left.getObservation()
        obs_right = self.agent_right.getObservation()
        self.hidden_left, action_left = baselinePolicy(
            obs_left, self.hidden_left, self.baselineParams)
        self.hidden_right, action_right = baselinePolicy(
            obs_right, self.hidden_right, self.baselineParams)
        # overwrite internal AI actions if the flags are turned on:
        action_left = jnp.where(
            self.action_left_flag, self.action_left, action_left)
        action_right = jnp.where(
            self.action_right_flag, self.action_right, action_right)
        self.agent_left.setAction(action_left)
        self.agent_right.setAction(action_right)

    def getGameState(self):
        return GameState(self.ball.p, self.agent_left.p, self.agent_right.p,
                         self.hidden_left, self.hidden_right,
                         self.action_left_flag, self.action_left,
                         self.action_right_flag, self.action_right)

    def step(self):
        """ main game loop """

        self.agent_left.update()
        self.agent_right.update()

        self.ball.applyAcceleration(0, GRAVITY)
        self.ball.limitSpeed(MAX_BALL_SPEED)
        self.ball.move()

        self.ball.bounceIfColliding(self.agent_left.p)
        self.ball.bounceIfColliding(self.agent_right.p)
        self.ball.bounceIfColliding(self.fenceStub.p)

        # negated, since we want reward to be from the persepctive
        # of right agent being trained.
        result = -self.ball.checkEdges()

        self.agent_left.updateLife(result)
        self.agent_right.updateLife(result)

        self.agent_left.updateState(self.ball.p, self.agent_right.p)
        self.agent_right.updateState(self.ball.p, self.agent_left.p)

        return result

    def display(self):
        canvas = create_canvas(c=BACKGROUND_COLOR)
        canvas = self.fence.display(canvas)
        canvas = self.fenceStub.display(canvas)
        canvas = self.agent_left.display(canvas, self.ball.p.x, self.ball.p.y)
        canvas = self.agent_right.display(
            canvas, self.ball.p.x, self.ball.p.y)
        canvas = self.ball.display(canvas)
        canvas = self.ground.display(canvas)
        canvas = downsize_image(canvas)
        # canvas = upsize_image(canvas)  # removed to save memory for render.
        return canvas


def initGameState(ball_vx, ball_vy):
    ball = initParticleState(0, REF_W/4, ball_vx, ball_vy, 0.5)
    agent_left = initAgentState(-1, -REF_W/4, 1.5)
    agent_right = initAgentState(1, REF_W/4, 1.5)
    hidden_left = initBaselinePolicyState()
    hidden_right = initBaselinePolicyState()
    action_left_flag = jnp.int32(0)  # left is the built-in AI
    action_left = jnp.array([0, 0, 1], dtype=jnp.float32)
    action_right_flag = jnp.int32(1)  # right is the agent being trained.
    action_right = jnp.array([0, 0, 1], dtype=jnp.float32)
    return GameState(ball, agent_left, agent_right,
                     hidden_left, hidden_right,
                     action_left_flag, action_left,
                     action_right_flag, action_right)


def newMatch(prevGameState: GameState, ball_vx, ball_vy) -> GameState:
    ball = initParticleState(0, REF_W/4, ball_vx, ball_vy, 0.5)
    p = prevGameState
    return GameState(ball, p.agent_left, p.agent_right,
                     p.hidden_left, p.hidden_right,
                     p.action_left_flag, p.action_left,
                     p.action_right_flag, p.action_right)


def get_random_ball_v(key: jnp.ndarray):
    result = random.uniform(key, shape=(2,)) * 2 - 1
    ball_vx = result[1]*20
    ball_vy = result[2]*7.5+17.5
    return ball_vx, ball_vy


def get_init_game_state_fn(key: jnp.ndarray):
    ball_vx, ball_vy = get_random_ball_v(key)
    return initGameState(ball_vx, ball_vy)


def get_new_match_state_fn(game_state: GameState,
                           key: jnp.ndarray) -> GameState:
    ball_vx, ball_vy = get_random_ball_v(key)
    return newMatch(game_state, ball_vx, ball_vy)


def update_state_for_new_match(game_state: GameState,
                               reward, key: jnp.ndarray):
    old_ball = game_state.ball
    ball_vx, ball_vy = get_random_ball_v(key)
    new_ball = initParticleState(0, REF_W/4, ball_vx, ball_vy, 0.5)
    x = jnp.where(reward == 0, old_ball.x, new_ball.x)
    y = jnp.where(reward == 0, old_ball.y, new_ball.y)
    prev_x = jnp.where(reward == 0, old_ball.prev_x, new_ball.prev_x)
    prev_y = jnp.where(reward == 0, old_ball.prev_y, new_ball.prev_y)
    vx = jnp.where(reward == 0, old_ball.vx, new_ball.vx)
    vy = jnp.where(reward == 0, old_ball.vy, new_ball.vy)
    ball = ParticleState(x, y, prev_x, prev_y, vx, vy, old_ball.r)
    p = game_state
    return GameState(ball, p.agent_left, p.agent_right,
                     p.hidden_left, p.hidden_right,
                     p.action_left_flag, p.action_left,
                     p.action_right_flag, p.action_right)


def update_state(action: jnp.ndarray, game_state: GameState, key: jnp.array):
    game = Game(game_state)
    game.setRightAction(action)
    game.setAction()
    reward = game.step()  # from perspective of the agent on the right
    updated_game_state = game.getGameState()
    obs = game.agent_right.getObservation()

    updated_game_state = update_state_for_new_match(
        updated_game_state, reward, key)

    return updated_game_state, reward, obs


def detect_done(game_state: GameState):
    result = jnp.bitwise_or(
        game_state.agent_left.life <= 0, game_state.agent_right.life <= 0)
    return result


def get_obs(game_state: GameState):
    game = Game(game_state)
    return game.agent_right.getObservation()


class SlimeVolley(VectorizedTask):
    """Neural Slime Volleyball Environment."""

    def __init__(self,
                 max_steps: int = 3000,
                 test: bool = False):

        self.max_steps = max_steps
        self.obs_shape = tuple([12, ])
        self.act_shape = tuple([3, ])
        self.test = test

        def reset_fn(key):
            next_key, key = random.split(key)
            game_state = get_init_game_state_fn(key)
            return State(game_state=game_state, obs=get_obs(game_state),
                         steps=jnp.zeros((), dtype=int), key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            next_key, key = random.split(state.key)
            cur_state, reward, obs = update_state(
                action=action, game_state=state.game_state, key=key)
            steps = state.steps + 1
            done_test = jnp.bitwise_or(
                detect_done(cur_state), steps >= max_steps)
            # during training, go for all 3000 steps.
            done = jnp.where(self.test, done_test, steps >= max_steps)
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            return State(game_state=cur_state, obs=obs,
                         steps=steps, key=next_key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

    @staticmethod
    def render(state: State, task_id: int = 0) -> Image:
        """Render a specified task."""
        game = Game(state.game_state)
        canvas = game.display()
        img = Image.fromarray(canvas)
        return img
