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

"""Train an ant locomotion controller with MAP-Elites.

To define a different BD extractor, see task/brax_task.py for example.

Example command:
python train_ant_map_elites.py --max-iter=3000
python train_ant_map_elites.py --max-iter=3000 --save-gif  # May cost some time.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functools import partial

from evojax import Trainer
from evojax.task.brax_task import BraxTask
from evojax.task.brax_task import AntBDExtractor
from evojax.policy import MLPPolicy
from evojax.algo import MAPElites
from evojax import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=1024, help='NE population size.')
    parser.add_argument(
        '--num-tests', type=int, default=128, help='Number of test rollouts.')
    parser.add_argument(
        '--n-repeats', type=int, default=8, help='Training repetitions.')
    parser.add_argument(
        '--max-iter', type=int, default=300, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=50, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=10, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--iso-sigma', type=float, default=0.05, help='Iso sigma.')
    parser.add_argument(
        '--line-sigma', type=float, default=0.3, help='Line sigma.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    parser.add_argument(
        '--save-gif', action='store_true', help='Save some GIFs.')
    config, _ = parser.parse_known_args()
    return config


def plot_figure(lattice, log_dir, title):
    grid = lattice.reshape((10, 10, 10, 10))
    fig, axes = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            ax = axes[i][j]
            ax.imshow(grid[i, j])
            ax.set_axis_off()
    fig.suptitle(title, fontsize=20, fontweight='bold')
    plt.savefig(os.path.join(log_dir, '{}.png'.format(title)))


def main(config):
    log_dir = './log/ant_map_elites'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='AntMapElites', log_dir=log_dir, debug=config.debug)

    logger.info('EvoJAX AntMapElites Demo')
    logger.info('=' * 30)

    bd_extractor = AntBDExtractor(logger=logger)
    train_task = BraxTask(
        env_name='ant', max_steps=500, bd_extractor=bd_extractor, test=False)
    test_task = BraxTask(
        env_name='ant', bd_extractor=bd_extractor, test=True)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[32, 32, 32, 32],
        output_dim=train_task.act_shape[0],
    )
    solver = MAPElites(
        pop_size=config.pop_size,
        param_size=policy.num_params,
        bd_extractor=bd_extractor,
        iso_sigma=config.iso_sigma,
        line_sigma=config.line_sigma,
        seed=config.seed,
        logger=logger,
    )

    # Train.
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        n_evaluations=config.num_tests,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Visualize the results.
    qd_file = os.path.join(log_dir, 'qd_lattices.npz')
    with open(qd_file, 'rb') as f:
        data = np.load(f)
        params_lattice = data['params_lattice']
        fitness_lattice = data['fitness_lattice']
        occupancy_lattice = data['occupancy_lattice']
    plot_figure(occupancy_lattice, log_dir, 'occupancy')
    plot_figure(fitness_lattice, log_dir, 'score')

    # Visualize the top policies.
    if config.save_gif:
        import jax
        import jax.numpy as jnp
        from brax import envs
        from brax.io import image

        @partial(jax.jit, static_argnums=(1,))
        def get_qp(state, ix):
            return jax.tree_map(lambda x: x[ix], state.qp)

        num_viz = 3
        idx = fitness_lattice.argsort()[-num_viz:]
        bins = [np.unravel_index(ix, (10, 10, 10, 10)) for ix in idx]
        logger.info(
            'Best {} policies: indices={}, bins={}'.format(num_viz, idx, bins))

        policy_params = jnp.array(params_lattice[idx])
        task_reset_fn = jax.jit(test_task.reset)
        policy_reset_fn = jax.jit(policy.reset)
        step_fn = jax.jit(test_task.step)
        act_fn = jax.jit(policy.get_actions)

        total_reward = jnp.zeros(num_viz)
        valid_masks = jnp.ones(num_viz)
        rollouts = {i: [] for i in range(num_viz)}
        keys = jnp.repeat(
            jax.random.PRNGKey(seed=42)[None, :], repeats=num_viz, axis=0)
        task_state = task_reset_fn(key=keys)
        policy_state = policy_reset_fn(task_state)

        for step in range(test_task.max_steps):
            for i in range(num_viz):
                rollouts[i].append(get_qp(task_state.state, i))
            act, policy_state = act_fn(task_state, policy_params, policy_state)
            task_state, reward, done = step_fn(task_state, act)
            total_reward = total_reward + reward * valid_masks
            valid_masks = valid_masks * (1 - done)
        logger.info('test_rewards={}'.format(total_reward))

        logger.info('Start saving GIFs, this can take some time ...')
        env_fn = envs.create_fn(env_name='ant', legacy_spring=True)
        env = env_fn()
        for i in range(num_viz):
            qps = jax.tree_map(lambda x: np.array(x), rollouts[i])
            frames = [
                Image.fromarray(
                    image.render_array(env.sys, qp, 320, 240, None, None, 2))
                for qp in qps]
            frames[0].save(
                os.path.join(log_dir, 'bin_{}.gif'.format(bins[i])),
                format='png',
                append_images=frames[1:],
                save_all=True,
                duration=env.sys.config.dt * 1000,
                loop=0)


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)
