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

"""JAX port of Fast Moving Natural Evolution Strategy 
    for High-Dimensional Problems (CR-FM-NES), see https://arxiv.org/abs/2201.11422 .
    Derived from https://github.com/nomuramasahir0/crfmnes"""   

import math
import numpy as np
from typing import Union
from typing import Optional
import logging
import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger

class CRFMNES(NEAlgorithm):
    """A wrapper of CR-FM-NES jax port."""

    def __init__(self,
                 param_size: int,
                 pop_size: int,
                 init_params: Optional[Union[jnp.ndarray, np.ndarray]] = None,                 
                 init_stdev: float = 0.1,
                 seed: int = 0,
                 logger: logging.Logger = None):
        if logger is None:
            self.logger = create_logger(name='FCRFM')
        else:
            self.logger = logger
        self.pop_size = pop_size

        if init_params is None:
            center = np.zeros(abs(param_size))
        else:
            center = init_params
            
        self.crfm = CRFM(param_size, pop_size, center, init_stdev, jax.random.PRNGKey(seed))    

        self.params = None
        self._best_params = None

        self.jnp_array = jax.jit(jnp.array)
        self.jnp_stack = jax.jit(jnp.stack)

    def ask(self) -> jnp.ndarray:
        self.params = self.crfm.ask()
        return self.jnp_stack(self.params)

    def tell(self, fitness: jnp.ndarray) -> None:
        self.crfm.tell(-np.array(fitness))
        self._best_params = self.crfm.x_best

    @property
    def best_params(self) -> jnp.ndarray:
        return self.jnp_array(self._best_params)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self._best_params = jnp.array(params)
        self.crfm.set_m(self._best_params.copy())

class CRFM():
    def __init__(self, num_dims: 
                 int, popsize: int, 
                 mean: Optional[Union[jnp.ndarray, np.ndarray]], 
                 input_sigma: float, 
                 rng: jax.random.PRNGKey):
        """Fast Moving Natural Evolution Strategy 
        for High-Dimensional Problems (CR-FM-NES), see https://arxiv.org/abs/2201.11422 .
        Derived from https://github.com/nomuramasahir0/crfmnes"""        
        if popsize % 2 == 1: # requires even popsize
            popsize += 1
        self.lamb = popsize
        self.dim = num_dims
        self.sigma = input_sigma 
        self.rng = rng       
        self.m = jnp.array([mean]).T
        self.v = jax.random.normal(rng, (self.dim, 1)) / jnp.sqrt(self.dim)       
        self.D = jnp.ones([self.dim, 1])

        self.w_rank_hat = (jnp.log(self.lamb / 2 + 1) - jnp.log(jnp.arange(1, self.lamb + 1))).reshape(self.lamb, 1)
        self.w_rank_hat = self.w_rank_hat.at[jnp.where(self.w_rank_hat < 0)].set(0)
        self.w_rank = self.w_rank_hat / sum(self.w_rank_hat) - (1. / self.lamb)        
        self.mueff = float(1 / jnp.dot((self.w_rank + (1 / self.lamb)).T, (self.w_rank + (1 / self.lamb)))[0][0])

        self.cs = (self.mueff + 2.) / (self.dim + self.mueff + 5.)
        self.cc = (4. + self.mueff / self.dim) / (self.dim + 4. + 2. * self.mueff / self.dim)
        self.c1_cma = 2. / (math.pow(self.dim + 1.3, 2) + self.mueff)
        # initialization
        self.chiN = math.sqrt(self.dim) * (1. - 1. / (4. * self.dim) + 1. / (21. * self.dim * self.dim))
        self.pc = jnp.zeros((self.dim, 1))
        self.ps = jnp.zeros((self.dim, 1))
        # distance weight parameter
        self.h_inv = get_h_inv(self.dim)
        self.alpha_dist = lambda lambF: self.h_inv * min(1., math.sqrt(self.lamb / self.dim)) * math.sqrt(
            lambF / self.lamb)
        self.w_dist_hat = lambda z, lambF: exp(self.alpha_dist(lambF) * jnp.linalg.norm(z))
        # learning rate
        self.eta_m = 1.0
        self.eta_move_sigma = 1.
        self.eta_stag_sigma = lambda lambF: math.tanh((0.024 * lambF + 0.7 * self.dim + 20.) / (self.dim + 12.))
        self.eta_conv_sigma = lambda lambF: 2. * math.tanh((0.025 * lambF + 0.75 * self.dim + 10.) / (self.dim + 4.))
        self.c1 = lambda lambF: self.c1_cma * (self.dim - 5) / 6 * (lambF / self.lamb)
        self.eta_B = lambda lambF: jnp.tanh((min(0.02 * lambF, 3 * jnp.log(self.dim)) + 5) / (0.23 * self.dim + 25))

        self.g = 0
        self.no_of_evals = 0
        self.iteration = 0
        self.stop = 0

        self.idxp = jnp.arange(self.lamb / 2, dtype=int)
        self.idxm = jnp.arange(self.lamb / 2, self.lamb, dtype=int)
        self.z = jnp.zeros([self.dim, self.lamb])

        self.f_best = float('inf')
        self.x_best = jnp.empty(self.dim)

    def set_m(self, params: jnp.ndarray):
        self.m = jnp.array(params).reshape((self.dim, 1))

    def ask(self) -> jnp.ndarray:
        key, self.rng = jax.random.split(self.rng)
        zhalf = jax.random.normal(key, (self.dim, int(self.lamb / 2)))
        self.z = self.z.at[:, self.idxp].set(zhalf)
        self.z = self.z.at[:, self.idxm].set(-zhalf)
        self.normv = jnp.linalg.norm(self.v)
        self.normv2 = self.normv ** 2
        self.vbar = self.v / self.normv
        self.y = self.z + ((jnp.sqrt(1 + self.normv2) - 1) * jnp.dot(self.vbar, jnp.dot(self.vbar.T, self.z)))
        self.x = self.m + (self.sigma * self.y) * self.D
        return self.x.T

    def tell(self, evals_no_sort: np.ndarray) -> None:
        sorted_indices = sort_indices_by(evals_no_sort, self.z)
        best_eval_id = sorted_indices[0]       
        f_best = evals_no_sort[best_eval_id]
        x_best = self.x[:, best_eval_id]
        self.z = self.z[:, sorted_indices]
        y = self.y[:, sorted_indices]
        x = self.x[:, sorted_indices]
        self.no_of_evals += self.lamb
        self.g += 1
        if f_best < self.f_best:
            self.f_best = f_best
            self.x_best = x_best   
                    
        # This operation assumes that if the solution is infeasible, infinity comes in as input.
        lambF = jnp.sum(evals_no_sort < jnp.finfo(float).max)
        # evolution path p_sigma
        self.ps = (1 - self.cs) * self.ps + jnp.sqrt(self.cs * (2. - self.cs) * self.mueff) * jnp.dot(self.z, self.w_rank)
        ps_norm = jnp.linalg.norm(self.ps)
        # distance weight
        f1 =  self.h_inv * min(1., math.sqrt(self.lamb / self.dim)) * math.sqrt(lambF / self.lamb)        
        w_tmp = self.w_rank_hat * jnp.exp(jnp.linalg.norm(self.z, axis = 0) * f1).reshape((self.lamb,1))
        weights_dist = w_tmp / sum(w_tmp) - 1. / self.lamb
        # switching weights and learning rate
        weights = weights_dist if ps_norm >= self.chiN else self.w_rank
        eta_sigma = self.eta_move_sigma if ps_norm >= self.chiN else self.eta_stag_sigma(
            lambF) if ps_norm >= 0.1 * self.chiN else self.eta_conv_sigma(lambF)
        # update pc, m
        wxm = jnp.dot(x - self.m, weights)
        self.pc = (1. - self.cc) * self.pc + jnp.sqrt(self.cc * (2. - self.cc) * self.mueff) * wxm / self.sigma
        self.m += self.eta_m * wxm
        normv4 = self.normv2 ** 2
        exY = jnp.append(y, self.pc / self.D, axis=1)  # dim x lamb+1
        yy = exY * exY  # dim x lamb+1
        ip_yvbar = jnp.dot(self.vbar.T, exY)
        yvbar = exY * self.vbar  # dim x lamb+1. exYのそれぞれの列にvbarがかかる
        gammav = 1. + self.normv2
        vbarbar = self.vbar * self.vbar
        alphavd = min(
            [1, math.sqrt(normv4 + (2 * gammav - math.sqrt(gammav)) / jnp.max(vbarbar)) / (2 + self.normv2)])  # scalar       
        t = exY * ip_yvbar - self.vbar * (ip_yvbar ** 2 + gammav) / 2  # dim x lamb+1
        b = -(1 - alphavd ** 2) * normv4 / gammav + 2 * alphavd ** 2
        H = jnp.ones([self.dim, 1]) * 2 - (b + 2 * alphavd ** 2) * vbarbar  # dim x 1
        invH = H ** (-1)
        s_step1 = yy - self.normv2 / gammav * (yvbar * ip_yvbar) - jnp.ones([self.dim, self.lamb + 1])  # dim x lamb+1
        ip_vbart = jnp.dot(self.vbar.T, t)  # 1 x lamb+1
        s_step2 = s_step1 - alphavd / gammav * ((2 + self.normv2) * (t * self.vbar) - self.normv2 * jnp.dot(vbarbar, ip_vbart))  # dim x lamb+1
        invHvbarbar = invH * vbarbar
        ip_s_step2invHvbarbar = jnp.dot(invHvbarbar.T, s_step2)  # 1 x lamb+1       
        div = 1 + b * jnp.dot(vbarbar.T, invHvbarbar)
        if jnp.amin(abs(div)) == 0:
            self.logger.info('error: div is zero')
            return      
        s = (s_step2 * invH) - b / div * jnp.dot(invHvbarbar, ip_s_step2invHvbarbar)  # dim x lamb+1
        ip_svbarbar = jnp.dot(vbarbar.T, s)  # 1 x lamb+1
        t = t - alphavd * ((2 + self.normv2) * (s * self.vbar) - jnp.dot(self.vbar, ip_svbarbar))  # dim x lamb+1
        # update v, D
        exw = jnp.append(self.eta_B(lambF) * weights, jnp.full((1, 1), self.c1(lambF)), axis=0)  # lamb+1 x 1
        self.v = self.v + jnp.dot(t, exw) / self.normv
        self.D = self.D + jnp.dot(s, exw) * self.D
        # calculate detA
        if jnp.amin(self.D) < 0:
            self.logger.info('error: invalid D')
            return
        nthrootdetA = exp(jnp.sum(jnp.log(self.D)) / self.dim + jnp.log(1 + jnp.dot(self.v.T, self.v)[0][0]) / (2 * self.dim))
        self.D = self.D / nthrootdetA
        # update sigma
        G_s = jnp.sum(  jnp.dot( (self.z * self.z - jnp.ones([self.dim, self.lamb])), weights  )) / self.dim
        self.sigma = self.sigma * exp(eta_sigma / 2 * G_s)    

def get_h_inv(dim: int) -> float:
    f = lambda a, b: ((1. + a * a) * exp(a * a / 2.) / 0.24) - 10. - dim
    f_prime = lambda a: (1. / 0.24) * a * exp(a * a / 2.) * (3. + a * a)
    h_inv = 1.0
    while abs(f(h_inv, dim)) > 1e-10:
        h_inv = h_inv - 0.5 * (f(h_inv, dim) / f_prime(h_inv))
    return h_inv

def exp(a: float) -> float:
    return math.exp(min(100, a)) # avoid overflow

def sort_indices_by(evals: np.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    lam = len(evals)
    sorted_indices = np.argsort(evals)
    sorted_evals = evals[sorted_indices]
    no_of_feasible_solutions = np.where(sorted_evals != jnp.inf)[0].size
    if no_of_feasible_solutions != lam:
        infeasible_z = z[:, np.where(evals == jnp.inf)[0]]
        distances = np.sum(infeasible_z ** 2, axis=0)
        infeasible_indices = sorted_indices[no_of_feasible_solutions:]
        indices_sorted_by_distance = np.argsort(distances)
        sorted_indices = sorted_indices.at[no_of_feasible_solutions:].set(infeasible_indices[indices_sorted_by_distance])
    return sorted_indices
