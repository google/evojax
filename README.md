# EvoJAX: Hardware-Accelerated Neuroevolution

This repository contains the implementation of EvoJAX, a toolkit for accelerated Neuroevolution experiments.

## Installation

EvoJAX is implemented in [JAX](https://github.com/google/jax) which needs to be installed first.

**Install JAX**: 
Please first follow JAX's [installation instruction](https://github.com/google/jax#installation) with optional GPU/TPU backend support.
In case JAX is not set up, EvoJAX installation will still try pulling a CPU-only version of JAX.
Note that Colab runtimes come with JAX pre-installed.


**Install EvoJAX**:
```shell
# Install from PyPI.
pip install evojax

# Or, install from our GitHub repo.
pip install git+https://github.com/google/evojax.git@main
```

## Code Overview

EvoJAX is a framework with three major components, which we expect the users to extend.
1. **Neuroevolution Algorithms** All neuroevolution algorithms should implement the `evojax.algo.base.NEAlgorithm` interface and reside in `evojax/algo/`.
We currently provide [PGPE](https://people.idsia.ch/~juergen/nn2010.pdf), with more coming soon.
2. **Policy Networks** All neural networks should implement the `evojax.policy.base.PolicyNetwork` interface and be saved in `evojax/policy/`.
In this repo, we give example implementations of the MLP, ConvNet, Seq2Seq and [PermutationInvariant](https://attentionneuron.github.io/) models.
3. **Tasks** All tasks should implement `evojax.task.base.VectorizedTask` and be in `evojax/task/`.

These components can be used either independently, or orchestrated by `evojax.trainer` and `evojax.sim_mgr` that manage the training pipeline.
While they should be sufficient for the currently provided policies and tasks, we plan to extend their functionality in the future as the need arises.

## Examples

As a quickstart, we provide non-trivial examples (scripts in `examples/` and notebooks in `examples/notebooks`) to illustrate the usage of EvoJAX.
We provide example commands to start the training process at the top of each script.
These scripts and notebooks are run with TPUs and/or NVIDIA V100 GPU(s):

*Supervised Learning Tasks*
* [MNIST Classification](https://github.com/google/evojax/blob/main/examples/train_mnist.py) -
We show that EvoJAX trains a ConvNet policy to achieve >98% test accuracy within 5 min on a single GPU.
* [Seq2Seq Learning](https://github.com/google/evojax/blob/main/examples/train_seq2seq.py) -
We demonstrate that EvoJAX is capable of learning a large network with hundreds of thousands parameters to accomplish a seq2seq task.

*Classic Control Tasks*
* [Locomotion](https://github.com/google/evojax/blob/main/examples/notebooks/BraxTasks.ipynb) -
[Brax](https://github.com/google/brax) is a differentiable physics engine implemented in JAX.
We wrap it as a task and train with EvoJAX on GPUs/TPUs. It takes EvoJAX tens of minutes to solve a locomotion task in Brax.
* [Cart-Pole Swing Up](https://github.com/google/evojax/blob/main/examples/train_cartpole.py) -
We illustrate how the classic control task can be implemented in JAX and be integrated into EvoJAX's pipeline for significant speed up training.

*Novel Tasks*
* [WaterWorld](https://github.com/google/evojax/blob/main/examples/train_waterworld.py) -
In this [task](https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html), an agent tries to get as much food as possible while avoiding poisons.
EvoJAX is able to learn the agent in tens of minutes on a single GPU.
Moreover, we demonstrate that [multi-agents training](https://github.com/google/evojax/blob/main/examples/train_waterworld_ma.py) in EvoJAX is possible, which is beneficial for learning policies that can deal with environmental complexity and uncertainties.
* Abstract Paintings ([notebook 1](https://github.com/google/evojax/blob/main/examples/notebooks/AbstractPainting01.ipynb) and [notebook 2](https://github.com/google/evojax/blob/main/examples/notebooks/AbstractPainting02.ipynb)) -
We reproduce the results from this [computational creativity work](https://es-clip.github.io/) and show how the original work, whose implementation requires multiple CPUs and GPUs, could be accelerated on a single GPU efficiently using EvoJAX, which was not possible before.
Moreover, with multiple GPUs/TPUs, EvoJAX can further speed up the mentioned work almost linearly.
We also show that the modular design of EvoJAX allows its components to be used independently -- in this case it is possible to use only the ES algorithms from EvoJAX while leveraging one's own training loops and environment implantation.

## Disclaimer
This is not an official Google product.

