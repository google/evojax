# Utilities for Benchmarking EvoJAX Algorithms 

This repository contains benchmark results, helper scripts, ES configurations and logs for testing the performance of evolutionary strategies in [`evojax`](https://github.com/google/evojax/). These can be useful, when aiming to merge a new JAX-based ES into the project.

## Installation

```
pip install evojax pyyaml
```

## Running the Benchmarks for an Evolution Strategy

1. Fork `evojax`. 
2. Add your strategy to `algo` and the `Strategies` wrapper in the `__init__.py` file.
3. Add the base task configurations for you ES to `configs/<es>/`.
4. Execute the individual training runs for the base/default configurations via:

```
python train.py -config configs/<es>/cartpole_easy.yaml
python train.py -config configs/<es>/cartpole_hard.yaml
python train.py -config configs/<es>/waterworld.yaml
python train.py -config configs/<es>/waterworld_ma.yaml
python train.py -config configs/<es>/brax_ant.yaml
python train.py -config configs/<es>/mnist.yaml
```

5. **[Optional]**: Tune hyperparameters using [`mle-hyperopt`](https://github.com/mle-infrastructure/mle-hyperopt).

```
pip install git+https://github.com/mle-infrastructure/mle-hyperopt.git@main
```

You can then specify hyperparameter ranges and the search strategy in a yaml file as follows:

```yaml
num_iters: 25
search_type: "Grid"
maximize_objective: true
verbose: true
search_params:
  real:
    es_config/optimizer_config/lrate_init:
      begin: 0.001
      end: 0.1
      bins: 5
    es_config/init_stdev:
      begin: 0.01
      end: 0.1
      bins: 5
```

Afterwards, you can easily execute the search using the `mle-search` CLI. Here is an example for running a grid search for ARS over different learning rates and perturbation standard deviations via:

```
mle-search train.py -base configs/ARS/mnist.yaml -search configs/ARS/search.yaml -iters 25 -log log/ARS/mnist/
```

This will sequentially execute 25 ARS-MNIST evolution runs for a grid of different learning rates and standard deviations. After the search has completed, you can access the search log at `log/ARS/mnist/search_log.yaml`. Finally, we provide some [utilities](viz_grid.ipynb) to visualize the search results.

## Benchmark Results

### CMA-ES

|   | Benchmarks | Parameters | Results (Avg) |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](configs/CMA_ES/cartpole_easy.yaml)| 927.3208 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](configs/CMA_ES/cartpole_hard.yaml)| 625.9829 |
MNIST	| 0.90 (max_iter=2000)	| [Link](configs/CMA_ES/mnist.yaml)| 0.9581 |
Brax Ant |	3000 (max_iter=1200) |[Link](configs/CMA_ES/brax_ant.yaml)| 3174.0608 |
Waterworld	| 6 (max_iter=1000)	 | [Link](configs/CMA_ES/waterworld.yaml)| 9.44 |
Waterworld (MA)	| 2 (max_iter=2000)	| [Link](configs/CMA_ES/waterworld_ma.yaml) | 0.5625 |


### Sep-CMA-ES

|   | Benchmarks | Parameters | Results (Avg) |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](configs/Sep_CMA_ES/cartpole_easy.yaml)| 924.3028 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](configs/Sep_CMA_ES/cartpole_hard.yaml)| 626.9728 |
MNIST	| 0.90 (max_iter=2000)	| [Link](configs/Sep_CMA_ES/mnist.yaml)| 0.9545 |
Brax Ant |	3000 (max_iter=1200) |[Link](configs/Sep_CMA_ES/brax_ant.yaml)| 3980.9194 |
Waterworld	| 6 (max_iter=1000)	 | [Link](configs/Sep_CMA_ES/waterworld.yaml)| 9.9000 |
Waterworld (MA)	| 2 (max_iter=2000)	| [Link](configs/Sep_CMA_ES/waterworld_ma.yaml) | 1.1875 |


### PGPE

|   | Benchmarks | Parameters | Results (Avg) |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](configs/PGPE/cartpole_easy.yaml)| 935.4268 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](configs/PGPE/cartpole_hard.yaml)| 631.1020 |
MNIST	| 0.90 (max_iter=2000)	| [Link](configs/PGPE/mnist.yaml)| 0.9743 |
Brax Ant |	3000 (max_iter=1200) |[Link](configs/PGPE/brax_ant.yaml)| 6054.3887 |
Waterworld	| 6 (max_iter=1000)	 | [Link](configs/PGPE/waterworld.yaml)| 11.6400 |
Waterworld (MA)	| 2 (max_iter=2000)	| [Link](configs/PGPE/waterworld_ma.yaml) | 2.0625 |

### CMA-ES-JAX

|   | Benchmarks | Parameters | Results (Avg) |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](configs/CMA_ES_JAX/cartpole_easy.yaml) | 917.8397 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](configs/CMA_JAX_ES/cartpole_hard.yaml)|  619.9166 |
MNIST	| 0.90 (max_iter=2000)	| [Link](configs/CMA_ES_JAX/mnist.yaml) | 0.9493 |
Waterworld	| 6 (max_iter=1000)	 | [Link](configs/CMA_ES_JAX/waterworld.yaml)  | 9.4500 |
Brax Ant |	3000 (max_iter=1200) | - | - |
Waterworld (MA)	| 2 (max_iter=2000)	| - | - |


### OpenES

|   | Benchmarks | Parameters | Results (Avg) |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](configs/OpenES/cartpole_easy.yaml)| 929.4153 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](configs/OpenES/cartpole_hard.yaml)| 604.6940 |
MNIST	| 0.90 (max_iter=2000)	| [Link](configs/OpenES/mnist.yaml)| 0.9669 |
Brax Ant |	3000 (max_iter=1200) |[Link](configs/OpenES/brax_ant.yaml)| 6726.2100 |
Waterworld	| 6 (max_iter=1000)	 | - | - |
Waterworld (MA)	| 2 (max_iter=2000)	| - | - |


*Note*: For the brax environment I reduced the population size from 1024 to 256 and increased the search iterations by the same factor (300 to 1200) in the main run. For the grid search I used a population size of 256 but with 500 iterations.

| Cartpole-Easy  | Cartpole-Hard | MNIST | Brax|
|---|---|---|---|
<img src="figures/OpenES/cartpole_easy.png?raw=true" alt="drawing" width="200" />|<img src="figures/OpenES/cartpole_hard.png?raw=true" alt="drawing" width="200" />| <img src="figures/OpenES/mnist.png?raw=true" alt="drawing" width="200" /> | <img src="figures/OpenES/brax.png?raw=true" alt="drawing" width="200" /> |
### Augmented Random Search


|   | Benchmarks | Parameters | Results |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](configs/ARS/cartpole_easy.yaml)| 902.107 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](configs/ARS/cartpole_hard.yaml)| 666.6442 |
Waterworld	| 6 (max_iter=1000)	 |[Link](configs/ARS/waterworld.yaml)| 6.1300 |
Waterworld (MA)	| 2 (max_iter=2000)	| [Link](configs/ARS/waterworld_ma.yaml)| 1.4831 |
Brax Ant |	3000 (max_iter=300) |[Link](configs/ARS/brax_ant.yaml)| 3298.9746 |
MNIST	| 0.90 (max_iter=2000)	| [Link](configs/ARS/mnist.yaml)| 0.9610 |


| Cartpole-Easy  | Cartpole-Hard | MNIST | 
|---|---|---|
<img src="figures/ARS/cartpole_easy.png?raw=true" alt="drawing" width="200" />|<img src="figures/ARS/cartpole_hard.png?raw=true" alt="drawing" width="200" />| <img src="figures/ARS/mnist.png?raw=true" alt="drawing" width="200" /> |
