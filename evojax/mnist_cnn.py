import logging
import numpy as np
from typing import Tuple

import wandb

import jax
from jax import random
import jax.numpy as jnp
from flax.training import train_state
from flax.core import FrozenDict

from evojax.models import CNN, Mask, cnn_final_layer_name, create_train_state
from evojax.datasets import dataset_names, DatasetUtilClass, full_data_loader, combined_dataset_key
from evojax.util import cross_entropy_loss, compute_metrics


def get_batch_masks(state, task_labels, mask_params=None, l1_pruning_proportion=None):
    if mask_params is not None:
        linear_weights = state.params[cnn_final_layer_name]["kernel"]
        mask_size = linear_weights.shape[0]
        batch_masks = Mask(mask_size=mask_size).apply(mask_params, task_labels)
    elif l1_pruning_proportion is not None:
        linear_weights = state.params[cnn_final_layer_name]["kernel"]
        avg_weight = jnp.sum(linear_weights, axis=1)
        mask_size = linear_weights.shape[0]
        sorted_weights = avg_weight.sort()
        batch_masks = jnp.where(avg_weight > sorted_weights[jnp.array(mask_size*l1_pruning_proportion, int)], 1, 0)
    else:
        batch_masks = None

    return batch_masks


@jax.jit
def train_step(state: train_state.TrainState,
               batch: dict,
               rng: jnp.ndarray,
               mask_params: FrozenDict = None,
               task_labels: jnp.ndarray = None,
               l1_pruning_proportion: float = None,
               l1_reg_lambda: float = None,
               dropout_rate: float = None,
               ):

    class_labels = batch['label'][:, 0]
    batch_masks = get_batch_masks(state, task_labels, mask_params, l1_pruning_proportion)

    def loss_fn(params):
        output_logits = CNN(dropout_rate=dropout_rate).apply({'params': params},
                                                             batch['image'],
                                                             batch_masks,
                                                             task_labels,
                                                             train=True,
                                                             rngs={'dropout': rng})

        loss = cross_entropy_loss(logits=output_logits, labels=class_labels)
        if l1_reg_lambda is not None:
            loss += l1_reg_lambda * jnp.sum(jnp.abs(params[cnn_final_layer_name]["kernel"]))

        return loss, output_logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=class_labels)
    return state, metrics


@jax.jit
def eval_step(state: train_state.TrainState,
              batch: dict,
              rng: jnp.ndarray,
              mask_params: FrozenDict = None,
              task_labels: jnp.ndarray = None,
              l1_pruning_proportion: float = None,
              l1_reg_lambda: float = None,
              dropout_rate: float = None,
              ) -> Tuple[train_state.TrainState, dict]:

    params = state.params
    class_labels = batch['label'][:, 0]
    batch_masks = get_batch_masks(state, task_labels, mask_params, l1_pruning_proportion)

    logits = CNN(dropout_rate=dropout_rate).apply({'params': params},
                                                  batch['image'],
                                                  batch_masks,
                                                  task_labels,
                                                  train=False,
                                                  rngs={'dropout': rng})

    return state, compute_metrics(logits=logits, labels=class_labels)


def epoch_step(test: bool,
               state: train_state.TrainState,
               dataset_class: DatasetUtilClass,
               batch_size: int,
               rng,
               mask_params: FrozenDict = None,
               use_task_labels: bool = False,
               l1_pruning_proportion: float = None,
               l1_reg_lambda: float = None,
               dropout_rate: float = None,
               ) -> Tuple[train_state.TrainState, DatasetUtilClass]:

    for dataset_name, dataset in dataset_class.dataset_holder.items():
        ds_size = dataset['image'].shape[0]
        steps_per_epoch = ds_size // batch_size

        if test:
            step_func = eval_step
        else:
            step_func = train_step

        perms = jax.random.permutation(rng, ds_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))

        batch_metrics = []
        for perm in perms:
            batch = {k: v[perm, ...] for k, v in dataset.items()}
            task_labels = batch['label'][:, 1] if use_task_labels else None
            state, metrics = step_func(state,
                                       batch,
                                       rng,
                                       mask_params,
                                       task_labels=task_labels,
                                       l1_pruning_proportion=l1_pruning_proportion,
                                       l1_reg_lambda=l1_reg_lambda,
                                       dropout_rate=dropout_rate)

            batch_metrics.append(metrics)

        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]}

        # Save the metrics for that datas
        dataset_class.metrics_holder[dataset_name] = {'loss': epoch_metrics_np['loss'],
                                                      'accuracy': epoch_metrics_np['accuracy']}

    return state, dataset_class


def calc_and_log_metrics(dataset_class: DatasetUtilClass, logger: logging.Logger, epoch: int,
                         wandb_logging: bool) -> float:

    if dataset_class.split == 'test':
        total_accuracy = np.mean([i['accuracy'] for i in dataset_class.metrics_holder.values()])
        total_loss = np.mean([i['loss'] for i in dataset_class.metrics_holder.values()])

        for dataset_name in dataset_names:
            ds_test_accuracy = dataset_class.metrics_holder[dataset_name].get("accuracy")
            logger.debug(f'TEST, {dataset_name} 'f'accuracy={ds_test_accuracy:.2f}')

            if wandb_logging:
                wandb.log({f'{dataset_name} Test Accuracy': ds_test_accuracy})
    else:
        total_accuracy = dataset_class.metrics_holder[combined_dataset_key]['accuracy']
        total_loss = dataset_class.metrics_holder[combined_dataset_key]['loss']

    logger.debug(f'{dataset_class.split.upper()}, epoch={epoch}, loss={total_loss}, accuracy={total_accuracy}')

    return total_accuracy


def run_mnist_training(
        logger: logging.Logger = None,
        wandb_logging: bool = True,
        eval_only: bool = False,
        seed: int = 0,
        num_epochs: int = 20,
        evo_epoch: int = 0,
        learning_rate: float = 1e-3,
        cnn_batch_size: int = 1024,
        state: train_state.TrainState = None,
        mask_params: FrozenDict = None,
        datasets_tuple: Tuple[DatasetUtilClass, DatasetUtilClass, DatasetUtilClass] = None,
        early_stopping: bool = False,
        # These are the parameters for the other sparsity baseline types
        use_task_labels: bool = False,
        l1_pruning_proportion: float = None,
        l1_reg_lambda: float = None,
        dropout_rate: float = None,
        weight_decay: float = None,
) -> Tuple[train_state.TrainState, dict]:

    rng = random.PRNGKey(seed)

    # Allow passing of a state, so only init if this is none
    if state is None:
        rng, init_rng = random.split(rng)
        task_labels = jnp.ones([1, ]) if use_task_labels else None
        state = create_train_state(init_rng, learning_rate, task_labels,
                                   dropout_rate=dropout_rate, weight_decay=weight_decay)
        del init_rng  # Must not be used anymore.

    if datasets_tuple:
        train_dataset_class, validation_dataset_class, test_dataset_class = datasets_tuple
    else:
        train_dataset_class, validation_dataset_class, test_dataset_class = full_data_loader()

    if eval_only:
        state, test_dataset_class = epoch_step(test=True,
                                               state=state,
                                               dataset_class=test_dataset_class,
                                               batch_size=cnn_batch_size,
                                               rng=rng,
                                               mask_params=mask_params,
                                               use_task_labels=use_task_labels,
                                               l1_pruning_proportion=l1_pruning_proportion,
                                               l1_reg_lambda=l1_reg_lambda,
                                               dropout_rate=dropout_rate)

        eval_accuracy = float(np.mean([i['accuracy'] for i in test_dataset_class.metrics_holder.values()]))
        logger.info(f'Final Test Accuracy: {eval_accuracy}')
        return state, {'train': [], 'validation': [], 'test': [eval_accuracy]}

    logger.info('Starting training MNIST CNN')

    previous_state = None
    accuracy_dict = {'train': [], 'validation': [], 'test': []}
    current_test_accuracy = previous_validation_accuracy = 0.
    for epoch in range(1, num_epochs + 1):
        # Since there can be multiple evo epochs count from the start of them
        relative_epoch = evo_epoch * num_epochs + epoch - 1

        logger.info(f'Starting epoch {relative_epoch} of CNN training')

        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)

        # Run an optimization step over a training batch
        state, train_dataset_class = epoch_step(test=False,
                                                state=state,
                                                dataset_class=train_dataset_class,
                                                batch_size=cnn_batch_size,
                                                rng=input_rng,
                                                mask_params=mask_params,
                                                use_task_labels=use_task_labels,
                                                l1_pruning_proportion=l1_pruning_proportion,
                                                l1_reg_lambda=l1_reg_lambda,
                                                dropout_rate=dropout_rate)

        # Check the validation dataset
        state, validation_dataset_class = epoch_step(test=True,
                                                     state=state,
                                                     dataset_class=validation_dataset_class,
                                                     batch_size=cnn_batch_size,
                                                     rng=input_rng,
                                                     mask_params=mask_params,
                                                     use_task_labels=use_task_labels,
                                                     l1_pruning_proportion=l1_pruning_proportion,
                                                     l1_reg_lambda=l1_reg_lambda,
                                                     dropout_rate=dropout_rate)

        current_validation_accuracy = validation_dataset_class.metrics_holder[combined_dataset_key]['accuracy']

        # If the validation accuracy decreases will want to end if doing early stopping
        if current_validation_accuracy > previous_validation_accuracy and early_stopping:
            previous_validation_accuracy = current_validation_accuracy
            previous_state = state
        elif early_stopping:
            logger.info(f'Validation accuracy decreased on epoch {epoch}, stopping early')
            logger.info(f'Final Test Accuracy: {current_test_accuracy}')
            return previous_state, accuracy_dict
        else:
            pass

        # Evaluate on the test set after each training epoch
        state, test_dataset_class = epoch_step(test=True,
                                               state=state,
                                               dataset_class=test_dataset_class,
                                               batch_size=cnn_batch_size,
                                               rng=input_rng,
                                               mask_params=mask_params,
                                               use_task_labels=use_task_labels,
                                               l1_pruning_proportion=l1_pruning_proportion,
                                               l1_reg_lambda=l1_reg_lambda,
                                               dropout_rate=dropout_rate)

        current_train_accuracy = calc_and_log_metrics(train_dataset_class, logger, epoch, wandb_logging)
        _ = calc_and_log_metrics(validation_dataset_class, logger, epoch,wandb_logging)
        current_test_accuracy = calc_and_log_metrics(test_dataset_class, logger, epoch, wandb_logging)

        accuracy_dict['train'].append(current_train_accuracy)
        accuracy_dict['validation'].append(current_validation_accuracy)
        accuracy_dict['test'].append(current_test_accuracy)

        if wandb_logging:
            wandb.log({'Combined Train Accuracy': current_train_accuracy,
                       'Combined Validation Accuracy': current_validation_accuracy,
                       'Combined Test Accuracy': current_test_accuracy})

    logger.info(f'Final Test Accuracy: {current_test_accuracy}')

    return state, accuracy_dict
