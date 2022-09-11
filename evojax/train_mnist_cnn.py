import logging
import numpy as np
from typing import Tuple

import wandb

import optax
import jax
from jax import random
import jax.numpy as jnp
from flax.training import train_state
from flaxmodels.flaxmodels.resnet import ResNet18

from evojax.models import CNN, Mask, linear_layer_name
from evojax.datasets import read_data_files, digit, fashion, kuzushiji, cifar

dataset_names = [digit, fashion, kuzushiji, cifar]


chosen_model = CNN()
# chosen_model = ResNet18(num_classes=10,
#                         pretrained='')


def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).sum()


def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


def create_train_state(rng, learning_rate):
    """Creates initial `TrainState`."""
    params = chosen_model.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=chosen_model.apply, params=params, tx=tx)


def get_masks(mask_params, mask_size, batch, pixel_input):
    if pixel_input:
        input_array = batch['image']
    else:
        input_array = batch['label'][:, 1]

    batch_masks = Mask(mask_size=mask_size, pixel_input=pixel_input).apply({'params': mask_params}, input_array)
    return batch_masks


@jax.jit
def train_step(state, batch, mask_params=None, pixel_input=False):
    """Train for a single step."""

    if mask_params is not None:
        linear_weights = state.params[linear_layer_name]["kernel"]
        mask_size = linear_weights.shape[0]
        batch_masks = get_masks(mask_params, mask_size, batch, pixel_input)
    else:
        batch_masks = None

    class_labels = batch['label'][:, 0]

    def loss_fn(params):
        output_logits = chosen_model.apply({'params': params}, batch['image'], batch_masks)
        loss = cross_entropy_loss(logits=output_logits, labels=class_labels)
        return loss, output_logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=class_labels)
    return state, metrics


@jax.jit
def eval_step(params, batch, mask_params=None, pixel_input=False):

    if mask_params is not None:
        linear_weights = params[linear_layer_name]["kernel"]
        mask_size = linear_weights.shape[0]
        batch_masks = get_masks(mask_params, mask_size, batch, pixel_input)
    else:
        batch_masks = None

    logits = chosen_model.apply({'params': params}, batch['image'], batch_masks)
    return compute_metrics(logits=logits, labels=batch['label'][:, 0])


def train_epoch(state, train_ds, batch_size, epoch, rng, logger: logging.Logger = None,
                mask_params=None, pixel_input=False):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []

    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch, mask_params, pixel_input)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}

    return state, epoch_metrics_np


def eval_model(params, test_dataset_class, batch_size, mask_params=None, pixel_input=False):

    for dataset_name, test_ds in test_dataset_class.dataset_holder.items():
        test_ds_size = len(test_ds['image'])
        steps_per_epoch = test_ds_size // batch_size

        batch_metrics = []
        for i in range(steps_per_epoch):
            batch = {k: v[i*batch_size: (i+1)*batch_size, ...] for k, v in test_ds.items()}
            metrics = eval_step(params, batch, mask_params, pixel_input)
            batch_metrics.append(metrics)

        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]}

        test_dataset_class.metrics_holder[dataset_name] = {'loss': epoch_metrics_np['loss'],
                                                           'accuracy': epoch_metrics_np['accuracy']}

    return test_dataset_class


class TestDatasetUtil:
    """
    Class to facilitate having separate test sets for the multiple datasets.
    """
    def __init__(self, dataset_names_to_use: list, dataset_dicts: list = None):
        self.dataset_holder = {}
        self.metrics_holder = {}

        if dataset_dicts:
            self.dataset_holder = dict(zip(dataset_names_to_use, dataset_dicts))
        else:
            for dataset_name in dataset_names_to_use:
                self.setup_dataset(dataset_name)

    def setup_dataset(self, dataset_name):
        test_dataset = {}
        x_array_test, y_array_test = [], []

        x_test, y_test = read_data_files(dataset_name, 'test')
        x_array_test.append(x_test)
        y_array_test.append(y_test)

        test_dataset['image'] = jnp.float32(np.concatenate(x_array_test)) / 255.
        test_dataset['label'] = jnp.int16(np.concatenate(y_array_test))

        self.dataset_holder[dataset_name] = test_dataset


def full_data_loader() -> Tuple[dict, TestDatasetUtil, TestDatasetUtil]:
    x_array_train, y_array_train = [], []
    for dataset_name in dataset_names:
        x_train, y_train = read_data_files(dataset_name, 'train')
        x_array_train.append(x_train)
        y_array_train.append(y_train)

    full_train_images = jnp.float32(np.concatenate(x_array_train)) / 255.
    full_train_labels = jnp.int16(np.concatenate(y_array_train))

    number_of_points = full_train_images.shape[0]
    number_for_validation = number_of_points // 5

    ix = random.permutation(key=random.PRNGKey(0), x=number_of_points)
    validation_ix = ix[:number_for_validation]
    train_ix = ix[number_for_validation:]

    train_dataset = {'image': jnp.take(full_train_images, indices=train_ix, axis=0),
                     'label': jnp.take(full_train_labels, indices=train_ix, axis=0)}

    validation_dataset = {'image': jnp.take(full_train_images, indices=validation_ix, axis=0),
                          'label': jnp.take(full_train_labels, indices=validation_ix, axis=0)}

    validation_dataset_class = TestDatasetUtil(['combined'], [validation_dataset])

    train_dataset['image'] = jnp.float32(np.concatenate(x_array_train)) / 255.
    # Only use the class label, not dataset label here
    train_dataset['label'] = jnp.int16(np.concatenate(y_array_train))

    # Sets up a separate test set for each of the datasets
    test_dataset_class = TestDatasetUtil(dataset_names)

    return train_dataset, validation_dataset_class, test_dataset_class


def run_mnist_training(
        logger: logging.Logger,
        num_epochs=20,
        learning_rate=1e-3,
        cnn_batch_size=1024,
        return_model=True,
        state=None,
        mask_params=None,
        pixel_input=False,
        datasets_tuple=None,
        evo_epoch=0
):

    logger.info('Starting training MNIST CNN')

    rng = random.PRNGKey(0)

    # Allow passing of a state, so only init if this is none
    if state is None:
        rng, init_rng = random.split(rng)
        state = create_train_state(init_rng, learning_rate)
        del init_rng  # Must not be used anymore.

    if datasets_tuple:
        train_dataset, validation_dataset_class, test_dataset_class = datasets_tuple
    else:
        train_dataset, validation_dataset_class, test_dataset_class = full_data_loader()

    best_state = None
    best_test_accuracy = 0
    previous_validation_accuracy = 0
    for epoch in range(1, num_epochs + 1):

        # Since there can be multiple evo epochs count from the start of them
        relative_epoch = evo_epoch * num_epochs + epoch - 1

        logger.info(f'Starting epoch {relative_epoch} of CNN training')
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)

        # Run an optimization step over a training batch
        state, train_metrics = train_epoch(state, train_dataset, cnn_batch_size, epoch, input_rng, logger=logger,
                                           mask_params=mask_params, pixel_input=pixel_input)

        logger.debug(f'TRAIN, epoch={epoch}, loss={train_metrics["loss"]}, accuracy={train_metrics["accuracy"]}')

        # Check the validation dataset
        validation_dataset_class = eval_model(state.params, validation_dataset_class, cnn_batch_size,
                                              mask_params=mask_params, pixel_input=pixel_input)
        current_validation_accuracy = np.mean([i['accuracy'] for i in validation_dataset_class.metrics_holder.values()])
        logger.debug(f'VALIDATION, epoch={epoch}, accuracy={current_validation_accuracy}')

        if current_validation_accuracy > previous_validation_accuracy:
            previous_validation_accuracy = current_validation_accuracy
            best_state = state
        else:
            logger.info(f'Validation accuracy decreased on epoch {epoch}, stopping early')
            break

        # Evaluate on the test set after each training epoch
        test_dataset_class = eval_model(state.params, test_dataset_class, cnn_batch_size)
        test_loss = np.mean([i['loss'] for i in test_dataset_class.metrics_holder.values()])
        test_accuracy = np.mean([i['accuracy'] for i in test_dataset_class.metrics_holder.values()])
        best_test_accuracy = max(best_test_accuracy, test_accuracy)

        logger.debug(f'TEST, epoch={epoch}, loss={test_loss}, accuracy={test_accuracy}')

        for dataset_name in dataset_names:
            ds_test_accuracy = test_dataset_class.metrics_holder[dataset_name].get("accuracy")
            logger.debug(f'TEST, {dataset_name} 'f'accuracy={ds_test_accuracy:.2f}')

            # wandb.log({f'{dataset_name} Test Accuracy': ds_test_accuracy}, step=relative_epoch, commit=False)
            wandb.log({f'{dataset_name} Test Accuracy': ds_test_accuracy})

        wandb.log({'Combined Train Accuracy': train_metrics['accuracy'],
                   'Combined Validation Accuracy': current_validation_accuracy,
                   'Combined Test Accuracy': test_accuracy})

    logger.info(f'Best test accuracy for unmasked CNN is {best_test_accuracy:.4f}')

    if return_model:
        return best_state, best_test_accuracy


if __name__ == '__main__':

    log = logging.Logger(level=logging.INFO, name='mnist_logger')
    run_mnist_training(logger=log, return_model=False)
