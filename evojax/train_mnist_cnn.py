import numpy as np

import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from flax.training import train_state

from evojax.datasets import read_data_files, digit, fashion, kuzushiji

import optax


class CNN(nn.Module):
    """CNN for MNIST."""

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME', name='conv1')
        self.bn1 = nn.BatchNorm(32, momentum=0.1, use_running_average=True, name='bn1')

        self.conv2 = nn.Conv(features=48, kernel_size=(3, 3), padding='SAME', name='conv2')
        self.bn2 = nn.BatchNorm(48, momentum=0.1, use_running_average=True, name='bn2')

        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', name='conv3')
        self.bn3 = nn.BatchNorm(64, momentum=0.1, use_running_average=True, name='bn3')

        self.conv4 = nn.Conv(features=80, kernel_size=(3, 3), padding='SAME', name='conv4')
        self.bn4 = nn.BatchNorm(80, momentum=0.1, use_running_average=True, name='bn4')

        self.conv5 = nn.Conv(features=96, kernel_size=(3, 3), padding='SAME', name='conv5')
        self.bn5 = nn.BatchNorm(96, momentum=0.1, use_running_average=True, name='bn5')

        self.conv6 = nn.Conv(features=112, kernel_size=(3, 3), padding='SAME', name='conv6')
        self.bn6 = nn.BatchNorm(112, momentum=0.1, use_running_average=True, name='bn6')

        self.conv7 = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME', name='conv7')
        self.bn7 = nn.BatchNorm(128, momentum=0.1, use_running_average=True, name='bn7')

        self.conv8 = nn.Conv(features=144, kernel_size=(3, 3), padding='SAME', name='conv8')
        self.bn8 = nn.BatchNorm(144, momentum=0.1, use_running_average=True, name='bn8')

        self.conv9 = nn.Conv(features=160, kernel_size=(3, 3), padding='SAME', name='conv9')
        self.bn9 = nn.BatchNorm(160, momentum=0.1, use_running_average=True, name='bn9')

        self.conv10 = nn.Conv(features=176, kernel_size=(3, 3), padding='SAME', name='conv10')
        self.bn10 = nn.BatchNorm(176, momentum=0.1, use_running_average=True, name='bn10')

        self.linear = nn.Dense(10, name='linear')

    @nn.compact
    def __call__(self, x):

        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.relu(self.bn3(self.conv3(x)))
        x = nn.relu(self.bn4(self.conv4(x)))
        x = nn.relu(self.bn5(self.conv5(x)))
        x = nn.relu(self.bn6(self.conv6(x)))
        x = nn.relu(self.bn7(self.conv7(x)))
        x = nn.relu(self.bn8(self.conv8(x)))
        x = nn.relu(self.bn9(self.conv9(x)))
        x = nn.relu(self.bn10(self.conv10(x)))

        # for i in range(1, 11):
        #     x = nn.relu(getattr(self, f'bn{i}')(getattr(self, f'conv{i}')(x)))

        x = x.reshape((x.shape[0], -1))  # flatten

        x = self.linear(x)
        x = nn.log_softmax(x)
        return x


def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
      'loss': loss,
      'accuracy': accuracy,
    }
    return metrics


def create_train_state(rng, learning_rate, momentum):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        output_logits = CNN().apply({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits=output_logits, labels=batch['label'])
        return loss, output_logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics


@jax.jit
def eval_step(params, batch):
    logits = CNN().apply({'params': params}, batch['image'])
    return compute_metrics(logits=logits, labels=batch['label'])


def train_epoch(state, train_ds, batch_size, epoch, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

    print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
      epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

    return state


def eval_model(params, test_ds):
    metrics = eval_step(params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']


if __name__ == "main":
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    learning_rate = 0.1
    momentum = 0.9

    state = create_train_state(init_rng, learning_rate, momentum)
    del init_rng  # Must not be used anymore.

    num_epochs = 10
    batch_size = 32

    # train_ds = MnistDataset(training=True, transform=None, dataset_names=[digit, fashion, kuzushiji])
    # test_ds = MnistDataset(training=False, transform=None, dataset_names=[digit, fashion, kuzushiji])

    train_ds = {}
    test_ds = {}
    x_array_train, y_array_train = [], []
    x_array_test, y_array_test = [], []

    for dataset_name in [digit, fashion, kuzushiji]:
        x_train, y_train = read_data_files(dataset_name, 'train')
        x_array_train.append(x_train)
        y_array_train.append(y_train)

        x_test, y_test = read_data_files(dataset_name, 'test')
        x_array_test.append(x_test)
        y_array_test.append(y_test)

    train_ds['image'] = jnp.float32(np.concatenate(x_array_train)) / 255.
    test_ds['image'] = jnp.float32(np.concatenate(x_array_test)) / 255.

    train_ds['label'] = jnp.int16(np.concatenate(y_array_train))
    test_ds['label'] = jnp.int16(np.concatenate(y_array_test))

    for epoch in range(1, num_epochs + 1):
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
        # Evaluate on the test set after each training epoch
        test_loss, test_accuracy = eval_model(state.params, test_ds)
        print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, test_loss, test_accuracy * 100))

