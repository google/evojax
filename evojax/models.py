import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax

mask_final_layer_name = 'DENSE_FINAL'
cnn_final_layer_name = 'DENSE_FINAL'
current_dataset_number = 4


class CNN(nn.Module):
    """CNN for MNIST."""
    dataset_number: int = current_dataset_number
    dropout_rate: float = None

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 mask: jnp.ndarray = None,
                 task_labels: jnp.ndarray = None,
                 train: bool = True
                 ):

        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3), padding="SAME", name="CONV1")(x))
        x = nn.relu(nn.Conv(features=16, kernel_size=(3, 3), padding="SAME", name="CONV2")(x))

        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(features=512, name="DENSE1")(x))

        if task_labels is not None:
            label_input = nn.one_hot(task_labels, self.dataset_number)
            x = jnp.concatenate([x, label_input], axis=1)

        # TODO N.B. need to change the dropout code in flax to remove ifs for jit compatibility
        if self.dropout_rate is not None:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        if mask is not None:
            x = x * mask

        x = nn.Dense(features=10, name=cnn_final_layer_name)(x)
        return x


def create_train_state(rng, learning_rate=1e-3, use_task_labels: bool = None,
                       dropout_rate: float = None, weight_decay: float = None):
    """Creates initial `TrainState`."""
    p_rng, d_rng = random.split(rng)
    init_rngs = {'params': p_rng, 'dropout': d_rng}
    task_labels = jnp.ones([1, ]) if use_task_labels else None
    params = CNN(dropout_rate=dropout_rate).init(init_rngs,
                                                 jnp.ones([1, 28, 28, 1]),
                                                 None,  # No need to provide mask as will not change model trace
                                                 task_labels,
                                                 True,  # Set train to True, not sure if needed though
                                                 )['params']
    if weight_decay is None:
        tx = optax.adam(learning_rate)
    else:
        tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    return train_state.TrainState.create(
        apply_fn=CNN(dropout_rate=dropout_rate).apply, params=params, tx=tx)


class Mask(nn.Module):
    """Mask network for to provide a mask based on task label."""
    mask_size: int
    dataset_number: int = current_dataset_number
    image_mask: bool = None

    @nn.compact
    def __call__(self, x):
        x = nn.one_hot(x, self.dataset_number)

        # x = nn.Dense(features=10, name="LAYER1")(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=100, name="LAYER2")(x)
        # x = nn.relu(x)
        x = nn.Dense(features=self.mask_size, name=mask_final_layer_name)(x)
        x = nn.sigmoid(x)
        # x = nn.tanh(x)

        return x


class PixelMask(nn.Module):
    """Mask network for to provide a mask based on image input."""
    mask_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=8, kernel_size=(3, 3), padding="SAME", name="CONV1")(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME", name="CONV2")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.mask_size, name=mask_final_layer_name)(x)
        # x = nn.sigmoid(x)
        x = nn.tanh(x)

        return x

