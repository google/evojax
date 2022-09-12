from typing import Optional
import jax.numpy as jnp
from flax import linen as nn


mask_final_layer_name = 'DENSE'
linear_layer_name = 'Dense_0'
current_dataset_number = 4


class CNN(nn.Module):
    """CNN for MNIST."""
    mask: Optional[jnp.ndarray] = None
    dropout: bool = False
    dataset_number: int = current_dataset_number

    @nn.compact
    def __call__(self, x,
                 cnn_labels: Optional[jnp.ndarray] = None):

        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3), padding="SAME", name="CONV1")(x))
        x = nn.relu(nn.Conv(features=16, kernel_size=(3, 3), padding="SAME", name="CONV2")(x))

        x = x.reshape((x.shape[0], -1))

        if self.dropout:
            x = nn.Dropout(0.05)(x)

        # TODO is this a fine way to implement the masking???
        if self.mask is not None:
            x = x * self.mask

        if cnn_labels is not None:
            label_input = nn.one_hot(cnn_labels, self.dataset_number)
            x = jnp.concatenate([x, label_input], axis=1)

        x = nn.Dense(features=10, name=linear_layer_name)(x)
        return x


class Mask(nn.Module):
    """Mask network for to provide a mask based on either dataset label or image input."""
    mask_size: int
    dataset_number: int = current_dataset_number
    round_output: bool = True
    test_no_mask: bool = False
    pixel_input: bool = False

    @nn.compact
    def __call__(self, x):
        if self.pixel_input:
            x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME", name="CONV1")(x)
            x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME", name="CONV2")(x)
            x = x.reshape((x.shape[0], -1))
        else:
            x = nn.one_hot(x, self.dataset_number)

        # x = nn.Dense(features=10, name="DENSE1")(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=100, name="DENSE2")(x)
        # x = nn.relu(x)
        x = nn.Dense(features=self.mask_size, name=mask_final_layer_name)(x)
        x = nn.sigmoid(x)
        if self.round_output:
            x = jnp.round(x)

        if self.test_no_mask:
            return jnp.ones((x.shape[0], self.mask_size))
        else:
            return x
