from typing import Optional
import jax.numpy as jnp
from flax import linen as nn


mask_final_layer_name = 'DENSE'
cnn_final_layer_name = 'DENSE'
current_dataset_number = 4


class CNN(nn.Module):
    """CNN for MNIST."""
    dataset_number: int = current_dataset_number

    @nn.compact
    def __call__(self, x,
                 mask: Optional[jnp.ndarray] = None):

        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3), padding="SAME", name="CONV1")(x))
        x = nn.relu(nn.Conv(features=16, kernel_size=(3, 3), padding="SAME", name="CONV2")(x))
        x = x.reshape((x.shape[0], -1))

        if mask is not None:
            x = x * mask

        x = nn.Dense(features=10, name=cnn_final_layer_name)(x)
        return x


class Mask(nn.Module):
    """Mask network for to provide a mask based on either dataset label or image input."""
    mask_size: int
    dataset_number: int = current_dataset_number
    pixel_input: bool = False

    @nn.compact
    def __call__(self, x):
        if self.pixel_input:
            x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME", name="CONV1")(x)
            x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME", name="CONV2")(x)
            x = x.reshape((x.shape[0], -1))
        else:
            x = nn.one_hot(x, self.dataset_number)

        x = nn.Dense(features=self.mask_size, name=mask_final_layer_name)(x)
        x = nn.sigmoid(x)

        return x
