from typing import Optional
import jax.numpy as jnp
from flax import linen as nn


mask_final_layer_name = 'DENSE'
linear_layer_name = 'Dense_0'


class CNN(nn.Module):
    """CNN for MNIST."""

    def setup(self):
        # self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME', name='conv1')
        # # self.bn1 = nn.BatchNorm(32, momentum=0.9, use_running_average=True, name='bn1')
        #
        # self.conv2 = nn.Conv(features=48, kernel_size=(3, 3), padding='SAME', name='conv2')
        # # self.bn2 = nn.BatchNorm(48, momentum=0.9, use_running_average=True, name='bn2')
        #
        # self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', name='conv3')
        # # self.bn3 = nn.BatchNorm(64, momentum=0.9, use_running_average=not self.train, name='bn3')
        #
        # self.conv4 = nn.Conv(features=80, kernel_size=(3, 3), padding='SAME', name='conv4')
        # # self.bn4 = nn.BatchNorm(80, momentum=0.9, use_running_average=not self.train, name='bn4')
        #
        # self.conv5 = nn.Conv(features=96, kernel_size=(3, 3), padding='SAME', name='conv5')
        # # self.bn5 = nn.BatchNorm(96, momentum=0.9, use_running_average=not self.train, name='bn5')
        #
        # self.conv6 = nn.Conv(features=112, kernel_size=(3, 3), padding='SAME', name='conv6')
        # # self.bn6 = nn.BatchNorm(112, momentum=0.9, use_running_average=not self.train, name='bn6')
        #
        # self.conv7 = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME', name='conv7')
        # # self.bn7 = nn.BatchNorm(128, momentum=0.9, use_running_average=not self.train, name='bn7')
        #
        # self.conv8 = nn.Conv(features=144, kernel_size=(3, 3), padding='SAME', name='conv8')
        # # self.bn8 = nn.BatchNorm(144, momentum=0.9, use_running_average=not self.train, name='bn8')
        #
        # self.conv9 = nn.Conv(features=160, kernel_size=(3, 3), padding='SAME', name='conv9')
        # # self.bn9 = nn.BatchNorm(160, momentum=0.9, use_running_average=not self.train, name='bn9')
        #
        # self.conv10 = nn.Conv(features=176, kernel_size=(3, 3), padding='SAME', name='conv10')
        # # self.bn10 = nn.BatchNorm(176, momentum=0.9, use_running_average=not self.train, name='bn10')
        #
        # self.linear = nn.Dense(10, name='linear')

        # self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME", name="CONV1")
        # self.conv2 = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME", name="CONV2")

        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME", name="CONV1")
        self.conv2 = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME", name="CONV2")

        self.linear1 = nn.Dense(features=10, name=linear_layer_name)

    @nn.compact
    def __call__(self, x,
                 cnn_labels: Optional[jnp.ndarray] = None,
                 mask: Optional[jnp.ndarray] = None):

        # x = nn.relu(self.bn1(self.conv1(x)))
        # x = nn.relu(self.bn2(self.conv2(x)))
        # x = nn.relu(self.bn3(self.conv3(x)))
        # x = nn.relu(self.bn4(self.conv4(x)))
        # x = nn.relu(self.bn5(self.conv5(x)))
        # x = nn.relu(self.bn6(self.conv6(x)))
        # x = nn.relu(self.bn7(self.conv7(x)))
        # x = nn.relu(self.bn8(self.conv8(x)))
        # x = nn.relu(self.bn9(self.conv9(x)))
        # x = nn.relu(self.bn10(self.conv10(x)))
        #
        # x = nn.relu(self.conv1(x))
        # x = nn.relu(self.conv2(x))
        # x = nn.relu(self.conv3(x))
        # x = nn.relu(self.conv4(x))
        # x = nn.relu(self.conv5(x))
        # x = nn.relu(self.conv6(x))
        # x = nn.relu(self.conv7(x))
        # x = nn.relu(self.conv8(x))
        # x = nn.relu(self.conv9(x))
        # x = nn.relu(self.conv10(x))

        # for i in range(1, 11):
        #     x = nn.relu(getattr(self, f'bn{i}')(getattr(self, f'conv{i}')(x)))

        # Use the example MNIST CNN
        # x = nn.Conv(features=8, kernel_size=(5, 5), padding='SAME')(x)
        # x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # x = nn.Conv(features=16, kernel_size=(5, 5), padding='SAME')(x)
        # x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        #
        # x = x.reshape((x.shape[0], -1))  # flatten
        #
        # x = self.linear(x)
        # x = nn.log_softmax(x)

        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))

        x = x.reshape((x.shape[0], -1))

        # TODO is this a fine way to implement the masking???
        if mask is not None:
            x = x * mask

        if cnn_labels is not None:
            label_input = nn.one_hot(cnn_labels, self.dataset_number)
            label_input = nn.Dense(features=10, name='LABEL1')(label_input)
            label_input = nn.Dense(features=100, name='LABEL2')(label_input)
            x = jnp.concatenate([x, label_input], axis=1)

        x = self.linear1(x)
        return x


class Mask(nn.Module):
    """Mask network for to provide a mask based on either dataset label or image input."""
    mask_size: int
    dataset_number: int = 4
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
