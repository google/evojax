import os
from typing import Tuple
import numpy as np
from jax import random
import jax.numpy as jnp

dirname = os.path.dirname(__file__)

digit = 'MNIST'
fashion = 'FMNIST'
kuzushiji = 'KMNIST'
cifar = 'CIFAR'

dataset_names = [digit, fashion, kuzushiji, cifar]

DATASET_LABELS = {digit: 0,
                  fashion: 1,
                  kuzushiji: 2,
                  cifar: 3}

FOLDER_NAMES = {digit: 'MNIST',
                fashion: 'FashionMNIST',
                kuzushiji: 'KMNIST',
                cifar: 'CIFAR'}

combined_dataset_key = 'combined'


def read_data_files(dataset_name, split):
    assert dataset_name in set(FOLDER_NAMES.keys())
    assert split in {'train', 'test'}

    folder_name = FOLDER_NAMES[dataset_name]
    folder_path = os.path.join(dirname, 'data', folder_name, 'raw')

    file_prefix = 't10k' if split == 'test' else split

    image_file = f'{file_prefix}-images-idx3-ubyte'
    label_file = f'{file_prefix}-labels-idx1-ubyte'

    with open(os.path.join(folder_path, image_file), 'rb') as f:
        # TODO unclear why the offset for CIFAR is different, just fix this way for now
        xs = np.array(np.frombuffer(f.read(), np.uint8, offset=8 if dataset_name == cifar else 16))

    with open(os.path.join(folder_path, label_file), 'rb') as f:
        ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))

    xs = np.reshape(xs, (-1, 28, 28, 1)).astype(np.float32)
    ys = ys.astype(np.int)

    # Stack the dataset label with the ys
    ys_with_label = np.full_like(ys, DATASET_LABELS[dataset_name])

    return xs, np.stack([ys, ys_with_label], axis=1)


class DatasetUtilClass:
    """
    Class to facilitate having separate test sets for the multiple datasets.
    """
    def __init__(self, split: str, dataset_names_to_use: list, dataset_dicts: list = None):
        self.split = split
        self.dataset_holder = {}
        self.metrics_holder = {}

        if dataset_dicts:
            self.dataset_holder = dict(zip(dataset_names_to_use, dataset_dicts))
        else:
            for dataset_name in dataset_names_to_use:
                self.setup_dataset(dataset_name)

    def setup_dataset(self, dataset_name):
        """
        Method to set up the separated test datasets - as is useful to get test metrics for each dataset type.
        """
        test_dataset = {}
        x_array_test, y_array_test = [], []

        x_test, y_test = read_data_files(dataset_name, 'test')
        x_array_test.append(x_test)
        y_array_test.append(y_test)

        test_dataset['image'] = jnp.float32(np.concatenate(x_array_test)) / 255.
        test_dataset['label'] = jnp.int16(np.concatenate(y_array_test))

        self.dataset_holder[dataset_name] = test_dataset


def get_train_val_split(validation: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    # Just select the section of the data corresponding to train/val indices
    if validation:
        image_data = jnp.take(full_train_images, indices=validation_ix, axis=0)
        labels = jnp.take(full_train_labels, indices=validation_ix, axis=0)
    else:
        image_data = jnp.take(full_train_images, indices=train_ix, axis=0)
        labels = jnp.take(full_train_labels, indices=train_ix, axis=0)

    return image_data, labels


def full_data_loader() -> Tuple[DatasetUtilClass, DatasetUtilClass, DatasetUtilClass]:
    """
    Load up DatasetUtilClass for train/val/test splits of the datasets.
    """
    train_images, train_labels = get_train_val_split(validation=False)
    val_images, val_labels = get_train_val_split(validation=True)

    train_dataset = {'image': train_images,
                     'label': train_labels}

    validation_dataset = {'image': val_images,
                          'label': val_labels}

    train_dataset_class = DatasetUtilClass('train', [combined_dataset_key], [train_dataset])
    validation_dataset_class = DatasetUtilClass('validation', [combined_dataset_key], [validation_dataset])

    # Sets up a separate test set for each of the datasets
    test_dataset_class = DatasetUtilClass('test', dataset_names)

    return train_dataset_class, validation_dataset_class, test_dataset_class
