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

# dataset_names = [digit, fashion, kuzushiji, cifar]

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
    ys = ys.astype(np.int16)

    # Stack the dataset label with the ys
    ys_with_label = np.full_like(ys, DATASET_LABELS[dataset_name])

    return xs, np.stack([ys, ys_with_label], axis=1)


class DatasetUtilClass:
    """
    Class to facilitate having separate test sets for the multiple datasets.
    """
    def __init__(self, split: str, dataset_names: list, dataset_dicts: list = None):
        self.split = split
        self.dataset_names = dataset_names
        self.dataset_holder = {}
        self.metrics_holder = {}

        if dataset_dicts:
            self.dataset_holder = dict(zip(dataset_names, dataset_dicts))
        else:
            for dataset_name in dataset_names:
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

    def return_data_arrays(self):
        if self.split == 'test':
            images = jnp.concatenate([ds['image'] for ds in self.dataset_holder.values()], axis=0)
            labels = jnp.concatenate([ds['label'] for ds in self.dataset_holder.values()], axis=0)
        else:
            dataset = self.dataset_holder[combined_dataset_key]
            images, labels = dataset['image'], dataset['label']
        return images, labels[:, 0], labels[:, 1]

    def get_data_count(self):
        data_count = sum(ds['image'].shape[0] for ds in self.dataset_holder.values())
        return data_count


def get_train_val_split(dataset_names, validation: bool, val_fraction) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x_array_train, y_array_train = [], []
    for dataset_name in dataset_names:
        x_train, y_train = read_data_files(dataset_name, 'train')
        x_array_train.append(x_train)
        y_array_train.append(y_train)

    full_train_images = jnp.float32(np.concatenate(x_array_train)) / 255.
    full_train_labels = jnp.int16(np.concatenate(y_array_train))

    number_of_points = full_train_images.shape[0]
    number_for_validation = int(number_of_points * val_fraction)

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


def full_data_loader(dataset_names, val_fraction=0.2) -> Tuple[DatasetUtilClass, DatasetUtilClass, DatasetUtilClass]:
    """
    Load up DatasetUtilClass for train/val/test splits of the datasets.
    """
    train_images, train_labels = get_train_val_split(dataset_names=dataset_names,
                                                     validation=False, val_fraction=val_fraction)
    val_images, val_labels = get_train_val_split(dataset_names=dataset_names,
                                                 validation=True, val_fraction=val_fraction)

    train_dataset = {'image': train_images,
                     'label': train_labels}

    validation_dataset = {'image': val_images,
                          'label': val_labels}

    train_dataset_class = DatasetUtilClass(split='train', dataset_names=[combined_dataset_key],
                                           dataset_dicts=[train_dataset])
    validation_dataset_class = DatasetUtilClass(split='validation', dataset_names=[combined_dataset_key],
                                                dataset_dicts=[validation_dataset])

    # Sets up a separate test set for each of the datasets
    test_dataset_class = DatasetUtilClass('test', dataset_names)

    return train_dataset_class, validation_dataset_class, test_dataset_class
