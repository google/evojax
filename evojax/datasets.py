import os
# import torch
import numpy as np
# from PIL import Image
# from sklearn.utils import shuffle
# from torchvision import transforms

dirname = os.path.dirname(__file__)

digit = 'MNIST'
fashion = 'FMNIST'
kuzushiji = 'KMNIST'
cifar = 'CIFAR'

# TODO are just different int labels fine?
DATASET_LABELS = {digit: 0,
                  fashion: 1,
                  kuzushiji: 2,
                  cifar: 3}

FOLDER_NAMES = {digit: 'MNIST',
                fashion: 'FashionMNIST',
                kuzushiji: 'KMNIST',
                cifar: 'CIFAR'}


def read_data_files(dataset_name, split):
    assert dataset_name in set(FOLDER_NAMES.keys())
    assert split in {'train', 'test'}

    folder_name = FOLDER_NAMES[dataset_name]
    folder_path = os.path.join(dirname, 'data', folder_name, 'raw')

    if dataset_name == cifar:
        file_name = os.path.join(folder_path, f'{split}.npz')
        loaded = np.load(file_name)
        xs, ys = loaded['X'], loaded['y']
    else:
        file_prefix = 't10k' if split == 'test' else split

        image_file = f'{file_prefix}-images-idx3-ubyte'
        label_file = f'{file_prefix}-labels-idx1-ubyte'

        with open(os.path.join(folder_path, image_file), 'rb') as f:
            xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))

        with open(os.path.join(folder_path, label_file), 'rb') as f:
            ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))

        xs = np.reshape(xs, (-1, 28, 28, 1)).astype(np.float32)
        ys = ys.astype(np.int)

    # Stack the dataset label with the ys
    ys_with_label = np.full_like(ys, DATASET_LABELS[dataset_name])

    return xs, np.stack([ys, ys_with_label], axis=1)
