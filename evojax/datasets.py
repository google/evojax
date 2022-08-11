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

# TODO are just different int labels fine?
DATASET_LABELS = {digit: 0,
                  fashion: 1,
                  kuzushiji: 2}

FOLDER_NAMES = {digit: 'MNIST',
                fashion: 'FashionMNIST',
                kuzushiji: 'KMNIST'}


def read_data_files(dataset_name, split):
    assert dataset_name in set(FOLDER_NAMES.keys())
    assert split in {'train', 'test'}

    folder_name = FOLDER_NAMES[dataset_name]

    folder_path = os.path.join(dirname, 'mnist_tests', 'data', folder_name, 'raw')
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


# class MnistDataset(torch.utils.data.Dataset):
#     def __init__(self, training=True, transform=None, dataset_names=()):
#         """
#         Create the class for MNIST datasets.
#
#         :param training: bool, whether this is the training or test set
#         :param transform: torchvision transforms, what transformations to apply to the images
#         :param dataset_names: tuple of the dataset names to include
#         """
#         split = 'train' if training else 'test'
#
#         x_dataset_list = []
#         y_dataset_list = []
#
#         for name in dataset_names:
#             x, y = read_data_files(name, split)
#             x_dataset_list.append(x)
#             y_dataset_list.append(y)
#         if not x_dataset_list or not y_dataset_list:
#             raise ValueError('No datasets have been selected.')
#
#         # Join datasets together
#         xs = np.concatenate(x_dataset_list, axis=0)
#         ys = np.concatenate(y_dataset_list, axis=0)
#
#         # Shuffle the image and label arrays, keep the same seed for now
#         xs, ys = shuffle(xs, ys, random_state=0)
#
#         self.x_data = xs
#         self.y_data = ys
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.x_data)
#
#     def __getitem__(self, idx):
#         x = Image.fromarray(self.x_data[idx].reshape(28, 28))
#         y = torch.tensor(np.array(self.y_data[idx]))
#         if self.transform:
#             x = self.transform(x)
#         x = transforms.ToTensor()(np.array(x)/255)
#         return x, y

