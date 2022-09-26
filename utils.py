import os
import re

import numpy as np
from torch.utils.data import DataLoader

from _datasets.datasets import TadGANDataset


def convert_to_windows(data, seq_len, stride=1):
    new_data = []
    for i in range(0, len(data) - seq_len, stride):
        _x = data[i:i + seq_len]
        new_data.append(_x)

    return np.array(new_data)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def create_dataloaders(datasets, batch_size, sampling_ratio=None):
    dataset_path = os.path.abspath(f'./_datasets/{datasets}')

    file_names = os.listdir(dataset_path)
    file_names = [file_name for file_name in file_names if 'npy' in file_name]
    file_names = sorted_nicely(file_names)

    train_loaders, test_data, label_data = [], [], []
    for file_name in file_names:
        data = np.load(os.path.join(dataset_path, file_name))

        if 'train' in file_name:
            signal_shape = data.shape[1]
            train_dataset = TadGANDataset(data=data, sampling_ratio=sampling_ratio)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, drop_last=True)
            train_loaders.append(train_loader)

        elif 'test' in file_name:
            test_data.append(data)

        elif 'labels' in file_name:
            label_data.append(data)

    test_data = np.vstack(test_data)
    label_data = np.hstack(label_data)

    test_dataset = TadGANDataset(data=test_data, label=label_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, drop_last=True)

    print(f'In {datasets}, number of tran dataloaders: {len(train_loaders)}')
    print(f'In {datasets}, length of test data: {len(test_data)}')
    print(f'In {datasets}, length of test labels: {len(label_data)}')

    return train_loaders, test_loader, signal_shape
