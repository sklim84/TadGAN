import numpy as np
import torch
from torch.utils.data import Dataset


class WADIDataset(Dataset):
    def __init__(self, path_data, path_label=None, sampling_ratio=None):
        self.data = np.load(path_data)
        if path_label is not None:
            self.label = np.load(path_label)
        else:
            self.label = np.zeros(shape=(len(self.data),), dtype=np.int8)

        if sampling_ratio is not None:
            self.data = self._sampling(self.data, sampling_ratio)
            self.label = self._sampling(self.label, sampling_ratio)

    def _sampling(self, target, sampling_ratio):
        length = len(target)
        idx_sample = np.random.permutation(length)[:int(np.floor(sampling_ratio * length))]
        return target[idx_sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.from_numpy(x)
        return {'signal': x, 'anomaly': self.label[idx]}
