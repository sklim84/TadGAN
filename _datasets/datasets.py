import numpy as np
import torch
from torch.utils.data import Dataset


class WADIDataset(Dataset):
    def __init__(self, data, label=None, sampling_ratio=None):
        self.data = data
        if label is not None:
            self.label = label
        else:
            self.label = np.zeros(shape=(len(self.data),), dtype=np.int8)

        if sampling_ratio is not None:
            self.data, self.label = self._sampling(self.data, self.label, sampling_ratio)

    def _sampling(self, data, label, sampling_ratio):
        length = len(data)
        idx_sample = np.random.permutation(length)[:int(np.floor(sampling_ratio * length))]
        idx_sample = np.sort(idx_sample)
        return data[idx_sample], label[idx_sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.from_numpy(x)
        return {'signal': x, 'anomaly': self.label[idx]}
