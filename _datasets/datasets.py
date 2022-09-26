import numpy as np
import torch
from torch.utils.data import TensorDataset


class TadGANDataset(TensorDataset):
    def __init__(self, data, label=None, sampling_ratio=None, seq_len=None):
        self.data = data
        if label is not None:
            self.label = label
        else:
            self.label = np.zeros(shape=(len(self.data),), dtype=np.int8)

        if sampling_ratio is not None:
            # self.data, self.label = self._sampling_with_perm(self.data, self.label, sampling_ratio, seq_len)
            self.data, self.label = self._sampling_with_seq(self.data, self.label, sampling_ratio)

    def _sampling_with_seq(self, data, label, sampling_ratio):
        length = len(data)
        idx_sample = int(sampling_ratio * length)
        return data[:idx_sample], label[:idx_sample]

    def _sampling_with_perm(self, data, label, sampling_ratio, seq_len):

        data = self._convert_to_windows(data, seq_len)
        label = self._convert_to_windows(label, seq_len)

        length = len(data)
        idx_sample = np.random.permutation(length)[:int(np.floor(sampling_ratio * length))]
        idx_sample = np.sort(idx_sample)
        return data[idx_sample], label[idx_sample]

    def _convert_to_windows(self, data, seq_len, stride=1):
        new_data = []
        for i in range(0, len(data) - seq_len, stride):
            _x = data[i:i + seq_len]
            new_data.append(_x)

        return np.array(new_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.from_numpy(x)
        return {'signal': x, 'anomaly': self.label[idx]}
