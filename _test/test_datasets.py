from _datasets.datasets import WADIDataset
import numpy as np
import pandas as pd
import openpyxl
from torch.utils.data import DataLoader

def test_WADIDataset_sampling():
    train_dataset = WADIDataset(data='../_datasets/WADI/train.npy', sampling_ratio=0.2)
    test_dataset = WADIDataset(data='../_datasets/WADI/test.npy', label='../_datasets/WADI/labels.npy',
                               sampling_ratio=0.2)
    assert len(train_dataset) != 0 and len(test_dataset) != 0


def test_dataset_npy():
    test_data = np.load('../_datasets/WADI/test.npy')
    labels = np.load('../_datasets/WADI/labels.npy')
    print(test_data.shape)
    print(test_data.shape[1])
    print(labels.shape)
    print(labels)


def test_WADIDataset_sampling2():
    train_data = np.load('../_datasets/WADI/train.npy')
    train_dataset = WADIDataset(data=train_data, sampling_ratio=0.2, seq_len=12)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8, drop_last=True)

    for batch, sample in enumerate(train_loader):
        signal = sample['signal']
        signal = signal.squeeze(dim=0)
        anomaly = sample['anomaly']
        print(signal.shape)
        print(anomaly)


def test_xlsx_to_csv():
    # df_attack = pd.read_excel('../_datasets/SWaT/SWaT_Dataset_Attack_v0.xlsx', sheet_name='Combined Data',
    #                           header=1)
    # print(df_attack)
    # df_attack.to_csv('../_datasets/SWaT/SWaT_Dataset_Attack_v0.csv')

    df_normal = pd.read_excel('../_datasets/SWaT/SWaT_Dataset_Normal_v0.xlsx', sheet_name='Normal.csv',
                              header=1)
    print(df_normal)
    df_normal.to_csv('../_datasets/SWaT/SWaT_Dataset_Normal_v0.csv')

def convert_to_windows(data, seq_len, stride=1):
    new_data = []
    for i in range(0, len(data) - seq_len, stride):
        _x = data[i:i + seq_len]
        new_data.append(_x)

    return np.array(new_data)

def test_dataset_sampling():
    train_data = np.random.rand(10000, 123)
    print(train_data.shape)
    train_data = convert_to_windows(train_data, 12, stride=1)
    print(len(train_data))
    print(train_data[0].shape)

    sampling_ratio = 0.2
    length = len(train_data)
    idx_sample = np.random.permutation(length)[:int(np.floor(sampling_ratio * length))]
    idx_sample = np.sort(idx_sample)
    train_data = train_data[idx_sample]
    print(len(train_data))
    print(train_data[0].shape)
