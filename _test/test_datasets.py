from _datasets.datasets import WADIDataset
import numpy as np
import pandas as pd
import openpyxl


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
    train_dataset = WADIDataset(data=train_data, sampling_ratio=0.2)


def test_xlsx_to_csv():
    # df_attack = pd.read_excel('../_datasets/SWaT/SWaT_Dataset_Attack_v0.xlsx', sheet_name='Combined Data',
    #                           header=1)
    # print(df_attack)
    # df_attack.to_csv('../_datasets/SWaT/SWaT_Dataset_Attack_v0.csv')

    df_normal = pd.read_excel('../_datasets/SWaT/SWaT_Dataset_Normal_v0.xlsx', sheet_name='Normal.csv',
                              header=1)
    print(df_normal)
    df_normal.to_csv('../_datasets/SWaT/SWaT_Dataset_Normal_v0.csv')
