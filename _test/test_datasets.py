from _datasets.datasets import WADIDataset
import numpy as np

def test_WADIDataset_sampling():
    train_dataset = WADIDataset(path_data='../_datasets/WADI/train.npy', sampling_ratio=0.2)
    test_dataset = WADIDataset(path_data='../_datasets/WADI/test.npy', path_label='../_datasets/WADI/labels.npy', sampling_ratio=0.2)
    assert len(train_dataset) != 0 and len(test_dataset) != 0

def test_dataset_npy():
    test_data = np.load('../_datasets/WADI/test.npy')
    labels = np.load('../_datasets/WADI/labels.npy')
    print(test_data.shape)
    print(labels.shape)
    print(labels)