import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import model
def test_metrics():
    print('\n')
    y_predict = np.random.randint(2, size=10)
    y_true = np.random.randint(2, size=10)

    print(y_predict)
    print(y_true)

    accuracy = accuracy_score(y_true, y_predict)
    print(accuracy)
    precision = precision_score(y_true, y_predict, pos_label=0)
    print(precision)
    recall = recall_score(y_true, y_predict, pos_label=0)
    print(recall)
    f1score = f1_score(y_true, y_predict, pos_label=0)
    print(f1score)


def test_permutation():
    data = np.random.rand(10000)
    label = np.random.rand(10000)

    length = len(data)
    sampling_ratio = 0.2
    idx_sample = np.random.permutation(length)[:int(np.floor(sampling_ratio * length))]
    print(type(idx_sample))
    # print(idx_sample)
    idx_sample = np.sort(idx_sample)
    print(idx_sample)

def test_load_model():
    encoder = model.Encoder('', 12, 123, 20)
    state_dict = torch.load('../models/encoder_wadi_0.001_20.pt')
    # print(state_dict)
    encoder.load_state_dict(state_dict)
    print(encoder.state_dict())


def test_np_dim_sum():
    x = np.random.rand(10000, 123)
    x_ = np.random.rand(10000, 123)

    result = np.absolute(x - x_)
    print(result.shape)

    result_sum = np.sum(result, axis=1)
    print(result_sum.shape)
    result_mean = np.mean(result, axis=1)
    print(result_mean.shape)

def test_threshold():
    for threshold in np.arange(0, 1, 0.001):
        print(threshold)