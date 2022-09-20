#!/usr/bin/env python
# coding: utf-8

import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class Encoder(nn.Module):

    def __init__(self, batch_size, signal_shape=100, out_features=20):
        super(Encoder, self).__init__()
        self.signal_shape = signal_shape
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=self.signal_shape, hidden_size=20, num_layers=1, bidirectional=True)
        self.dense = nn.Linear(in_features=40, out_features=out_features)

    def forward(self, x):
        x = x.view(1, self.batch_size, self.signal_shape).float()
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return (x)

class Decoder(nn.Module):
    def __init__(self, signal_shape=100, input_size=20):
        super(Decoder, self).__init__()
        self.signal_shape = signal_shape
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=2, bidirectional=True)
        self.dense = nn.Linear(in_features=128, out_features=self.signal_shape)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return (x)

class CriticX(nn.Module):
    def __init__(self, batch_size, signal_shape=100):
        super(CriticX, self).__init__()
        self.signal_shape = signal_shape
        self.batch_size = batch_size
        self.dense1 = nn.Linear(in_features=self.signal_shape, out_features=20)
        self.dense2 = nn.Linear(in_features=20, out_features=1)

    def forward(self, x):
        x = x.view(1, self.batch_size, self.signal_shape).float()
        x = self.dense1(x)
        x = self.dense2(x)
        return (x)

class CriticZ(nn.Module):
    def __init__(self, in_features=20):
        super(CriticZ, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=1)

    def forward(self, x):
        x = self.dense1(x)
        return (x)

def unroll_signal(self, x):
    x = np.array(x).reshape(100)
    return np.median(x)

def test(self):
    """
    Returns a dataframe with original value, reconstructed value, reconstruction error, critic score
    """
    df = self.test_dataset.copy()
    X_ = list()

    RE = list()  #Reconstruction error
    CS = list()  #Critic score

    for i in range(0, df.shape[0]):
        x = df.rolled_signal[i]
        x = tf.reshape(x, (1, 100, 1))
        z = encoder(x)
        z = tf.expand_dims(z, axis=2)
        x_ = decoder(z)

        re = dtw_reconstruction_error(tf.squeeze(x_).numpy(), tf.squeeze(x).numpy()) #reconstruction error
        cs = critic_x(x)
        cs = tf.squeeze(cs).numpy()
        RE.append(re)
        CS.append(cs)

        x_ = unroll_signal(x_)

        X_.append(x_)

    df['generated_signals'] = X_

    return df
