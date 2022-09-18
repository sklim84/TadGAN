import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
