import numpy as np

from anomaly_detection import pw_reconstruction_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn import metrics

def test_pw_reconstruction_error():
    x = np.random.rand(10000)
    x_ = np.random.rand(10000)
    result = pw_reconstruction_error(x, x_)
    print(result)

    assert len(result) == 10000


def test_precision():
    print('\n')
    y_true = np.random.randint(0, 2, size=100)
    y_predict = np.random.randint(0, 2, size=100)
    print(y_true)
    print(y_predict)
    f1_score1 = f1_score(y_true, y_predict, pos_label=0)
    f1_score2 = f1_score(y_true, y_predict, pos_label=1)
    print(f1_score1, f1_score2)

