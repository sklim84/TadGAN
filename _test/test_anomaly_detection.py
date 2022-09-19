import numpy as np

from anomaly_detection import pw_reconstruction_error


def test_pw_reconstruction_error():
    x = np.random.rand(10000)
    x_ = np.random.rand(10000)
    result = pw_reconstruction_error(x, x_)
    print(result)

    assert len(result) == 10000
