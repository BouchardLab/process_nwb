import numpy as np
from numpy.testing import assert_allclose

from process_nwb.common_referencing import CAR, subtract_CAR


def test_CAR():
    X = np.tile(np.arange(100)[np.newaxis], (3, 1))

    mean = CAR(X, mean_frac=1.)
    assert_allclose(mean, 49.5)

    X[:, 0] = -100
    X[:, -1] = 111
    ninety_five = CAR(X)
    assert_allclose(ninety_five, 49.5)


def test_subtract_CAR():
    X = np.tile(np.arange(100)[np.newaxis], (3, 1))

    X_CAR = subtract_CAR(X, mean_frac=1.)
    assert_allclose(X_CAR.mean(axis=1), 0)

    X[:, 0] = -100
    X[:, -1] = 111
    X_CAR = subtract_CAR(X)
    assert_allclose(X_CAR.mean(axis=1), -.88)
