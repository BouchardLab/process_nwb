import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from process_nwb.wavelet_transform import (wavelet_transform,
                                           gaussian,
                                           hamming)


@pytest.mark.parametrize("filters,hg_only,dim,rate", [('human', False, 40, 400.),
                                                      ('human', True, 8, 400.),
                                                      ('changlab', False, 40, 400.),
                                                      ('changlab', True, 8, 400.),
                                                      ('rat', False, 54, 2400.),
                                                      ('rat', True, 6, 2400.)])
def test_wavelet_return(filters, hg_only, dim, rate):
    """Test the return shape and dtype.
    """
    X = np.random.randn(1000, 32)
    Xh, _, cfs, sds = wavelet_transform(X, rate, filters=filters, hg_only=hg_only)
    assert Xh.shape == (X.shape[0], X.shape[1], dim)
    assert np.dtype(Xh.dtype) == np.dtype(np.complex64)

    Xh, _, cfs, sds = wavelet_transform(X, rate, filters=filters, hg_only=hg_only,
                                        precision='double')
    assert Xh.shape == (X.shape[0], X.shape[1], dim)
    assert np.dtype(Xh.dtype) == np.dtype(complex)


@pytest.mark.parametrize("filters,hg_only,dim,rate", [('human', False, 40, 399.),
                                                      ('human', True, 8, 200.),
                                                      ('changlab', False, 40, 399.),
                                                      ('changlab', True, 8, 200.),
                                                      ('rat', False, 54, 2399.),
                                                      ('rat', True, 6, 200.)])
def test_wavelet_nyquist(filters, hg_only, dim, rate):
    """Test the return shape and dtype.
    """
    X = np.random.randn(1000, 32)
    with pytest.raises(ValueError):
        wavelet_transform(X, rate, filters=filters, hg_only=hg_only)


def test_gaussian_kernel():
    """Test the gaussian kernel.
    """
    ker = gaussian(1111, 200, 50, 10)
    assert_allclose(np.linalg.norm(ker), 1)
    assert_equal(ker >= 0., True)


def test_hamming_kernel():
    """Test the hamming kernel.
    """
    ker = hamming(1111, 200, 50, 60)
    assert_allclose(np.linalg.norm(ker), 1)
    assert_equal(ker >= 0., True)
    """
                                                  , ,
                                                  """
