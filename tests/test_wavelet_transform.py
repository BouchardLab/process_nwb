import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from process_nwb.wavelet_transform import (wavelet_transform,
                                           gaussian,
                                           hamming)


@pytest.mark.parametrize("filters,hg_only,dim", [('human', False, 40),
                                                 ('human', True, 8),
                                                 ('changlab', False, 40),
                                                 ('changlab', True, 8),
                                                 ('rat', False, 54),
                                                 ('rat', True, 6)])
def test_wavelet_return(filters, hg_only, dim):
    """Test the return shape and dtype.
    """
    X = np.random.randn(1000, 32)
    rate = 200
    Xh, _, cfs, sds = wavelet_transform(X, rate, filters=filters, hg_only=hg_only)
    assert Xh.shape == (X.shape[0], X.shape[1], dim)
    assert Xh.dtype == np.complex


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
