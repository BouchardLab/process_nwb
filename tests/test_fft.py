import pytest
import numpy as np
from numpy.testing import assert_allclose
from process_nwb.fft import (mklrfft, mklirfft, mklfft, mklifft,
                             rfft, irfft, fft, ifft)


@pytest.fixture
def X():
    return np.random.randn(18134054, 2)


@pytest.mark.parametrize("myfft", [mklrfft, mklirfft, mklfft, mklifft])
def test_mklrfft_fix(myfft, X):
    """Check that mkl bug still exists. When this test fails, the mkl logic in fft.py can be
    reassesed."""
    with pytest.raises(ValueError):
        myfft(X, axis=0)


@pytest.mark.parametrize("myfft", [rfft, irfft, fft, ifft])
def test_rfft_fix(myfft, X):
    """Check that patched functions work on shape with known problem."""
    myfft(X, axis=0)


@pytest.mark.parametrize("forwinv", [(rfft, irfft), (fft, ifft)])
def test_roundtrip(forwinv, X):
    """Test that patched functions are setup correctly.
    """
    forw, inv = forwinv
    assert_allclose(inv(forw(X, axis=0), axis=0), X)
    assert_allclose(inv(forw(X[:150], axis=0), axis=0), X[:150])
