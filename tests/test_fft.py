import pytest
import numpy as np
from numpy.testing import assert_allclose
from process_nwb.fft import (mklrfft, mklirfft, mklfft, mklifft,
                             rfft, irfft, fft, ifft)


def test_mklrfft_fix():
    """Check that mkl bug still exists. When this test passes, the mkl logic in fft.py can be
    reassesed."""
    X = np.random.randn(18134053, 2)
    with pytest.raises(ValueError):
        mklrfft(X, axis=0)
    with pytest.raises(ValueError):
        mklirfft(X, axis=0)
    with pytest.raises(ValueError):
        mklfft(X, axis=0)
    with pytest.raises(ValueError):
        mklifft(X, axis=0)


def test_rfft_fix():
    """Check that patched and original functions work on shape with known problem."""
    X = np.random.randn(18134053, 2)
    rfft(X, axis=0)
    irfft(X, axis=0)
    fft(X, axis=0)
    ifft(X, axis=0)


def test_roundtrip():
    """Test that patched functions are setup correctly.
    """
    X = np.random.randn(18134054, 2)
    assert_allclose(irfft(rfft(X, axis=0), axis=0), X)
    assert_allclose(ifft(fft(X, axis=0), axis=0), X)
