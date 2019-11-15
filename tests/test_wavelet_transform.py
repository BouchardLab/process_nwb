import numpy as np

from process_nwb.wavelet_transform import wavelet_transform

def test_wavelet_return():
    """
    Test the return shape and dtype.
    """
    X = np.random.randn(1000, 32)
    rate = 200
    Xh, _ = wavelet_transform(X, rate)
    assert Xh.shape == (X.shape[0], X.shape[1], 40)
    assert Xh.dtype == np.complex
