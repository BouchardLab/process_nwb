import numpy as np

from .fft import fftfreq, fft, ifft


__all__ = ['gaussian', 'hamming', 'wavelet_transform']


def gaussian(X, rate, center, sd):
    n_time = X.shape[0]
    freq = fftfreq(n_time, 1./rate)

    k = np.exp((-(np.abs(freq) - center)**2)/(2 * (sd**2)))
    k /= np.linalg.norm(k)

    return k


def hamming(X, rate, min_freq, max_freq):
    n_time = X.shape[0]
    freq = fftfreq(n_time, 1./rate)

    pos_in_window = np.logical_and(freq >= min_freq, freq <= max_freq)
    neg_in_window = np.logical_and(freq <= -min_freq, freq >= -max_freq)

    k = np.zeros(len(freq))
    window_size = np.count_nonzero(pos_in_window)
    window = np.hamming(window_size)
    k[pos_in_window] = window
    window_size = np.count_nonzero(neg_in_window)
    window = np.hamming(window_size)
    k[neg_in_window] = window
    k /= np.linalg.norm(k)

    return k


def wavelet_transform(X, rate, filters=None, X_fft_h=None):
    """
    Apply bandpass filtering with wavelet transform using
    a prespecified set of filters.

    Parameters
    ----------
    X : ndarray (n_time, n_channels)
        Input data, dimensions
    rate : float
        Number of samples per second.
    filters : filter or list of filters (optional)
        One or more bandpass filters

    Returns
    -------
    Xh : ndarray, complex
        Bandpassed analytic signal
    X_fft_h : ndarray, complex
        Product of X_ff and heavyside.
    """
    n_time = X.shape[0]
    freq = fftfreq(n_time, 1. / rate)

    squeeze_0 = False
    if filters is None:
        squeeze_0 = True

    if not isinstance(filters, list):
        filters = [filters]

    Xh = np.zeros((len(filters),) + X.shape, dtype=np.complex)
    if X_fft_h is None:
        # Heavyside filter
        h = np.zeros(len(freq))
        h[freq > 0] = 2.
        h = h[:, np.newaxis]
        X_fft_h = fft(X, axis=0) * h
    for ii, f in enumerate(filters):
        if f is None:
            Xh[ii] = ifft(X_fft_h, axis=0)
        else:
            f = f / np.linalg.norm(f)
            Xh[ii] = ifft(X_fft_h * f, axis=0)
    if squeeze_0:
        Xh = Xh[0]

    return Xh, X_fft_h


def store_wavelet_transform(X, nwb, rate, filters=None, X_fft_h=None):
    """
    Apply bandpass filtering with wavelet transform using
    a prespecified set of filters.

    Parameters
    ----------
    X : ndarray (n_time, n_channels)
        Input data, dimensions
    rate : float
        Number of samples per second.
    filters : filter or list of filters (optional)
        One or more bandpass filters

    Returns
    -------
    Xh : ndarray, complex
        Bandpassed analytic signal
    X_fft_h : ndarray, complex
        Product of X_ff and heavyside.
    """
    n_time = X.shape[0]
    freq = fftfreq(n_time, 1. / rate)

    squeeze_0 = False
    if filters is None:
        squeeze_0 = True

    if not isinstance(filters, list):
        filters = [filters]

    Xh = np.zeros((len(filters),) + X.shape, dtype=np.complex)
    if X_fft_h is None:
        # Heavyside filter
        h = np.zeros(len(freq))
        h[freq > 0] = 2.
        h = h[:, np.newaxis]
        X_fft_h = fft(X, axis=0) * h
    for ii, f in enumerate(filters):
        if f is None:
            Xh[ii] = ifft(X_fft_h, axis=0)
        else:
            f = f / np.linalg.norm(f)
            Xh[ii] = ifft(X_fft_h * f, axis=0)
    if squeeze_0:
        Xh = Xh[0]

    return Xh, X_fft_h
