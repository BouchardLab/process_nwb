import numpy as np
from scipy.signal import firwin2, filtfilt
from scipy.fft import rfftfreq, rfft, irfft

from process_nwb.utils import _npads, _smart_pad, _trim, dtype


def _apply_notches(X, notches, rate, fft=True, precision='single'):
    """Low-level code which applies notch filters.

    Parameters
    ----------
    X : ndarray, (n_time, n_channels)
        Input data.
    notches : ndarray
        Frequencies to notch filter.
    rate : float
        Number of samples per second for X.
    fft : bool
        Whether to filter in the time or frequency domain.
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.

    Returns
    -------
    Xp : ndarray, (n_time, n_channels)
        Notch filtered data.
    """
    X_dtype = dtype(X, precision)
    X = X.astype(X_dtype, copy=False)
    delta = 1.
    if fft:
        fs = rfftfreq(X.shape[0], 1. / rate)
        fd = rfft(X, axis=0, workers=-1)
    else:
        nyquist = rate / 2.
        n_taps = 1001
        gain = [1, 1, 0, 0, 1, 1]
    for notch in notches:
        if fft:
            window_mask = np.logical_and(fs > notch - delta, fs < notch + delta)
            window_size = window_mask.sum()
            window = np.hamming(window_size)
            fd[window_mask] = fd[window_mask] * (1. - window)[:, np.newaxis]
        else:
            freq = np.array([0, notch - delta, notch - delta / 2.,
                             notch + delta / 2., notch + delta, nyquist]) / nyquist
            filt = firwin2(n_taps, freq, gain)
            Xp = filtfilt(filt, np.array([1]), X, axis=0)
    if fft:
        Xp = irfft(fd, n=X.shape[0], axis=0, workers=-1)
    return Xp.astype(X_dtype, copy=False)


def apply_linenoise_notch(X, rate, fft=True, noise_hz=60., npad=0, precision='single'):
    """Apply notch filters at 60 Hz (by default) and its harmonics.

    Filters +/- 1 Hz around the frequencies.

    Parameters
    ----------
    X : ndarray, (n_time, n_channels)
        Input data.
    rate : float
        Number of samples per second for X.
    fft : bool
        Whether to filter in the time or frequency domain.
    noise_hz: float
        Frequency to notch out
    npad : int
        Padding to add to beginning and end of timeseries. Default 0.
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.

    Returns
    -------
    Xp : ndarray, (n_time, n_channels)
        Notch filtered data.
    """
    X_dtype = dtype(X, precision)
    X = X.astype(X_dtype, copy=False)
    nyquist = rate / 2.
    if nyquist < noise_hz:
        return X
    notches = np.arange(noise_hz, nyquist, noise_hz)
    npads, to_removes, _ = _npads(X, npad)
    X = _smart_pad(X, npads)

    Xp = _apply_notches(X, notches, rate, fft=fft, precision=precision)
    Xp = _trim(Xp, to_removes)
    return Xp.astype(X_dtype, copy=False)
