import numpy as np
from scipy.signal import firwin2, filtfilt

from .fft import rfftfreq, rfft, irfft
from .utils import _npads, _smart_pad, _trim


__all__ = ['apply_linenoise_notch']

def apply_notches(X, notches, rate, fft=True):
    delta = 1.
    if fft:
        fs = rfftfreq(X.shape[0], 1. / rate)
        fd = rfft(X, axis=0)
    else:
        nyquist = rate/2.
        n_taps = 1001
        gain = [1, 1, 0, 0, 1, 1]
    for notch in notches:
        if fft:
            window_mask = np.logical_and(fs > notch - delta, fs < notch + delta)
            window_size = window_mask.sum()
            window = np.hamming(window_size)
            fd[window_mask] = fd[window_mask] * (1.-window)[:, np.newaxis]
        else:
            freq = np.array([0, notch-delta, notch-delta/2.,
                             notch+delta/2, notch+delta, nyquist]) / nyquist
            filt = firwin2(n_taps, freq, gain)
            X = filtfilt(filt, np.array([1]), X, axis=0)
    if fft:
        X = irfft(fd, axis=0)
    return X


def apply_linenoise_notch(X, rate, fft=True):
    """
    Apply Notch filter at 60 Hz and its harmonics

    Parameters
    ----------
    X : array
        Input data, dimensions (n_channels, n_timePoints)
    rate : float
        Number of samples per second

    Returns
    -------
    X : array
        Denoised data, dimensions (n_channels, n_timePoints)
    """

    nyquist = rate / 2.
    noise_hz = 60.
    npad = rate
    if nyquist < noise_hz:
        return X
    notches = np.arange(noise_hz, nyquist, noise_hz)
    npads, to_removes, _ = _npads(X, npad)
    X = _smart_pad(X, npads)

    Xp = apply_notches(X, notches, rate, fft=fft)
    Xp = _trim(Xp, to_removes)
    return Xp
