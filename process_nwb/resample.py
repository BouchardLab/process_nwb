import numpy as np
import scipy as sp
from pynwb.ecephys import ElectricalSeries

from .utils import _npads, _smart_pad, _trim
from .fft import fft, ifft, rfft, irfft, rfftfreq


__all__ = ['resample',
           'store_resample']


"""
The fft resampling code is based on MNE-Python

Copyright © 2011-2019, authors of MNE-Python
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def resample_func(X, num, npad=100, pad='reflect_limited', real=True):
    """Resample an array.
    Operates along the last dimension of the array.

    Parameters
    ----------
    X : ndarray, (n_time, ...)
        Signal to resample.
    up : float
        Factor to upsample by.
    down : float
        Factor to downsample by.
    npad : int
        Padding to add to beginning and end of timeseries.
    pad : str
        Type of padding. The default is ``'reflect_limited'``.

    Returns
    -------
    y : array
        The x array resampled.

    Notes
    -----
    This uses edge padding to improve scipy.signal.resample's resampling method,
    which we have adapted for our use here.
    """
    n_time = X.shape[0]
    ratio = float(num) / n_time
    npads, to_removes, new_len = _npads(X, npad, ratio=ratio)

    # do the resampling using an adaptation of scipy's FFT-based resample()
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    X = _smart_pad(X, npads, pad)
    old_len = len(X)
    shorter = new_len < old_len
    use_len = new_len if shorter else old_len
    if real:
        X_fft = rfft(X, axis=0)
        if use_len % 2 == 0:
            nyq = use_len // 2
            X_fft[nyq:nyq + 1] *= 2 if shorter else 0.5
        X_fft *= ratio
    else:
        X_fft = fft(X, axis=0)
        X_fft[0] *= ratio
    if real:
        y = irfft(X_fft, n=new_len, axis=0)
    else:
        y = ifft(X_fft, n=new_len, axis=0).real

    # now let's trim it back to the correct size (if there was padding)
    y = _trim(y, to_removes)

    return y


def resample(X, new_freq, old_freq, kind=1, same_sign=False, real=True):
    """
    Resamples the ECoG signal from the original
    sampling frequency to a new frequency.

    Parameters
    ----------
    X : ndarray, (n_time, ...)
        Input timeseries.
    new_freq : float
        New sampling frequency
    old_freq : float
        Original sampling frequency

    Returns
    -------
    Xds : array
        Downsampled data, dimensions (n_time_new, ...)
    """
    ratio = float(old_freq) / new_freq
    if np.allclose(ratio, int(ratio)) and same_sign:
        ratio = int(ratio)
        if (ratio % 2) == 0:
            med = ratio + 1
        else:
            med = ratio
        meds = [1] * X.ndim
        meds[0] = med
        slices = [slice(None)] * X.ndim
        slices[0] = slice(None, None, ratio)
        Xds = sp.signal.medfilt(X, meds)[slices]
    else:
        n_time = X.shape[0]
        new_n_time = int(np.ceil(n_time * new_freq / old_freq))
        if kind == 0:
            ratio = int(ratio)
            if (ratio % 2) == 0:
                med = ratio + 1
            else:
                med = ratio
            meds = [1] * X.ndim
            meds[0] = med
            Xf = sp.signal.medfilt(X, meds)
            f = sp.interpolate.interp1d(np.linspace(0, 1, n_time), Xf, axis=0)
            Xds = f(np.linspace(0, 1, new_n_time))
        else:
            npad = int(max(new_freq, old_freq))
            Xds = resample_func(X, new_n_time, npad=npad, real=real)

    return Xds


def store_resample(electrical_series, processing, new_freq, kind=1, same_sign=False,
                   scaling=1e6):
    new_freq = float(new_freq)
    X = electrical_series.data[:] * scaling
    old_freq = electrical_series.rate

    Xds = resample(X, new_freq, old_freq, kind=kind, same_sign=same_sign)

    electrical_series_ds = ElectricalSeries('downsampled_' + electrical_series.name,
                                            Xds,
                                            electrical_series.electrodes,
                                            starting_time=electrical_series.starting_time,
                                            rate=new_freq,
                                            description='Downsampled: ' + electrical_series.description)
    processing.add(electrical_series_ds)
    return Xds, electrical_series_ds
