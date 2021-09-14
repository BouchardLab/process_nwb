import numpy as np
from scipy.fft import fft, ifft, rfft, irfft

from pynwb.ecephys import ElectricalSeries

from process_nwb.utils import _npads, _smart_pad, _trim, dtype


"""
The `resample_func` code is based on MNE-Python

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


def resample_func(X, num, npad=0, pad='reflect_limited', real=True, precision='single'):
    """Resample an array. Operates along the first dimension of the array. This is the low-level
    code. Users shoud likely use `resample()` rather than this function.

    Parameters
    ----------
    X : ndarray, (n_time, ...)
        Signal to resample.
    num : int
        Number of samples in resampled signal.
    npad : int
        Padding to add to beginning and end of timeseries. Default 0.
    pad : str
        Type of padding. The default is ``'reflect_limited'``.
    real : bool
        Whether rfft should be used for resampling or fft.
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.

    Returns
    -------
    y : array
        The x array resampled.

    Notes
    -----
    This uses edge padding to improve scipy.signal.resample's resampling method,
    which we have adapted for our use here.
    """
    X_dtype = dtype(X, precision)
    X = X.astype(X_dtype, copy=False)
    n_time = X.shape[0]
    ratio = float(num) / n_time
    npads, to_removes, new_len = _npads(X, npad, ratio=ratio)

    # do the resampling using an adaptation of scipy's FFT-based resample()
    X = _smart_pad(X, npads, pad)
    old_len = len(X)
    shorter = new_len < old_len
    use_len = new_len if shorter else old_len
    if real:
        X_fft = rfft(X, axis=0, workers=-1, overwrite_x=True)
        if use_len % 2 == 0:
            nyq = use_len // 2
            X_fft[nyq:nyq + 1] *= 2 if shorter else 0.5
        X_fft *= ratio
    else:
        X_fft = fft(X, axis=0, workers=-1, overwrite_x=True)
        X_fft[0] *= ratio
    del X
    if real:
        y = irfft(X_fft, n=new_len, axis=0, workers=-1, overwrite_x=True)
    else:
        y = ifft(X_fft, n=new_len, axis=0, workers=-1, overwrite_x=True).real

    # now let's trim it back to the correct size (if there was padding)
    y = _trim(y, to_removes)

    return y


def resample(X, new_freq, old_freq, real=True, axis=0, npad=0, precision='single'):
    """Resamples the timeseries from the original sampling frequency to a new frequency.

    Parameters
    ----------
    X : ndarray
        Input timeseries.
    new_freq : float
        New sampling frequency
    old_freq : float
        Original sampling frequency
    real : bool
        Whether rfft should be used for resampling or fft.
    axis : int
        Which axis to resample.
    npad : int
        Padding to add to beginning and end of timeseries. Default 0.
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.

    Returns
    -------
    Xds : array
        Downsampled data, dimensions (n_time_new, ...)
    """
    X_dtype = dtype(X, precision)
    X = X.astype(X_dtype, copy=False)
    axis = axis % X.ndim
    if axis != 0:
        X = np.swapaxes(X, 0, axis)

    n_time = X.shape[0]
    new_n_time = int(np.ceil(n_time * new_freq / old_freq))

    loop = False
    if X.size >= 10**8 and X.shape[1] > 1:
        loop = True

    if loop:
        Xds = np.zeros((new_n_time,) + X.shape[1:], dtype=X_dtype)
        for ii in range(X.shape[1]):
            Xds[:, ii] = resample_func(X[:, [ii]], new_n_time, npad=npad, real=real,
                                       precision=precision)[:, 0]
    else:
        Xds = resample_func(X, new_n_time, npad=npad, real=real, precision=precision)
    if axis != 0:
        Xds = np.swapaxes(Xds, 0, axis)

    return Xds


def store_resample(elec_series, processing, new_freq, axis=0, scaling=1e6, npad=0, precision='single'):
    """Resamples the `ElectricalSeries` from the original sampling frequency to a new frequency and
    store the results in a new ElectricalSeries.

    Parameters
    ----------
    elec_series : ElectricalSeries
        ElectricalSeries to process.
    processing : Processing module
        NWB Processing module to save processed data.
    new_freq : float
        New sampling frequency
    axis : int
        Which axis to downsample. Default is 0.
    scaling : float
        Scale the values by this. Can help with accuracy of downstream operations if the raw values
        are too small.
    npad : int
        Padding to add to beginning and end of timeseries. Default 0.
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.

    Returns
    -------
    X_ds : ndarray, (n_time_new, ...)
        Downsampled data.
    elec_series_ds : ElectricalSeries
        ElectricalSeries that holds X_ds.
    """
    new_freq = float(new_freq)
    X = elec_series.data[:] * scaling
    X_dtype = dtype(X, precision)
    X = X.astype(X_dtype, copy=False)
    old_freq = elec_series.rate

    X_ds = resample(X, new_freq, old_freq, axis=axis, npad=npad, precision=precision)

    elec_series_ds = ElectricalSeries('downsampled_' + elec_series.name,
                                      X_ds,
                                      elec_series.electrodes,
                                      starting_time=elec_series.starting_time,
                                      rate=new_freq,
                                      description='Downsampled: ' + elec_series.description)
    processing.add(elec_series_ds)
    return X_ds, elec_series_ds
