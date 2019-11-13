import numpy as np
import scipy as sp


from .fft import rfft, irfft
from numpy.fft import ifftshift, rfftfreq


__all__ = ['resample']


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


def _smart_pad(X, n_pad, pad='reflect_limited'):
    """Pad vector X."""
    n_time, n_channels = X.shape
    n_pad = np.asarray(n_pad)
    assert n_pad.shape == (2,)
    if (n_pad == 0).all():
        return x
    elif (n_pad < 0).any():
        raise RuntimeError('n_pad must be non-negative')
    if pad == 'reflect_limited':
        # need to pad with zeros if len(x) <= npad
        l_z_pad = np.zeros((max(n_pad[0] - len(X) + 1, 0), n_channels), dtype=X.dtype)
        r_z_pad = np.zeros((max(n_pad[1] - len(X) + 1, 0), n_channels), dtype=X.dtype)
        return np.concatenate([l_z_pad, 2 * X[[0]] - X[n_pad[0]:0:-1], X,
                               2 * X[[-1]] - X[-2:-n_pad[1] - 2:-1], r_z_pad], axis=0)
    else:
        return np.pad(X, (tuple(n_pad), 0), pad)


def _fft_resample(X, new_len, npads, to_removes, pad='reflect_limited'):
    """Do FFT resampling with a filter function
    Parameters
    ----------
    x : 1-d array
        The array to resample. Will be converted to float64 if necessary.
    new_len : int
        The size of the output array (before removing padding).
    npads : tuple of int
        Amount of padding to apply to the start and end of the
        signal before resampling.
    to_removes : tuple of int
        Number of samples to remove after resampling.
    pad : str
        The type of padding to use. Supports all :func:`np.pad` ``mode``
        options. Can also be "reflect_limited" (default), which pads with a
        reflected version of each vector mirrored on the first and last values
        of the vector, followed by zeros.
    Returns
    -------
    x : 1-d array
        Filtered version of x.
    """
    # add some padding at beginning and end to make this work a little cleaner
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    X = _smart_pad(X, npads, pad)
    old_len = len(X)
    shorter = new_len < old_len
    use_len = new_len if shorter else old_len
    X_fft = rfft(X, axis=0)
    if use_len % 2 == 0:
        nyq = use_len // 2
        X_fft[nyq:nyq + 1] *= 2 if shorter else 0.5
    y = irfft(X_fft, n=new_len, axis=0)

    # now let's trim it back to the correct size (if there was padding)
    if (to_removes > 0).any():
        y = y[to_removes[0]:y.shape[0] - to_removes[1]]

    return y


def resample_func(X, num, npad=100, axis=0, pad='reflect_limited'):
    """Resample an array.
    Operates along the last dimension of the array.
    Parameters
    ----------
    X : ndarray
        Signal to resample.
    up : float
        Factor to upsample by.
    down : float
        Factor to downsample by.
    npad : int
        Padding to add to beginning and end of timeseries.
    axis : int
        Axis along which to resample (default is the first axis).
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
    if axis < 0:
        axis = X.ndim + axis
    orig_last_axis = X.ndim - 1
    bad_msg = 'npad must be "auto" or an integer'
    if isinstance(npad, str):
        if npad != 'auto':
            raise ValueError(bad_msg)
        # Figure out reasonable pad that gets us to a power of 2
        min_add = min(n_time // 8, 100) * 2
        npad = 2 ** int(np.ceil(np.log2(n_time + min_add))) - n_time
        npad, extra = divmod(npad, 2)
        npads = np.array([npad, npad + extra], int)
    else:
        if npad != int(npad):
            raise ValueError(bad_msg)
        npads = np.array([npad, npad], int)

    # prep for resampling now
    orig_len = n_time + npads.sum()  # length after padding
    new_len = int(round(ratio * orig_len))  # length after resampling
    final_len = int(round(ratio * n_time))
    to_removes = [int(round(ratio * npads[0]))]
    to_removes.append(new_len - final_len - to_removes[0])
    to_removes = np.array(to_removes)
    # This should hold:
    # assert np.abs(to_removes[1] - to_removes[0]) <= int(np.ceil(ratio))

    # do the resampling using an adaptation of scipy's FFT-based resample()
    y = _fft_resample(X, new_len, npads, to_removes, pad)

    return y


def resample(X, new_freq, old_freq, kind=1, axis=0, same_sign=False):
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
    axis : int (optional)
        Axis along which to resample the data

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
        meds[axis % X.ndim] = med
        slices = [slice(None)] * X.ndim
        slices[axis % X.ndim] = slice(None, None, ratio)
        Xds = sp.signal.medfilt(X, meds)[slices]
    else:
        time = X.shape[axis]
        new_time = int(np.ceil(time * new_freq / old_freq))
        if kind == 0:
            ratio = int(ratio)
            if (ratio % 2) == 0:
                med = ratio + 1
            else:
                med = ratio
            meds = [1] * X.ndim
            meds[axis % X.ndim] = med
            Xf = sp.signal.medfilt(X, meds)
            f = sp.interpolate.interp1d(np.linspace(0, 1, time), Xf, axis=axis)
            Xds = f(np.linspace(0, 1, new_time))
        else:
            Xds = resample_func(X, new_time, axis=axis)

    return Xds
