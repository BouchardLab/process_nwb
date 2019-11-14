import numpy as np
import scipy as sp

from .fft import rfft, irfft
from numpy.fft import ifftshift, rfftfreq

from pynwb.ecephys import ElectricalSeries


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


def _npads(X, npad, ratio=1.):
    n_time = X.shape[0]
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
    return npads, to_removes, new_len


def _trim(X, to_removes):
    if (to_removes > 0).any():
        n_times = X.shape[0]
        X = X[to_removes[0]:n_times - to_removes[1]]
    return X

def _smart_pad(X, npads, pad='reflect_limited'):
    """Pad vector X."""
    n_time, n_channels = X.shape
    npads = np.asarray(npads)
    assert npads.shape == (2,)
    if (npads == 0).all():
        return x
    elif (npads < 0).any():
        raise RuntimeError('npad must be non-negative')
    if pad == 'reflect_limited':
        # need to pad with zeros if len(x) <= npad
        l_z_pad = np.zeros((max(npads[0] - len(X) + 1, 0), n_channels), dtype=X.dtype)
        r_z_pad = np.zeros((max(npads[1] - len(X) + 1, 0), n_channels), dtype=X.dtype)
        return np.concatenate([l_z_pad, 2 * X[[0]] - X[npads[0]:0:-1], X,
                               2 * X[[-1]] - X[-2:-npads[1] - 2:-1], r_z_pad], axis=0)
    else:
        return np.pad(X, (tuple(npads), 0), pad)


def resample_func(X, num, npad=100, pad='reflect_limited'):
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
    X_fft = rfft(X, axis=0)
    if use_len % 2 == 0:
        nyq = use_len // 2
        X_fft[nyq:nyq + 1] *= 2 if shorter else 0.5
    y = irfft(X_fft, n=new_len, axis=0)

    # now let's trim it back to the correct size (if there was padding)
    y = _trim(y, to_removes)

    return y


def resample(X, new_freq, old_freq, kind=1, same_sign=False):
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
            Xds = resample_func(X, new_n_time)

    return Xds


def store_resample(nwbfile, series_name, new_freq, kind=1, same_sign=False):
    new_freq = float(new_freq)
    electrical_series = nwbfile.acquisition[series_name]
    X = electrical_series.data[:]
    old_freq = electrical_series.rate

    Xds = resample(X, new_freq, old_freq, kind=kind, same_sign=same_sign)

    electrical_series_ds = ElectricalSeries('downsampled_' + electrical_series.name,
                                            Xds,
                                            electrical_series.electrodes,
                                            starting_time=electrical_series.starting_time,
                                            rate=new_freq,
                                            description='Downsampled: ' + electrical_series.description)
    nwbfile.add_acquisition(electrical_series_ds)
    return Xds, electrical_series_ds
