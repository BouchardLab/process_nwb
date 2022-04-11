import numpy as np
from datetime import datetime
from dateutil.tz import tzlocal

from pynwb import NWBFile


def dtype(X, precision):
    """Return the type to cast to depending on precision and whether `X` is complex.

    Parameters
    ----------
    X : ndarray
        Input data.
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.
    """
    precision = precision.lower()
    if precision not in ['single', 'double']:
        raise ValueError(f'`precision` should be either `single` or `double`. Got {precision}.')
    if np.iscomplexobj(X):
        if precision == 'single':
            return np.complex64
        else:
            return complex
    else:
        if precision == 'single':
            return np.float32
        else:
            return float


def log_spaced_cfs(fmin, fmax, ncfs):
    """Return log-spaced center frequencies.

    Parameters
    ----------
    fmin : float
        Minimum center frequency.
    fmax : float
        Maximum center frequency.
    ncfs : int
        Number of bands to generate.
    """
    return np.logspace(np.log10(fmin), np.log10(fmax), ncfs)


def const_Q_sds(cfs, Q=8):
    """Compute constant Q bandwidths for a set of center frequencies.

    Parameters
    ----------
    cfs : ndarray
        Center frequencies.
    Q : float
        Q value for the wavelet. Default = 8.
    """
    return cfs / Q


def chang_sds(cfs):
    """Compute variable bandwidths for a set of center frequencies used by the Chang lab.

    Parameters
    ----------
    cfs : ndarray
        Center frequencies.
    Q : float
        Q value for the wavelet. Default = 8.
    """
    scale = 0.39
    return 10. ** (np.log10(scale) + .5 * (np.log10(cfs))) * np.sqrt(2.)


"""
The following code is based on MNE-Python

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
    """Calculate padding parameters.
    """
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
    """Trim the padding.
    """
    if (to_removes > 0).any():
        n_times = X.shape[0]
        X = X[to_removes[0]:n_times - to_removes[1]]
    return X


def _smart_pad(X, npads, pad='reflect_limited'):
    """Pad vector X.
    """
    other_shape = X.shape[1:]
    npads = np.asarray(npads)
    assert npads.shape == (2,)
    if (npads == 0).all():
        return X
    elif (npads < 0).any():
        raise RuntimeError('npad must be non-negative')
    if pad == 'reflect_limited':
        # need to pad with zeros if len(x) <= npad
        l_z_pad = np.zeros((max(npads[0] - len(X) + 1, 0),) + other_shape, dtype=X.dtype)
        r_z_pad = np.zeros((max(npads[1] - len(X) + 1, 0),) + other_shape, dtype=X.dtype)
        return np.concatenate([l_z_pad, 2 * X[[0]] - X[npads[0]:0:-1], X,
                               2 * X[[-1]] - X[-2:-npads[1] - 2:-1], r_z_pad], axis=0)
    else:
        return np.pad(X, (tuple(npads), 0), pad)


def generate_synthetic_data(duration, nchannels, rate, high_gamma=True,
                            linenoise=True, seed=0):
    """Generate synthetic data by smoothing white noise with a boxcar for
    testing or tutorials. Optional high gamma and 60 Hz linenoise can be added.

    Parameters
    ----------
    duration : float
        Data duration in seconds.
    nchannels : int
        Number of channels
    rate : float
        Sampling rate in Hz
    high_gamma : bool
        If True, include extra modulating power in the high gamma range.
    linenoise : bool
        If True, include 60 Hz linenoise.

    Returns
    -------
    neural_data : ndarray (time, channels)
        Synthetic neural data
    """
    kernel_length = 50
    rng = np.random.default_rng(seed=seed)
    neural_data = rng.standard_normal((int(duration * rate), nchannels)) / 100.
    kernel = np.ones(kernel_length) / kernel_length
    for ch in range(nchannels):
        neural_data[:, ch] = np.convolve(neural_data[:, ch], kernel, mode='same')
    neural_data /= neural_data.std() * 2.

    if high_gamma or linenoise:
        t = np.linspace(0, duration, neural_data.shape[0])[:, np.newaxis]
    if high_gamma:
        # Add a high gamma signal at 100 Hz that modulates at 2 Hz
        phase = 2 * np.pi * rng.random(nchannels)[np.newaxis]
        high_gamma = np.sin(2 * np.pi * t * 100. + phase)
        phase = 2 * np.pi * rng.random(nchannels)[np.newaxis]
        high_gamma *= np.sin(2 * np.pi * t * 1. + phase)**2 + 0.2
        neural_data += high_gamma

    if linenoise:
        # Add common noise but with different weights to each channel
        weights = rng.standard_normal((1, nchannels))
        if rate > 120.:
            for ii, hz in enumerate(np.arange(60., rate, 60.)):
                line_noise = np.sin(2 * np.pi * t * hz) / 2. ** (ii + 1)
                neural_data += line_noise * weights

    return neural_data


def generate_nwbfile(nchannels=4):
    """Generate an `NWBFile` object that an `ElectricalSeries` can be added to.

    Returns
    -------
    nwbfile : NWBFile
        NWBFile object
    Device : Device
        Device for the ElectricalSeries
    electrode_group : ElectrodeGroup
        ElectrodeGroup for the ElectricalSeries
    electrodes : Electrodes
        Electrodes for the ElectricalSeries
    """
    start_time = datetime(2020, 12, 31, 11, 28, tzinfo=tzlocal())
    nwbfile = NWBFile(session_description='Demonstrate `process_nwb` on an NWBFile',
                      identifier='NWB123',
                      session_start_time=start_time)
    device = nwbfile.create_device(name='ECoG_grid')
    electrode_group = nwbfile.create_electrode_group('Grid',
                                                     description='Grid',
                                                     location='cortex',
                                                     device=device)
    for idx in range(nchannels):
        nwbfile.add_electrode(id=idx,
                              x=1.0, y=2.0, z=3.0,
                              imp=float(-idx),
                              location='cortex', filtering='none',
                              group=electrode_group)
    electrodes = nwbfile.create_electrode_table_region(list(range(nchannels)), 'Electrodes')

    return nwbfile, device, electrode_group, electrodes
