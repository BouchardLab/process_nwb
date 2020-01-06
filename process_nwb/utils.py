import numpy as np


def log_spaced_cfs(fmin, fmax, ncfs):
    """
    Center frequencies that are uniform in log space
    """
    return np.logspace(np.log10(fmin), np.log10(fmax), ncfs)


def const_Q_sds(cfs, Q=8):
    return cfs / Q



def chang_sds(cfs):
    scale=0.39
    return 10. ** ( np.log10(scale) + .5 * (np.log10(cfs))) * np.sqrt(2.)

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
