import numpy as np

from .fft import fftfreq, fft, ifft
from .utils import (_npads, _smart_pad, _trim,
                    log_spaced_cfs, const_Q_sds,
                    chang_sds)

from pynwb.misc import DecompositionSeries


def gaussian(n_time, rate, center, sd):
    freq = fftfreq(n_time, 1. / rate)

    k = np.exp((-(np.abs(freq) - center) ** 2) / (2 * (sd ** 2)))
    k /= np.linalg.norm(k)

    return k


def hamming(n_time, rate, min_freq, max_freq):
    freq = fftfreq(n_time, 1. / rate)

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


def wavelet_transform(X, rate, filters='rat', hg_only=True, X_fft_h=None, npad=None):
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
        Product of X_fft and heavyside.
    """
    if npad is None:
        npad = int(rate)
    if X_fft_h is None:
        npads, to_removes, _ = _npads(X, npad)
        X = _smart_pad(X, npads)
        n_time = X.shape[0]
    else:
        n_time = X_fft_h.shape[0]
    freq = fftfreq(n_time, 1. / rate)

    # Calculate center frequencies
    if filters in ['human', 'changlab']:
        cfs = log_spaced_cfs(4.0749286538265, 200, 40)
    elif filters == 'rat':
        cfs = log_spaced_cfs(2.6308, 1200., 54)
    else:
        raise NotImplementedError

    # Subselect high gamma bands
    if hg_only:
        idxs = np.logical_and(cfs >= 70., cfs <= 150.)
        cfs = cfs[idxs]

    # Calculate bandwidths
    if filters in ['rat', 'human']:
        sds = const_Q_sds(cfs)
    elif filters == 'changlab':
        sds = chang_sds(cfs)
    else:
        raise NotImplementedError

    filters = []
    for cf, sd in zip(cfs, sds):
        filters.append(gaussian(n_time, rate, cf, sd))

    Xh = np.zeros(X.shape + (len(filters),), dtype=np.complex)
    if X_fft_h is None:
        # Heavyside filter with 0 DC
        h = np.zeros(len(freq))
        h[freq > 0] = 2.
        h = h[:, np.newaxis]
        X_fft_h = fft(X, axis=0) * h

    for ii, f in enumerate(filters):
        if f is None:
            Xh[..., ii] = ifft(X_fft_h, axis=0)
        else:
            f = f / np.linalg.norm(f)
            Xh[..., ii] = ifft(X_fft_h * f[:, np.newaxis], axis=0)

    Xh = _trim(Xh, to_removes)

    return Xh, X_fft_h, cfs, sds


def store_wavelet_transform(elec_series, processing, npad=None, filters='rat',
                            X_fft_h=None, abs_only=True, hg_only=True):
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
    X_fft_h : ndarray (n_time, n_channels)
        Precomputed product of X_fft and heavyside.
    abs_only : bool
        If True, only the amplitude is stored.
    hg_only : bool
        If True, only the amplitudes in the high gamma range is computed.

    Returns
    -------
    Xh : ndarray, complex
        Bandpassed analytic signal
    X_fft_h : ndarray, complex
        Product of X_fft and heavyside.
    """
    X = elec_series.data[:]
    rate = elec_series.rate
    if npad is None:
        npad = int(rate)
    X_wvlt, _, cfs, sds = wavelet_transform(X, rate, filters=filters, X_fft_h=X_fft_h,
                                            hg_only=hg_only, npad=npad)
    elec_series_wvlt_amp = DecompositionSeries('wvlt_amp_' + elec_series.name,
                                               abs(X_wvlt),
                                               metric='amplitude',
                                               source_timeseries=elec_series,
                                               starting_time=elec_series.starting_time,
                                               rate=rate,
                                               description=('Wavlet: ' +
                                                            elec_series.description))
    series = [elec_series_wvlt_amp]
    if not abs_only:
        elec_series_wvlt_phase = DecompositionSeries('wvlt_phase_' + elec_series.name,
                                                     np.angle(X_wvlt),
                                                     metric='phase',
                                                     source_timeseries=elec_series,
                                                     starting_time=elec_series.starting_time,
                                                     rate=rate,
                                                     description=('Wavlet: ' +
                                                                  elec_series.description))
        series.append(elec_series_wvlt_phase)

    for es in series:
        for ii, (cf, sd) in enumerate(zip(cfs, sds)):
            es.add_band(band_name=str(ii), band_mean=cf,
                        band_stdev=sd, band_limits=(-1, -1))

        processing.add(es)
    return X_wvlt, series
