import numpy as np

from .fft import fftfreq, fft, ifft
from .utils import (_npads, _smart_pad, _trim,
                    log_spaced_cfs, const_Q_sds,
                    chang_sds)

from pynwb.misc import DecompositionSeries


def gaussian(n_time, rate, center, sd):
    """Generates a normalized gaussian kernel.

    Parameters
    ----------
    n_time : int
        Number of samples
    rate : float
        Sampling rate of kernel (Hz).
    center : float
        Center frequency (Hz).
    sd : float
        Bandwidth (Hz).
    """
    freq = fftfreq(n_time, 1. / rate)

    k = np.exp((-(np.abs(freq) - center) ** 2) / (2 * (sd ** 2)))
    k /= np.linalg.norm(k)

    return k


def hamming(n_time, rate, min_freq, max_freq):
    """Generates a normalized Hamming kernel.

    Parameters
    ----------
    n_time : int
        Number of samples
    rate : float
        Sampling rate of kernel (Hz).
    min_freq : float
        Band minimum frequency (Hz).
    max_freq : float
        Band maximum frequency (Hz).
    """
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
    """Apply a wavelet transform using a prespecified set of filters.

    Calculates the center frequencies and bandwidths for the wavelets and applies them along with
    a heavyside function to the fft of the signal before performing an inverse fft.

    Parameters
    ----------
    X : ndarray (n_time, n_channels)
        Input data, dimensions
    rate : float
        Number of samples per second.
    filters : str (optional)
        Which type of filters to use. Options are
        'rat': center frequencies spanning 2-1200 Hz, constant Q, 54 bands
        'human': center frequencies spanning 4-200 Hz, constant Q, 40 bands
        'changlab': center frequencies spanning 4-200 Hz, variable Q, 40 bands
        Note - calculating center frequencies above rate/2 raises a ValueError
    hg_only : bool
        If True, only the amplitudes in the high gamma range [70-150 Hz] is computed.
    X_fft_h : ndarray (n_time, n_channels)
        Precomputed product of X_fft and heavyside. Useful for when bands are computed
        independently.
    npad : int
        Length of padding in samples.

    Returns
    -------
    Xh : ndarray, complex
        Bandpassed analytic signal
    X_fft_h : ndarray, complex
        Product of X_fft and heavyside.
    cfs : ndarray
        Center frequencies used.
    sds : ndarray
        Bandwidths used.
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

    # Raise exception if sample rate too small
    if max(cfs) > rate/2:
        raise ValueError('Unable to compute wavelet transform above Nyquist rate.' +
            ' Increase your rate to at least twice your desired maximum frequency of interest.')

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


def store_wavelet_transform(elec_series, processing, filters='rat', hg_only=True, X_fft_h=None,
                            abs_only=True, npad=None):
    """Apply a wavelet transform using a prespecified set of filters. Results are stored in the
    NWB file as a `DecompositionSeries`.

    Calculates the center frequencies and bandwidths for the wavelets and applies them along with
    a heavyside function to the fft of the signal before performing an inverse fft. The center
    frequencies and bandwidths are also stored in the NWB file.

    Parameters
    ----------
    elec_series : ElectricalSeries
        ElectricalSeries to process.
    processing : Processing module
        NWB Processing module to save processed data.
    filters : str (optional)
        Which type of filters to use. Options are
        'rat': center frequencies spanning 2-1200 Hz, constant Q, 54 bands
        'human': center frequencies spanning 4-200 Hz, constant Q, 40 bands
        'changlab': center frequencies spanning 4-200 Hz, variable Q, 40 bands
    hg_only : bool
        If True, only the amplitudes in the high gamma range [70-150 Hz] is computed.
    X_fft_h : ndarray (n_time, n_channels)
        Precomputed product of X_fft and heavyside.
    abs_only : bool
        If True, only the amplitude is stored.
    npad : int
        Length of padding in samples.

    Returns
    -------
    X_wvlt : ndarray, complex
        Complex wavelet coefficients.
    series : list of DecompositionSeries
        List of NWB objects.
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
