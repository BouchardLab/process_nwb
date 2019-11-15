import numpy as np

from .fft import fftfreq, fft, ifft
from .utils import (_npads, _smart_pad, _trim,
                    log_spaced_cfs, const_Q_sds)

from pynwb.misc import DecompositionSeries


__all__ = ['gaussian',
           'hamming',
           'wavelet_transform',
           'store_wavelet_transform']


def gaussian(n_time, rate, center, sd):
    freq = fftfreq(n_time, 1./rate)

    k = np.exp((-(np.abs(freq) - center)**2)/(2 * (sd**2)))
    k /= np.linalg.norm(k)

    return k


def hamming(n_time, rate, min_freq, max_freq):
    freq = fftfreq(n_time, 1./rate)

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


def wavelet_transform(X, rate, filters='default', X_fft_h=None,
                      npad=100):
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
    if X_fft_h is None:
        npads, to_removes, _ = _npads(X, npad)
        X = _smart_pad(X, npads)
        n_time = X.shape[0]
    else:
        n_time = X_fft_h.shape[0]
    freq = fftfreq(n_time, 1. / rate)

    if filters == 'default':
        filters = []
        cfs = log_spaced_cfs(4.0749286538265, 200, 40)
        sds = const_Q_sds(cfs)
        for cf, sd in zip(cfs, sds):
            filters.append(gaussian(n_time, rate, cf, sd))

    if not isinstance(filters, list):
        filters = [filters]

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

    return Xh, X_fft_h


def store_wavelet_transform(nwbfile, series_name, filters='default',
                            X_fft_h=None, npad=100, abs_only=True):
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
        Product of X_ff and heavyside.
    """
    electrical_series = nwbfile.acquisition[series_name]
    X = electrical_series.data[:]
    rate = electrical_series.rate
    X_wvlt, _ = wavelet_transform(X, rate, filters=filters, X_fft_h=X_fft_h,
                                  npad=npad)
    electrical_series_wvlt_amp = DecompositionSeries('wvlt_amp' + electrical_series.name,
                                                     abs(X_wvlt),
                                                     metric='amplitude',
                                                     source_timeseries=electrical_series,
                                                     starting_time=electrical_series.starting_time,
                                                     rate=rate,
                                                     description=('Wavlet: ' +
                                                                  electrical_series.description))
    series = [electrical_series_wvlt_amp]
    if not abs_only:
        electrical_series_wvlt_phase = DecompositionSeries('wvlt_phase' + electrical_series.name,
                                                           np.angle(X_wvlt),
                                                           metric='phase',
                                                           source_timeseries=electrical_series,
                                                           starting_time=electrical_series.starting_time,
                                                           rate=rate,
                                                           description=('Wavlet: ' +
                                                                        electrical_series.description))
        series.append(electrical_series_wvlt_phase)

    for es in series:
        if filters == 'default':
            cfs = log_spaced_cfs(4.0749286538265, 200, 40)
            sds = const_Q_sds(cfs)
            for ii , (cf, sd) in enumerate(zip(cfs, sds)):
                es.add_band(band_name=str(ii), band_mean=cf,
                            band_stdev=sd, band_limits=(-1, -1))

        nwbfile.add_acquisition(es)
    if len(series) == 1:
        series = series[0]
    return X_wvlt, series
