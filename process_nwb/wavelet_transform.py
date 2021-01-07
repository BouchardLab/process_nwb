import numpy as np

from .fft import fftfreq, fft, ifft
from .utils import (_npads, _smart_pad, _trim,
                    log_spaced_cfs, const_Q_sds,
                    chang_sds)

from pynwb.misc import DecompositionSeries
from hdmf.data_utils import AbstractDataChunkIterator, DataChunk


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

def get_filterbank(filters, constant_Q, n_time, rate):
    if filters == 'default':
        filters = []
        cfs = log_spaced_cfs(4.0749286538265, 200, 40)
        if constant_Q:
            sds = const_Q_sds(cfs)
        else:
            raise NotImplementedError
        for cf, sd in zip(cfs, sds):
            filters.append(gaussian(n_time, rate, cf, sd))
    elif filters == 'chang':
        filters = []
        cfs = log_spaced_cfs(4.0749286538265, 200, 40)
        sds = chang_sds(cfs)
        for cf, sd in zip(cfs, sds):
            filters.append(gaussian(n_time, rate, cf, sd))
    elif isinstance(filters, list):
        pass # `filters` is already a filter bank
    else:
        raise NotImplementedError

    if not isinstance(filters, list):
        filters = [filters]

    return filters

def wavelet_transform(X, rate, filters='default', X_fft_h=None, npad=None,
                      constant_Q=True):
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
        # Assume X already padded
        n_time = X_fft_h.shape[0]
    freq = fftfreq(n_time, 1. / rate)

    filters = get_filterbank(filters, constant_Q, n_time, rate)
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


class ChBandIterator(AbstractDataChunkIterator):
    def __init__(self, X, rate, filters='default', npad=None, constant_Q=True):
        self.X = X
        self.rate = rate
        self.filters = filters
        self.npad = npad
        self.constant_Q = constant_Q

        # Need to pad X before predicting chunk and filter shape:
        X = self.X[:, 0]
        npads, to_removes, _ = _npads(X, self.npad)
        X = _smart_pad(X, npads)
        self.ntimepts = X.shape[0]
        self.filterbank = get_filterbank(self.filters, self.constant_Q, self.ntimepts, self.rate)

        self.nch = self.X.shape[1]
        self.nbands = len(self.filterbank)
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        ch = self._i // self.nbands
        band = self._i % self.nbands
        self._i += 1

        if ch >= self.X.shape[1]:
            raise StopIteration

        print("Doing channel {} band {}".format(ch, band))

        if band == 0:
            self.X_fft_h = None

        X_ch = self.X[:, [ch]]
        data, self.X_fft_h = wavelet_transform(
            X_ch,
            self.rate,
            filters=[self.filterbank[band]],
            # X_fft_h=self.X_fft_h, # Reusing X_fft_h doesn't work here because wavelet_transform() decides whether to pad x based on whether X_fft_h is passed in. Could be optimized, but this is a small price to pay for the memory gains
            npad=self.npad,
            constant_Q=self.constant_Q
        )
        data = np.squeeze(data)
        return DataChunk(data=np.abs(data), selection=np.s_[:data.shape[0], ch, band])

    next = __next__

    @property
    def dtype(self):
        return self.X.dtype

    @property
    def maxshape(self):
        return (None, self.nch, self.nbands)

    def recommended_chunk_shape(self):
        return (self.ntimepts, 1, 1)

    def recommended_data_shape(self):
        return (self.ntimepts, self.nch, self.nbands)


def store_wavelet_transform(elec_series, processing, npad=None, filters='default',
                            X_fft_h=None, abs_only=True, constant_Q=True, chunked=True):
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
    X = elec_series.data
    rate = elec_series.rate
    if npad is None:
        npad = int(rate)
    if chunked:
        if not abs_only:
            raise NotImplementError("Can't get phase from chunked wavelet transform")
        X_wvlt_abs = ChBandIterator(X, rate, filters=filters, npad=npad, constant_Q=constant_Q)
        elec_series_wvlt_amp = DecompositionSeries('wvlt_amp_' + elec_series.name,
                                                   X_wvlt_abs,
                                                   metric='amplitude',
                                                   source_timeseries=elec_series,
                                                   starting_time=elec_series.starting_time,
                                                   rate=rate,
                                                   description=('Wavlet: ' +
                                                                elec_series.description))
        X_wvlt = None # this function still needs to return something
    else:
        X_wvlt, _ = wavelet_transform(X[:], rate, filters=filters, X_fft_h=X_fft_h,
                                      npad=npad, constant_Q=constant_Q)
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
        if filters == 'default':
            cfs = log_spaced_cfs(4.0749286538265, 200, 40)
            if constant_Q:
                sds = const_Q_sds(cfs)
            else:
                raise NotImplementedError
            for ii, (cf, sd) in enumerate(zip(cfs, sds)):
                iii = '{:02}'.format(ii)
                es.add_band(band_name=iii, band_mean=cf,
                            band_stdev=sd, band_limits=(-1, -1))

        processing.add(es)

    return X_wvlt, series
