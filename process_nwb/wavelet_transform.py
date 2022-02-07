import numpy as np

from process_nwb.resample import resample
from scipy.fft import fftfreq, fft, ifft

from pynwb.misc import DecompositionSeries
from hdmf.data_utils import AbstractDataChunkIterator, DataChunk

from process_nwb.utils import (_npads, _smart_pad, _trim,
                               log_spaced_cfs, const_Q_sds,
                               chang_sds, dtype)


def gaussian(n_time, rate, center, sd, precision='single'):
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
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.
    """
    freq = fftfreq(n_time, 1. / rate)
    X_dtype = dtype(freq, precision)

    k = np.exp((-(np.abs(freq) - center) ** 2) / (2 * (sd ** 2)))
    k /= np.linalg.norm(k)

    return k.astype(X_dtype, copy=False)


def hamming(n_time, rate, min_freq, max_freq, precision='single'):
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
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.
    """
    freq = fftfreq(n_time, 1. / rate)
    X_dtype = dtype(freq, precision)

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

    return k.astype(X_dtype, copy=False)


def get_filterbank(filters, n_time, rate, hg_only, precision='single'):
    """Get the filterbank and parameters.

    Parameters
    ----------
    filters : str or list
        Which type of filters to use. Options are
        'rat': center frequencies spanning 2-1200 Hz, constant Q, 54 bands
        'human': center frequencies spanning 4-200 Hz, constant Q, 40 bands
        'changlab': center frequencies spanning 4-200 Hz, variable Q, 40 bands
        Note - calculating center frequencies above rate/2 raises a ValueError
        If filters is a list, it is assumed to already be correctly formatted.
    n_time : int
        Input data time dimension.
    rate : float
        Number of samples per second.
    hg_only : bool
        If True, only the amplitudes in the high gamma range [70-150 Hz] is computed.
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.

    Returns
    -------
    filters : list of ndarrays
        List of filters to apply.
    cfs : ndarray
        Center frequencies used.
    sds : ndarray
        Bandwidths used.
    """
    if isinstance(filters, list):
        return filters, None, None

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
    if cfs.max() * 2. > np.nextafter(rate, np.inf):  # Allow floating point tolerance
        string = ('Unable to compute wavelet transform above Nyquist rate ({} Hz).' +
                  ' Increase your rate ({} Hz) to at least twice your desired maximum' +
                  'frequency of interest.')
        raise ValueError(string.format(cfs.max() * 2., np.nextafter(rate, np.inf)))

    # Calculate bandwidths
    if filters in ['rat', 'human']:
        sds = const_Q_sds(cfs)
    elif filters == 'changlab':
        sds = chang_sds(cfs)
    else:
        raise NotImplementedError

    filters = []
    for cf, sd in zip(cfs, sds):
        filters.append(gaussian(n_time, rate, cf, sd, precision=precision))

    return filters, cfs, sds


class ChannelBandIterator(AbstractDataChunkIterator):
    """Class for iterative write over channels and bands.

    Parameters
    ----------
    X : ndarray (n_time, n_channels)
        Data array.
    filters : str (optional)
        Which type of filters to use. Options are
        'rat': center frequencies spanning 2-1200 Hz, constant Q, 54 bands
        'human': center frequencies spanning 4-200 Hz, constant Q, 40 bands
        'changlab': center frequencies spanning 4-200 Hz, variable Q, 40 bands
    npad : int
        Padding to add to beginning and end of timeseries. Default 0.
    hg_only : bool
        If True, only the amplitudes in the high gamma range [70-150 Hz] is computed.
    post_resample_rate : float
        If not `None`, resample the computed wavelet amplitudes to this rate.
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.
    """

    def __init__(self, X, rate, filters='rat', npad=None, hg_only=True, post_resample_rate=None,
                 precision='single'):
        self.X_dtype = dtype(X, precision)
        X = X.astype(self.X_dtype, copy=False)
        self.X = X
        self.rate = rate
        self.npad = npad
        self.post_resample_rate = post_resample_rate
        self.precision = precision

        # Need to pad X before predicting chunk and filter shape:
        self.npads, self.to_removes, _ = _npads(X, npad)
        self.wavelet_time = X.shape[0] + 2 * npad
        self.filterbank, self.cfs, self.sds = get_filterbank(filters, self.wavelet_time, self.rate,
                                                             hg_only, precision=self.precision)
        self.resample_time = self.X.shape[0]
        if post_resample_rate is not None:
            self.resample_time = int(np.ceil(self.wavelet_time * post_resample_rate / rate))

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

        if band == 0:
            self.X_fft_h = None

        X_ch = self.X[:, [ch]]
        X_ch = _smart_pad(X_ch, self.npads)
        data, self.X_fft_h, cfs, sds = wavelet_transform(
            X_ch,
            self.rate,
            filters=[self.filterbank[band]],
            X_fft_h=self.X_fft_h,
            to_removes=self.to_removes,
            precision=self.precision
        )
        data = np.abs(data)
        if self.post_resample_rate is not None:
            data = resample(data, self.post_resample_rate, self.rate, precision=self.precision)
        data = np.squeeze(data)
        return DataChunk(data=data, selection=np.s_[:data.shape[0], ch, band])

    next = __next__

    @property
    def dtype(self):
        return self.X.dtype

    @property
    def maxshape(self):
        return (None, self.nch, self.nbands)

    def recommended_chunk_shape(self):
        return (self.resample_time, 1, 1)

    def recommended_data_shape(self):
        return (self.resample_time, self.nch, self.nbands)


def wavelet_transform(X, rate, filters='rat', hg_only=True, X_fft_h=None, npad=0, to_removes=None,
                      precision='single'):
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
        Length of padding in samples. Default 0.
    to_removes : int
        Number of samples to remove at the beginning and end of the timeseries. Default None.
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.

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
    if X_fft_h is None:
        X_dtype = dtype(X, precision)
        X = X.astype(X_dtype, copy=False)
        npads, to_removes, _ = _npads(X, npad)
        X = _smart_pad(X, npads)
        n_time = X.shape[0]
    else:
        n_time = X_fft_h.shape[0]
        X_fft_h = X_fft_h.astype(dtype(X_fft_h, precision), copy=False)
    freq = fftfreq(n_time, 1. / rate)

    filters, cfs, sds = get_filterbank(filters, n_time, rate, hg_only, precision=precision)

    Xh = np.zeros(X.shape + (len(filters),), dtype=dtype(complex(1.), precision=precision))
    if X_fft_h is None:
        # Heavyside filter with 0 DC
        h = np.zeros(len(freq))
        h[freq > 0] = 2.
        h = h[:, np.newaxis]
        X_fft_h = fft(X, axis=0, workers=-1) * h

    for ii, f in enumerate(filters):
        if f is None:
            Xh[..., ii] = ifft(X_fft_h, axis=0, workers=-1)
        else:
            f = f / np.linalg.norm(f)
            Xh[..., ii] = ifft(X_fft_h * f[:, np.newaxis], axis=0, workers=-1)

    Xh = _trim(Xh, to_removes)

    return Xh, X_fft_h, cfs, sds


def store_wavelet_transform(elec_series, processing, filters='rat', hg_only=True, abs_only=True,
                            npad=0, post_resample_rate=None, chunked=True, precision='single'):
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
    abs_only : bool
        If True, only the amplitude is stored.
    npad : int
        Padding to add to beginning and end of timeseries. Default 0.
    post_resample_rate : float
        If not `None`, resample the computed wavelet amplitudes to this rate.
    chunked : bool
        If True, calculate wavelet transform one channel and band at a time and store iteratively into nwb. Default True
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex. Default single.

    Returns
    -------
    X_wvlt : ndarray, complex
        Complex wavelet coefficients.
    series : list of DecompositionSeries
        List of NWB objects.
    """
    X = elec_series.data[:]
    X_dtype = dtype(X, precision)
    X = X.astype(X_dtype, copy=False)
    rate = elec_series.rate

    final_rate = rate
    if post_resample_rate is not None:
        final_rate = post_resample_rate

    if chunked:
        if not abs_only:
            raise NotImplementedError("Phase is not implemented for chunked wavelet transform.")
        X_wvlt_abs = ChannelBandIterator(X, rate, filters=filters, npad=npad, hg_only=hg_only,
                                         post_resample_rate=post_resample_rate, precision=precision)
        cfs = X_wvlt_abs.cfs
        sds = X_wvlt_abs.sds
        elec_series_wvlt_amp = DecompositionSeries('wvlt_amp_' + elec_series.name,
                                                   X_wvlt_abs,
                                                   metric='amplitude',
                                                   source_timeseries=elec_series,
                                                   starting_time=elec_series.starting_time,
                                                   rate=final_rate,
                                                   description=('Wavlet: ' +
                                                                elec_series.description))
        series = [elec_series_wvlt_amp]
        X_wvlt = None
    else:
        X_wvlt, _, cfs, sds = wavelet_transform(X, rate, filters=filters, hg_only=hg_only,
                                                npad=npad, precision=precision)
        amplitude = abs(X_wvlt)
        if post_resample_rate is not None:
            amplitude = resample(amplitude, post_resample_rate, rate, precision=precision)
            X_wvlt = amplitude
            rate = post_resample_rate
        elec_series_wvlt_amp = DecompositionSeries('wvlt_amp_' + elec_series.name,
                                                   amplitude,
                                                   metric='amplitude',
                                                   source_timeseries=elec_series,
                                                   starting_time=elec_series.starting_time,
                                                   rate=final_rate,
                                                   description=('Wavlet: ' +
                                                                elec_series.description))
        series = [elec_series_wvlt_amp]
        if not abs_only:
            if post_resample_rate is not None:
                raise ValueError('Wavelet phase should not be resampled.')
            elec_series_wvlt_phase = DecompositionSeries('wvlt_phase_' + elec_series.name,
                                                         np.angle(X_wvlt),
                                                         metric='phase',
                                                         source_timeseries=elec_series,
                                                         starting_time=elec_series.starting_time,
                                                         rate=final_rate,
                                                         description=('Wavlet: ' +
                                                                      elec_series.description))
            series.append(elec_series_wvlt_phase)

    for es in series:
        for ii, (cf, sd) in enumerate(zip(cfs, sds)):
            es.add_band(band_name=str(ii), band_mean=cf,
                        band_stdev=sd, band_limits=(-1, -1))

        processing.add(es)
    return X_wvlt, series
