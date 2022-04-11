import numpy as np
import warnings

from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBHDF5IO
from pynwb.ecephys import ElectricalSeries

from process_nwb.common_referencing import subtract_CAR, CAR
from process_nwb.linenoise_notch import apply_linenoise_notch
from process_nwb.resample import resample, store_resample, _scaling as scaling
from process_nwb.utils import dtype
from process_nwb.wavelet_transform import store_wavelet_transform


def preprocess_block(nwb_path,
                     acq_name='ECoG',
                     initial_resample_rate=3200.,
                     final_resample_rate=400.,
                     filters='rat',
                     hg_only=True,
                     all_steps=False,
                     logger=None):
    """This is the default preprocessing pipeline.

    Perform the following steps:
    1) Resample to initial_resample_rate,
    2) Remove 60Hz noise and remove the CAR, and
    3) Perform and store a wavelet decomposition.
    4) Optionally resample the wavelet amplitudes.

    Parameters
    -------
    nwb_path : str or pathlike
        Path to the .nwb file. This file will be modified as a result of this function.
    acq_name : str
        Name of the acquisition, either 'ECoG' or 'Poly'.
    initial_resample_rate : float
        Frequency (in Hz) to resample to before performing wavelet transform.
    final_resample_rate : float
        Frequency (in Hz) to resample to after calculating wavelet amplitudes.
    filters : str
        Type of filter bank to use for wavelets. Choose from ['rat', 'human', 'changlab'].
    hg_only : bool
        Whether to store high gamma bands only. If False, use all filters.
    all_steps : bool
        Whether to store intermediate data between preprocessing steps.
    logger : logger
        Optional logger passed from upstream.

    Returns
    -------
    Returns nothing, but changes the NWB file at nwb_path.
    A ProcessingModule with name 'preprocessing' will be added to the NWB.
    """
    with NWBHDF5IO(nwb_path, 'a') as io:
        if logger is not None:
            logger.info('==================================')
            logger.info(f'Running preprocessing for {nwb_path}.')

        nwbfile = io.read()
        try:
            electrical_series = nwbfile.acquisition[acq_name]
        except KeyError:
            # in case NWB file is in a legacy format
            electrical_series = nwbfile.acquisition['Raw'][acq_name]

        nwbfile.create_processing_module(name='preprocessing',
                                         description='Preprocessing.')
        if all_steps:
            if logger is not None:
                logger.info('Resampling...')
            _, electrical_series_ds = store_resample(electrical_series,
                                                     nwbfile.processing['preprocessing'],
                                                     initial_resample_rate)
            del _

            if logger is not None:
                logger.info('Filtering and re-referencing...')
            _, electrical_series_CAR = store_linenoise_notch_CAR(electrical_series_ds,
                                                                 nwbfile.processing['preprocessing'])
            del _
            series = electrical_series_CAR
        else:
            rate = electrical_series.rate
            if logger is not None:
                logger.info('Resampling...')
            ts = resample(electrical_series.data[:] * scaling,
                          initial_resample_rate, rate)
            if logger is not None:
                logger.info('Filtering and re-referencing...')
            ts = apply_linenoise_notch(ts, initial_resample_rate)
            ts = subtract_CAR(ts)
            electrical_series_CAR = ElectricalSeries('CAR_ln_downsampled_' + electrical_series.name,
                                                     ts,
                                                     electrical_series.electrodes,
                                                     starting_time=electrical_series.starting_time,
                                                     rate=initial_resample_rate)
            series = electrical_series

        if logger is not None:
            logger.info('Running wavelet transform...')
        _, electrical_series_wvlt = store_wavelet_transform(electrical_series_CAR,
                                                            nwbfile.processing['preprocessing'],
                                                            filters=filters,
                                                            hg_only=hg_only,
                                                            post_resample_rate=final_resample_rate,
                                                            source_series=series)

        io.write(nwbfile)
        if logger is not None:
            logger.info(f'Preprocessing added to {nwb_path}.')


def store_linenoise_notch_CAR(elec_series, processing, mean_frac=.95, round_func=np.ceil,
                              precision='single'):
    """Apply a notch filter at 60 Hz and its harmonics, calculate and remove the common average
    reference (CAR), and finally store the signal and the CAR.

    Parameters
    ----------
    elec_series : ElectricalSeries
        ElectricalSeries to process.
    processing : Processing module
        NWB Processing module to save processed data.
    mean_frac : float
        Fraction of the data to be taken in the mean. 0. < mean_frac <= 1.
    round_func : callable
        Function for rounding the fraction of channels.
    precision : str
        Either `single` for float32/complex64 or `double` for float/complex.

    Returns
    -------
    X_CAR_ln : ndarray, (n_time, n_channels)
        Data with line noise and CAR removed.
    elec_series_CAR_ln : ElectricalSeries
        ElectricalSeries that holds X_CAR_ln.
    """
    rate = elec_series.rate
    X = elec_series.data[:]
    X_dtype = dtype(X, precision)
    X = X.astype(X_dtype, copy=False)

    X_ln = apply_linenoise_notch(X, rate, precision=precision)
    avg = CAR(X_ln, mean_frac=mean_frac, round_func=round_func, precision=precision)
    X_CAR_ln = X_ln - avg

    elec_series_CAR_ln = ElectricalSeries('CAR_ln_' + elec_series.name,
                                          H5DataIO(X_CAR_ln,
                                                   compression=True,
                                                   shuffle=True,
                                                   fletcher32=True),
                                          elec_series.electrodes,
                                          starting_time=elec_series.starting_time,
                                          rate=rate,
                                          description=('CAR_lned: ' +
                                                       elec_series.description))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        CAR_series = ElectricalSeries('CAR_of_' + elec_series.name,
                                      H5DataIO(avg,
                                               compression=True,
                                               shuffle=True,
                                               fletcher32=True),
                                      elec_series.electrodes,
                                      starting_time=elec_series.starting_time,
                                      rate=rate,
                                      description=('CAR: ' + elec_series.description))

    processing.add(elec_series_CAR_ln)
    processing.add(CAR_series)
    return X_CAR_ln, elec_series_CAR_ln
