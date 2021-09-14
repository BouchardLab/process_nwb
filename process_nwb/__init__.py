import numpy as np

from pynwb.ecephys import ElectricalSeries

from process_nwb.common_referencing import CAR
from process_nwb.linenoise_notch import apply_linenoise_notch
from process_nwb.utils import dtype


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
                                          X_CAR_ln,
                                          elec_series.electrodes,
                                          starting_time=elec_series.starting_time,
                                          rate=rate,
                                          description=('CAR_lned: ' +
                                                       elec_series.description))
    CAR_series = ElectricalSeries('CAR', avg, elec_series.electrodes,
                                  starting_time=elec_series.starting_time,
                                  rate=rate,
                                  description=('CAR: ' + elec_series.description))

    processing.add(elec_series_CAR_ln)
    processing.add(CAR_series)
    return X_CAR_ln, elec_series_CAR_ln
