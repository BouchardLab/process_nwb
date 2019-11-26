import numpy as np

import pynwb
from pynwb.ecephys import ElectricalSeries


__all__ = ['CAR',
           'subtract_CAR',
           'store_CAR']


def CAR(X, mean_frac=.95, round_func=np.ceil):
    """
    Compute the common average (mean) reference across channels.

    Parameters
    ----------
    X : ndarray, (n_time, n_channels)
        Input timeseries.
    mean_frac : float
        Fraction of the channels to include in the mean. Between 0 and 1.
        mean_frac must include at least one channel.
    round_func : callable
        Function which specifies how to round to the channel number.

    Returns
    -------
    avg : ndarray, (n_time, 1)
       Common average reference.
    """
    n_time, n_channels = X.shape
    if mean_frac == 1.:
        avg = np.nanmean(X, axis=1, keepdims=True)
    else:
        n_exclude = int(round_func(n_channels * (1. - mean_frac) / 2.))
        if 2 * n_exclude >= n_channels:
            raise ValueError
        avg = np.nanmean(np.sort(X, axis=1)[:, n_exclude:n_channels - n_exclude],
                         axis=1, keepdims=True)
    return avg


def subtract_CAR(X, mean_frac=.95, round_func=np.ceil):
    """
    Compute and subtract the common average (mean) reference across channels.

    Parameters
    ----------
    X : ndarray, (n_time, n_channels)
        Input timeseries.
    mean_frac : float
        Fraction of the channels to include in the mean. Between 0 and 1.
        mean_frac must include at least one channel.
    round_func : callable
        Function which specifies how to round to the channel number.

    Returns
    -------
    Xp : ndarray, (n_time, n_channels)
       Common average reference.
    """
    X_CAR = X - CAR(X, mean_frac=mean_frac, round_func=round_func)
    return X_CAR


def store_subtract_CAR(electrical_series, processing, mean_frac=.95, round_func=np.ceil):
    """
    Compute and subtract the common average (mean) reference across channels.

    Parameters
    ----------
    X : ndarray, (n_time, n_channels)
        Input timeseries.
    nwb : NWBFile
        NWBFile to write to.
    mean_frac : float
        Fraction of the channels to include in the mean. Between 0 and 1.
        mean_frac must include at least one channel.
    round_func : callable
        Function which specifies how to round to the channel number.

    Returns
    -------
    Xp : ndarray, (n_time, n_channels)
       Common average reference.
    """
    X = electrical_series.data[:]
    rate = electrical_series.rate

    avg = CAR(X, mean_frac=mean_frac, round_func=round_func)
    X_CAR = X - avg

    electrical_series_CAR = ElectricalSeries('CAR_' + electrical_series.name,
                                             X_CAR,
                                             electrical_series.electrodes,
                                             starting_time=electrical_series.starting_time,
                                             rate=rate,
                                             description=('CARed: ' +
                                                          electrical_series.description))
    CAR = ElectricalSeries('CAR', avg, electrical_series.electrodes,
                           starting_time=electrical_series.starting_time,
                           rate=rate,
                           description=('CAR: ' + electrical_series.description))

    processing.add(electrical_series_CAR)
    processing.add(CAR)
    return X_CAR, electrical_series_CAR
