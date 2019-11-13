from __future__ import division
import numpy as np


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
        n_exclude = int(round_func(n_channels * (1. - mean_frac)) / 2.)
        if n_exclude == n_channels:
            raise ValueError
        avg = np.nanmean(np.sort(X, axis=1)[:, n_exclude:-n_exclude],
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
    Xp = X - CAR(X, mean_frac=mean_frac, round_func=round_func)
    return Xp


def store_CAR(X, nwb, mean_frac=.95, round_func=np.ceil):
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
    raise NotImplementedError
