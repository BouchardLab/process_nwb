import numpy as np


__all__ = ['nans',
           'is_overlap',
           'is_in']


def log_spaced_cfs(fmin, fmax, nbin):
    """
    Center frequencies that are uniform in log space
    """
    noct = np.ceil(np.log2(fmax / fmin))
    return fmin * 2 ** (np.arange(noct * nbin) / nbin)


def const_Q_sds(cfs, Q=8):
    return cfs/Q


def nans(shape, dtype=float):
    """
    Create np.array of nans

    :param shape: tuple, dimensions of array
    :param dtype:
    :return:
    """
    a = np.full(shape, np.nan, dtype)
    return a

def is_overlap(time_window, time_windows_array):
    """
    Does time_window overlap with the time windows in times_window_array.
    Used for bad time segments.

    Parameters
    ----------
    time_window : ndarray (2,)
       Single window of time to compare.
    time_windows_array : ndarray (n, 2)
       Windows of time to compare against.

    Returns
    -------
    Boolean overlap comparing time_window to each window in time_windows_array.

    """
    def overlap(tw1, tw2):
        return not ((tw1[1] < tw2[0]) | (tw1[0] > tw2[1]))

    return [overlap(time_window,this_time_window) for this_time_window in time_windows_array]

def is_in(tt, tbounds):
    """
    util: Is time inside time window(s)?

    :param tt:      1 x n np.array        time counter
    :param tbounds: k, 2  np.array   time windows

    :return:        1 x n bool          logical indicating if time is in any of the windows
    """

    tbounds = np.array(tbounds)

    tf = np.zeros(tt.shape, dtype=bool)

    if tbounds.ndim is 1:
        tf = (tt > tbounds[0]) & (tt < tbounds[1])
    else:
        for i in range(tbounds.shape[0]):
            tf = (tf | (tt > tbounds[i,0]) & (tt < tbounds[i,1]))
    return tf
