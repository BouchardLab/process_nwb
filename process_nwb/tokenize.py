import numpy as np


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


def load_anatomy(nwb):
    electrode_labels = nwb.ec_electrodes['location'].data[:]

    return electrode_labels


def load_bad_electrodes(nwb):
    """
    Load bad electrodes.

    Parameters
    ----------
    nwb : NWBFile
        Open NWB file for the block.

    Returns
    -------
    bad_electrodes : ndarray
        Python (0-based) indices of bad electrodes.
    """

    bad_electrodes = nwb.ec_electrodes['bad'].data[:]

    return bad_electrodes


def load_bad_times(nwb):
    """
    Load bad time segments.

    Parameters
    ----------
    block_path : str
        Path to block to load bad electrodes from.

    Returns
    -------
    bad_times : ndarray, (n_windows, 2)
        Pairs of start and stop times for bad segments.
    """
    times = nwb.invalid_times
    bad_times = None
    if times is not None:
        start = nwb.invalid_times['start_time'].data[:]
        stop = nwb.invalid_times['stop_time'].data[:]
        bad_times = np.stack([start, stop], axis=1)
    return bad_times


def get_speak_event(nwb, align_pos):
    transcript = nwb.trials.to_dataframe()
    transcript = transcript.loc[transcript['speak']]
    if align_pos == 0:
        event_times = transcript['start_time']
    elif align_pos == 1:
        event_times = transcript['cv_transition_time']
    elif align_pos == 2:
        event_times = transcript['stop_time']
    else:
        raise ValueError

    # Make sure we don't have the old format
    if event_times.max() < 10:
        raise ValueError
    event_labels = transcript['condition']
    event_labels = np.array([standardize_token(el) for el in event_labels])
    return event_times, event_labels


def standardize_token(token):
    """
    Standardizations to make to tokens.

    Parameters
    ----------
    token : str
        Token to be standarized.

    Returns
    -------
    token : str
        Standardized token.
    """
    token = token.lower()
    token = token.replace('uu', 'oo')
    token = token.replace('ue', 'oo')
    token = token.replace('gh', 'g')
    token = token.replace('who', 'hoo')
    token = token.replace('aw', 'aa')
    if len(token) == 2:
        token = token.replace('ha', 'haa')
    elif len(token) == 3:
        token = token.replace('she', 'shee')

    return token


def extract_windows(data, sampling_freq, event_times, align_window,
                    bad_times=None):
    """
    Extracts windows of aligned data. Assumes constant sampling frequency.
    Assumes last two dimensions of data are electrodes and time.

    Parameters
    ----------
    data : ndarray (n_time, ...)
        Timeseries data.
    sampling_freq : float
        Sampling frequency of data.
    event_time : list of floats
        Time (in seconds) of events.
    align_window : ndarray
        Window around event alignment.
    bad_times : ndarray (n_windows, 2)
        Start and stop points of bad times segments.

    Returns
    -------
    D : ndarray (events, n_time, n_elects, n_bands)
        Event data aligned to event times.
    """
    dtype = data.dtype

    if align_window is None:
        align_window = np.array([-1., 1.])
    else:
        align_window = np.array(align_window)
        assert align_window[0] <= 0., 'align window start'
        assert align_window[1] >= 0., 'align window end'
        assert align_window[0] < align_window[1], 'align window order'

    if bad_times is None:
        bad_times = np.array([])
    else:
        bad_times = np.array(bad_times)

    n_data = len(event_times)
    window_length = int(np.ceil(np.diff(align_window) * sampling_freq))
    other_shape = data.shape[1:]
    window_start = int(np.floor(align_window[0] * sampling_freq))
    D = np.full((n_data, window_length,) + other_shape, np.nan, dtype)

    def time_idx(time):
        return int(np.around(time * sampling_freq))

    for ii, etime in enumerate(event_times):
        sl = slice(time_idx(etime) + window_start,
                   time_idx(etime) + window_start + window_length)
        event_data = data[sl]
        if event_data.shape == D.shape[1:]:
            D[ii] = event_data
        else:
            raise ValueError('Event shape mismatch ' +
                             '{}, {}'.format(event_data.shape, D.shape))

    if bad_times.size:
        bad_trials = []
        for ii, etime in enumerate(event_times):
            if np.any(is_overlap(align_window + etime,
                                 bad_times)):
                bad_trials.append(ii)
        if len(bad_trials) > 0:
            D[bad_trials] = np.nan

    return D
