import numpy as np

from pynwb.ecephys import ElectricalSeries

from .common_referencing import CAR
from .linenoise_notch import apply_linenoise_notch

def store_linenoise_notch_CAR(electrical_series, processing, mean_frac=.95, round_func=np.ceil):
    rate = electrical_series.rate
    X = electrical_series.data[:]

    X_ln = apply_linenoise_notch(X, rate)
    avg = CAR(X_ln, mean_frac=mean_frac, round_func=round_func)
    X_CAR_ln = X_ln - avg

    electrical_series_CAR_ln = ElectricalSeries('CAR_ln_' + electrical_series.name,
                                                X_CAR_ln,
                                                electrical_series.electrodes,
                                                starting_time=electrical_series.starting_time,
                                                rate=rate,
                                                description=('CAR_lned: ' +
                                                             electrical_series.description))
    CAR_series = ElectricalSeries('CAR', avg, electrical_series.electrodes,
                                  starting_time=electrical_series.starting_time,
                                  rate=rate,
                                  description=('CAR: ' + electrical_series.description))

    processing.add(electrical_series_CAR_ln)
    processing.add(CAR_series)
    return X_CAR_ln, electrical_series_CAR_ln
