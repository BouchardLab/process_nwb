import numpy as np

from pynwb.ecephys import ElectricalSeries

from .common_referencing import store_subtract_CAR
from .linenoise_notch import apply_linenoise_notch

def store_linenoise_notch_store_CAR(nwbfile, series_name, mean_frac=.95, round_func=np.ceil):
    electrical_series = nwbfile.acquisition[series_name]
    rate = electrical_series.rate
    X_CAR, electrical_series_CAR = store_subtract_CAR(nwbfile, series_name, mean_frac=mean_frac, round_func=round_func)
    X_CAR_ln = apply_linenoise_notch(X_CAR, rate)

    electrical_series_CAR_ln = ElectricalSeries('CAR_ln_' + electrical_series.name,
                                                X_CAR_ln,
                                                electrical_series.electrodes,
                                                starting_time=electrical_series.starting_time,
                                                rate=rate,
                                                description=('CAR and linenoise: ' +
                                                             electrical_series.description))
    nwbfile.add_acquisition(electrical_series_CAR_ln)
    return X_CAR_ln, electrical_series_CAR_ln
