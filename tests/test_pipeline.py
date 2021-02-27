import numpy as np
from numpy.testing import assert_array_equal

from pynwb.ecephys import ElectricalSeries

from process_nwb.utils import generate_synthetic_data, generate_nwbfile
from process_nwb.resample import store_resample, resample
from process_nwb.wavelet_transform import store_wavelet_transform, wavelet_transform
from process_nwb import store_linenoise_notch_CAR
from process_nwb.linenoise_notch import apply_linenoise_notch
from process_nwb.common_referencing import subtract_CAR


def test_pipeline():
    """Test that the NWB pipeline gives equal results to running preprocessing functions
    by hand.
    """
    num_channels = 64
    duration = 10.  # seconds
    sample_rate = 10000.  # Hz
    new_sample_rate = 500.  # hz

    nwbfile, device, electrode_group, electrodes = generate_nwbfile()
    neural_data = generate_synthetic_data(duration, num_channels, sample_rate)
    ECoG_ts = ElectricalSeries('ECoG_data',
                               neural_data,
                               electrodes,
                               starting_time=0.,
                               rate=sample_rate)
    nwbfile.add_acquisition(ECoG_ts)

    electrical_series = nwbfile.acquisition['ECoG_data']
    nwbfile.create_processing_module(name='preprocessing',
                                     description='Preprocessing.')

    # Resample
    rs_data_nwb, rs_series = store_resample(electrical_series,
                                            nwbfile.processing['preprocessing'],
                                            new_sample_rate)
    rs_data = resample(neural_data * 1e6, new_sample_rate, sample_rate)
    assert_array_equal(rs_data_nwb, rs_data)
    assert_array_equal(rs_series.data[:], rs_data)

    # Linenoise and CAR
    car_data_nwb, car_series = store_linenoise_notch_CAR(rs_series,
                                                         nwbfile.processing['preprocessing'])
    nth_data = apply_linenoise_notch(rs_data, new_sample_rate)
    car_data = subtract_CAR(nth_data)
    assert_array_equal(car_data_nwb, car_data)
    assert_array_equal(car_series.data[:], car_data)

    # Wavelet transform
    tf_data_nwb, tf_series = store_wavelet_transform(car_series,
                                                     nwbfile.processing['preprocessing'],
                                                     filters='rat',
                                                     hg_only=True,
                                                     abs_only=False)
    tf_data, _, _, _ = wavelet_transform(car_data, new_sample_rate,
                                         filters='rat', hg_only=True)
    assert_array_equal(tf_data_nwb, tf_data)
    assert_array_equal(tf_series[0].data[:], abs(tf_data))
    assert_array_equal(tf_series[1].data[:], np.angle(tf_data))
