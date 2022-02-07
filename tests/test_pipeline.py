import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest, os

from pynwb.ecephys import ElectricalSeries
from pynwb import NWBHDF5IO

from process_nwb.utils import generate_synthetic_data, generate_nwbfile
from process_nwb.resample import store_resample, resample
from process_nwb.wavelet_transform import store_wavelet_transform, wavelet_transform
from process_nwb import store_linenoise_notch_CAR
from process_nwb.linenoise_notch import apply_linenoise_notch
from process_nwb.common_referencing import subtract_CAR


@pytest.fixture
def neural_data():
    num_channels = 64
    duration = 10.  # seconds
    sample_rate = 10000.  # Hz
    neural_data = generate_synthetic_data(duration, num_channels, sample_rate)
    return neural_data


@pytest.mark.parametrize("post_resample_rate", [None, 200.])
def test_pipeline(neural_data, post_resample_rate):
    """Test that the NWB pipeline gives equal results to running preprocessing functions
    by hand.
    """
    sample_rate = 10000.  # Hz
    new_sample_rate = 500.  # hz

    nwbfile, device, electrode_group, electrodes = generate_nwbfile()
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
    # This part checks amplitude and phase, does not post-resample
    tf_data_nwb, tf_series = store_wavelet_transform(car_series,
                                                     nwbfile.processing['preprocessing'],
                                                     filters='rat',
                                                     hg_only=True,
                                                     chunked=False,
                                                     abs_only=False)
    tf_data, _, _, _ = wavelet_transform(car_data, new_sample_rate,
                                         filters='rat', hg_only=True)
    assert_array_equal(tf_data_nwb, tf_data)
    assert_array_equal(tf_series[0].data[:], abs(tf_data))
    assert_array_equal(tf_series[1].data[:], np.angle(tf_data))

    # This part only checks amplitude, does post-resample
    nwbfile, device, electrode_group, electrodes = generate_nwbfile()
    ECoG_ts = ElectricalSeries('ECoG_data',
                               neural_data,
                               electrodes,
                               starting_time=0.,
                               rate=sample_rate)
    nwbfile.add_acquisition(ECoG_ts)

    electrical_series = nwbfile.acquisition['ECoG_data']
    nwbfile.create_processing_module(name='preprocessing',
                                     description='Preprocessing.')
    tf_data_nwb, tf_series = store_wavelet_transform(car_series,
                                                     nwbfile.processing['preprocessing'],
                                                     filters='rat',
                                                     hg_only=True,
                                                     chunked=False,
                                                     abs_only=True,
                                                     post_resample_rate=post_resample_rate)
    if post_resample_rate is not None:
        tf_data = resample(abs(tf_data), post_resample_rate, new_sample_rate)
        assert_array_equal(tf_data_nwb, tf_data)
        assert tf_series[0].rate == post_resample_rate
    else:
        assert tf_series[0].rate == new_sample_rate


@pytest.mark.parametrize("post_resample_rate", [None, 200.])
def test_chunked_pipeline(tmpdir, neural_data, post_resample_rate):
    """Test that the NWB runs with the chunked versions.
    """
    sample_rate = 10000.  # Hz
    new_sample_rate = 500.  # Hz

    arrays = []
    for chunked, name in zip([True, False], ['chunked.nwb', 'all.nwb']):
        nwbfile, device, electrode_group, electrodes = generate_nwbfile()
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
        _, rs_series = store_resample(electrical_series,
                                      nwbfile.processing['preprocessing'],
                                      new_sample_rate)

        # Linenoise and CAR
        _, car_series = store_linenoise_notch_CAR(rs_series,
                                                  nwbfile.processing['preprocessing'])

        # Wavelet transform
        store_wavelet_transform(car_series,
                                nwbfile.processing['preprocessing'],
                                filters='rat',
                                chunked=chunked,
                                post_resample_rate=post_resample_rate,
                                hg_only=True)
        with NWBHDF5IO(os.path.join(tmpdir, name), mode='w') as io:
            io.write(nwbfile)
        with NWBHDF5IO(os.path.join(tmpdir, name), mode='r') as io:
            nwbfile = io.read()
            series = nwbfile.processing['preprocessing']['wvlt_amp_CAR_ln_downsampled_ECoG_data']
            final_rate = series.rate
            if post_resample_rate is None:
                assert final_rate == new_sample_rate
            else:
                assert final_rate == post_resample_rate
            arrays.append(series.data[:])
    assert np.dtype(arrays[0].dtype) == np.dtype(arrays[1].dtype)
    assert_allclose(*arrays, rtol=0.01)
