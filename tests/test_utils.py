import numpy as np

from process_nwb.utils import generate_synthetic_data


def test_synthetic_data():
    """Test that synthetic data generation works.
    """
    time = 100
    nch = 4
    rate = 400

    X0 = generate_synthetic_data(time, nch, rate)
    assert X0.shape == (int(time * rate), nch)

    X1 = generate_synthetic_data(time, nch, rate, high_gamma=False)
    assert not np.array_equal(X0, X1)

    X1 = generate_synthetic_data(time, nch, rate, linenoise=False)
    assert not np.array_equal(X0, X1)

    X1 = generate_synthetic_data(time, nch, rate, seed=1)
    assert not np.array_equal(X0, X1)
