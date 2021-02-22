import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.distance import cosine

from process_nwb.resample import resample


def test_resample_shape():
    X = np.random.randn(2000, 32)

    Xp = resample(X, 100, 200)
    assert Xp.shape == (1000, 32)


def test_resample_ones():
    chs = [2, 32, 100]
    ts = [999, 1000, 1001, 5077]
    rate = 200
    fracs = [.5, 1, 1.5, 2]
    for ch in chs:
        for t in ts:
            for fr in fracs:
                X = np.ones((t, ch))
                Xp = resample(X, rate * fr, rate)
                assert_allclose(Xp, 1., atol=1e-3)


def test_resample_low_freqs():
    """Resampling should not impact low frequencies.
    """
    dt = 40.  # seconds
    rate = 400.  # Hz
    t = np.linspace(0, dt, int(dt * rate))
    t = np.tile(t[:, np.newaxis], (1, 5))
    freqs = np.linspace(1, 5.33, 20)

    X = np.zeros_like(t)
    for f in freqs:
        X += np.sin(2 * np.pi * f * t)

    new_rate = 211.  # Hz
    t = np.linspace(0, dt, int(dt * new_rate))
    t = np.tile(t[:, np.newaxis], (1, 5))
    X_new_rate = np.zeros_like(t)
    for f in freqs:
        X_new_rate += np.sin(2 * np.pi * f * t)

    Xds = resample(X, new_rate, rate)
    assert_allclose(cosine(Xds.ravel(), X_new_rate.ravel()), 0., atol=1e-3)
    assert_allclose(X.mean(), Xds.mean(), atol=1e-3)
    assert_allclose(X.std(), Xds.std(), atol=1e-3)
