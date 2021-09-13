import numpy as np

from process_nwb.linenoise_notch import apply_linenoise_notch


def test_linenoise_notch_return():
    """Test the return shape.
    """
    X = np.random.randn(1000, 32)
    rate = 200
    Xh = apply_linenoise_notch(X, rate)
    assert Xh.shape == X.shape


def test_frequency_specificity():
    """Test that multiples of 60 Hz are removed and other frequencies are not
    highly filtered.
    """
    dt = 52.  # seconds
    rate = 400.  # Hz
    t = np.linspace(0, dt, int(dt * rate))
    t = np.tile(t[:, np.newaxis], (1, 5))
    n_harmonics = int((rate / 2.) // 60)

    X = np.zeros_like(t)
    for ii in range(n_harmonics):
        X += np.sin(2 * np.pi * (ii + 1) * 60. * t)

    Xp = apply_linenoise_notch(X, rate)
    X = X[int(rate):-int(rate)]
    Xp = Xp[int(rate):-int(rate)]

    assert np.linalg.norm(Xp) < np.linalg.norm(X) / 1000.

    # Offset signals by 2 Hz
    X = np.zeros_like(t)
    for ii in range(n_harmonics):
        X += np.sin(2 * np.pi * ((ii + 1) * 60. + 2) * t)

    Xp = apply_linenoise_notch(X, rate)
    X = X[int(rate):-int(rate)]
    Xp = Xp[int(rate):-int(rate)]

    assert np.allclose(np.linalg.norm(Xp), np.linalg.norm(X), atol=0.1)
