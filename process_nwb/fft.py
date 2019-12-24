from numpy.fft import rfftfreq, fftfreq, ifftshift
try:
    from mkl_fft._numpy_fft import rfft, irfft, fft, ifft
except ImportError:
    try:
        from pyfftw.interfaces.numpy_fft import rfft, irfft, fft, ifft
    except ImportError:
        from numpy.fft import rfft, irfft, fft, ifft


__all__ = ['rfftfreq', 'fftfreq', 'ifftshift',
           'fft', 'ifft', 'rfft', 'irfft']
