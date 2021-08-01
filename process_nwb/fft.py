import signal
from numpy.fft import rfftfreq, fftfreq, ifftshift
try:
    from mkl_fft._numpy_fft import (rfft as mklrfft, irfft as mklirfft,
                                    fft as mklfft, ifft as mklifft)
    from numpy.fft import (rfft as nprfft, irfft as npirfft,
                           fft as npfft, ifft as npifft)

    string = ("Internal error occurred: b'Intel MKL DFTI ERROR: "
              + "Inconsistent configuration parameters'")

    def segfault(*args):
        raise ValueError(string)

    def rfft(*args, **kwargs):
        """MKL rfft has a bug for some shapes. This catches those errors and falls back
        on numpy rfft.
        See https://github.com/IntelPython/mkl_fft/issues/57
        """
        signal.signal(signal.SIGSEGV, segfault)
        try:
            return mklrfft(*args, **kwargs)
        except ValueError as e:
            if str(e) == string:
                return nprfft(*args, **kwargs)
            else:
                raise e

    def irfft(*args, **kwargs):
        """MKL irfft has a bug for some shapes. This catches those errors and falls back
        on numpy irfft.
        See https://github.com/IntelPython/mkl_fft/issues/57
        """
        signal.signal(signal.SIGSEGV, segfault)
        try:
            return mklirfft(*args, **kwargs)
        except ValueError as e:
            if str(e) == string:
                return npirfft(*args, **kwargs)
            else:
                raise e

    def fft(*args, **kwargs):
        """MKL fft has a bug for some shapes. This catches those errors and falls back
        on numpy fft.
        See https://github.com/IntelPython/mkl_fft/issues/57
        """
        signal.signal(signal.SIGSEGV, segfault)
        try:
            return mklfft(*args, **kwargs)
        except ValueError as e:
            if str(e) == string:
                return npfft(*args, **kwargs)
            else:
                raise e

    def ifft(*args, **kwargs):
        """MKL ifft has a bug for some shapes. This catches those errors and falls back
        on numpy ifft.
        See https://github.com/IntelPython/mkl_fft/issues/57
        """
        signal.signal(signal.SIGSEGV, segfault)
        try:
            return mklifft(*args, **kwargs)
        except ValueError as e:
            if str(e) == string:
                return npifft(*args, **kwargs)
            else:
                raise e

except ImportError:
    try:
        from pyfftw.interfaces.numpy_fft import rfft, irfft, fft, ifft
    except ImportError:
        from numpy.fft import rfft, irfft, fft, ifft

from scipy.fft import rfft, irfft, fft, ifft

__all__ = ['rfftfreq', 'fftfreq', 'ifftshift', 'fft', 'ifft', 'rfft', 'irfft']
