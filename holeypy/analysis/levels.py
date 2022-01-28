from .base import AnalysisBase
from .. import Trace
import warnings

import numpy as np
from scipy.optimize import curve_fit


def get_levels(trace: Trace, sigma=3, *, t0=None, t1=None) -> tuple:
    """Level detection algorithm.
    
    This function uses a gaussian fit around the main signal to determine the open pore current and threshold cut-off.
        
    Parameters
    ----------
    signal : numpy array
        The signal should be fed as a array of arrays.
        Each top-level array is threated as a trace of a signal allowing easy cross-trace analysis (for e.g. current dependent analysis).
    sigma : int
        The number of sigma (multiplier), that the threshold is set from the open pore.
    trace : int ! Deprecated !, active trace should be set in Trace object
        The trace to be analysed (n-th number array)
    t0 : int ! Deprecated !, active trace should be set in Trace object
        The first datapoint to be analysed in the trace.
    t1 : int
        The last datapoint to be analysed in the trace
    
    Returns
    -------
    tuple
        A tuple containing (in order), centroid of the open pore current and threshold.
        
    """
    if any(t is not None for t in [t0, t1]):
        raise UserWarning("Some parameters are deprecated, please set them in the trace objects")
    # Fetch and trim signal
    signal = trace.as_array()

    # Determine the orientation of the signal
    signal_orientation = np.sign(sum(signal))*-1

    # Ensure the signal is always negative, such that the baseline is always the lowest possible level
    signal = abs(np.array(signal))*-1

    # Fetch the lowest central normal distribution
    centres, variances, residual, *_ = _ndf_deconvolution(signal)
    l0 = centres[0] * signal_orientation
    l1 = variances[0] * sigma * signal_orientation
    return l0, l1


def _ndf(x, *p) -> np.array:
    a, mu, sigma = p
    return a*np.exp(-(x - mu)**2 / float(2 * sigma**2))


def _ndf_deconvolution(signal, n_peaks=2) -> tuple:
    # Initialise variables
    centres = []
    variance = []
    amplitudes = []

    # Create a histogram of the data
    n_bins = 2000
    x_bins = np.linspace(min(signal), max(signal), n_bins+1)
    x_edges = np.linspace(min(signal), max(signal), n_bins)
    residual, _ = np.histogram(signal, bins=x_bins)

    # Try to fit a NDF around the user defined number of peaks (default=2)
    for i in range(n_peaks):
        # Estimate starting parameters
        amplitude_0 = max(residual)
        centre_0 = np.median(signal)
        variance_0 = 1
        p0 = [amplitude_0, centre_0, variance_0]

        # Fit a normal distribution function around the (residual) data
        try:
            p, var_matrix = curve_fit(_ndf,
                                      x_edges,
                                      residual,
                                      p0=p0)
        except RuntimeError:
            p = (0, 0, 0)
            warnings.warn("Could not fit normal distribution around data", UserWarning)

        # Subtract the fitted normal distribution from the data
        # As the normal distribution returns floats and the histogram consists of integers
        # Typecast the resulting distribution to integers prior to subtraction
        residual -= _ndf(x_edges, *p).astype('int32')

        # Extract the peak centre and variance, add them to the list of fitted centres and variance
        # I need amplitudes for other stuff
        amplitude, centre, vrs = p
        amplitudes.append(amplitude)
        centres.append(centre)
        variance.append(vrs)
    # Return the peak centre and variance of the normal distribution with the lowest peak centre
    # Why not return all?
    sort_arg = np.argsort(centres)
    return np.array(centres)[sort_arg], np.array(variance)[sort_arg], residual, np.array(amplitudes)[sort_arg]


class Levels(AnalysisBase):
    def _operation(self):
        trace = self.trace
        try:
            sigma = self.__getattribute__('sigma')
        except AttributeError:
            sigma = 3 # default sigma
        self.result = get_levels(trace, sigma)

    def _after(self):
        if self.result[0] is False:
            warnings.warn("Found no levels")
            self.success = False
