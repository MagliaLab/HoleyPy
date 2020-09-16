from .base import AnalysisBase
import warnings

import numpy as np
from scipy.optimize import curve_fit


def get_levels(signal: np.array, sigma=1, trace=0, t0=0, t1=-1) -> tuple:
    """Level detection algorithm.
    
    This function uses a gaussian fit around the main signal to determine the open pore current and threshold cut-off.
        
    Parameters
    ----------
    signal : numpy array
        The signal should be fed as a array of arrays.
        Each top-level array is threated as a trace of a signal allowing easy cross-trace analysis (for e.g. current dependent analysis).
    sigma : int
        The number of sigma (multiplier), that the threshold is set from the open pore.
    trace : int
        The trace to be analysed (n-th number array)
    t0 : int
        The first datapoint to be analysed in the trace.
    t1 : int
        The last datapoint to be analysed in the trace
    
    Returns
    -------
    tuple
        A tuple containing (in order), centroid of the open pore current and threshold.
        
    """
    # Fetch and trim signal
    signal = signal[trace][t0:t1]

    # Determine the orientation of the signal
    signal_orientation = np.sign(sum(signal))*-1

    # Ensure the signal is always negative, such that the baseline is always the lowest possible level
    signal = abs(np.array(signal))*-1

    # Fetch the lowest central normal distribution
    centre, variance, residual = _ndf_deconvolution(signal)
    l0 = centre * signal_orientation
    l1 = variance * sigma * signal_orientation
    return l0, l1


def _ndf(x, *p) -> np.array:
    a, mu, sigma = p
    return a*np.exp(-(x - mu)**2 / float(2 * sigma**2))


def _ndf_deconvolution(signal, n_peaks=2) -> tuple:
    # Initialise variables
    centres, variance = ([], [])

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
        _, centre, vrs = p
        centres.append(centre)
        variance.append(vrs)
    # Return the peak centre and variance of the normal distribution with the lowest peak centre
    return centres[np.argmax(np.abs(centres))], abs(variance[np.argmax(np.abs(centres))]), residual


class Levels(AnalysisBase):
    def _operation(self):
        signal = self.trace.data
        trace = self.trace.active_trace
        self.result = get_levels(signal,
                                 trace=trace,
                                 t0=self.trace.t0,
                                 t1=self.trace.t1
                                 )

    def _after(self):
        if self.result[0] is False:
            warnings.warn("Found no levels")
            self.success = False
