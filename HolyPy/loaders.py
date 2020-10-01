import numpy as np
from neo.io import AxonIO


def axonabf(filename) -> (np.array, float):
    """
    Load axon binary file (abf) files.
    :param filename:
    :return: signal trace, sampling_period
    """
    try:
        bl = AxonIO(filename=filename).read()                                               # Load axon file into memory
        traces = bl[0].segments                                                             # Extract all traces
        signal = np.array([np.array(seg.analogsignals[0])[:, 0] for seg in traces])         # Get signal traces
        sampling_period = np.asscalar(bl[0].segments[0].analogsignals[0].sampling_period)   # Time between data points
        return signal, sampling_period
    except NameError:
        print('Error while loading. .abf')
        return None, None


def csv(filename, delimiter=''):
    """Function to load .csv files.

    Loads data traces from comma separated files.

    Parameters
    ----------
    filename: str
            File path


    Returns
    -------
    list
            Return list of data traces
            :param filename: str
            :param delimiter: str
    """

    return np.genfromtxt(filename, delimiter=delimiter)