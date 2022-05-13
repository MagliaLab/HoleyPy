try:
    from neo.io import AxonIO
    import numpy as np
    _state = True
except ModuleNotFoundError:
    _state = False


_trace_extension = '.abf'


def load(filename):
    """Load data from axon binary file into nanolyse.

    Load data from axon binary file into nanolyse.

    Parameters
    ----------
    filename : str
        String with the location of the axon binary file to be loaded

    Returns
    -------
    Numpy array, float
        Returns the signal extracted from the input, and the sampling period as a float

    """

    try:

        # Load axon file into memory
        bl = AxonIO(filename=filename).read()

        # Get signal traces
        signal = [np.array(seg.analogsignals[0])[:, 0] for seg in bl[0].segments]

        # Time between data points
        sampling_period = [np.asscalar(bl[0].segments[0].analogsignals[0].sampling_period) for seg in bl[0].segments][0]
        return signal, sampling_period
    except:
        print("Unable to load .abf file")
