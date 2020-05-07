import numpy as np


def load(filename, delimiter=''):
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

    return [np.genfromtxt(filename, delimiter=delimiter)]
