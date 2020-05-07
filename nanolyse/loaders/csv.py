import numpy as np
from ..loaders import ext_to_loader

def load_csv(filename, delimiter=''):
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

def load_csv_lazy(filename, lb=0, ub=None):
    filehandle = open(filename)
    try:
        for i, line in enumerate(filehandle):
            if i >= lb:
                yield float(line.strip())
            if ub is not None and i > ub:
                break
    finally:
        filehandle.close()

ext_to_loader[".csv"] = load_csv