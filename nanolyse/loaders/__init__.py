# -*- coding: utf-8 -*-
import os

from . import csv
from . import axonabf


_ext_dict = {
    ".csv": "csvloader",
    ".abf": "axonabf",
}


def load(filename):
    """Function to load data traces.

    Dynamically load data traces based on file extension.
    Support:
        .csv
        .abf

    Parameters
    ----------
    filename: str
            File path

    Returns
    -------
    list
            Return list containing data traces
    """
    _, ext = os.path.splitext(filename)
    try:
        return eval(_ext_dict[ext]).load(filename)
    except OSError:
        return False
