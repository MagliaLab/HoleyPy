"""
This file will contain the core object class(es)
"""
import pathlib
import warnings
import numpy as np
from .loaders import ext_to_loader

def get_extension(datafile):
    """
    Determine the type of data from the datafile

    returns:
        extension (str)
    """
    return pathlib.Path(datafile).suffix

def get_loader(extension):
    """
    Get appropriate loader for datafile and extension

    return:
        file handle? generator?
    """
    try:
        return ext_to_loader[extension]
    except KeyError:
        msg = "No loader for extension: " + extension + "\n"
        msg += "Available loaders: " + ext_to_loader.keys()
        raise KeyError(msg)

def get_preprocessor(loader, extension):
    """
    Get appropriate preprocessor for data type
    with wrapped reader so that data is automagically preprocessed

    input:
        extension (str)
        reader (generator)

    return:
        generator
    """
    # Not implemented for now
    return loader

h5_config = {
    "traces": {
        "force1": "Force LF/Force 1x",
        "force2": "Force LF/Force 2x"
    }
}

class Core:
    """
    self.data is some dictionary-like object
    """
    def __init__(self, filename):
        extension = get_extension(filename)
        loader = get_loader(extension)
        self.data = loader(filename)
        self.config = h5_config

    def __getitem__(self, item):
        return self.data[self.config["traces"][item]]

    def keys(self):
        return self.config["traces"]
