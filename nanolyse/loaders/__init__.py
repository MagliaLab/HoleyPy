"""
loaders submodule only takes care of
importing the appropriate libraries
and presenting the available loaders
in ext_to_loader
All loaders return some dictionary-like object
"""
ext_to_loader = {}

from . import csv
from . import hdf5
try:
    # Axonabf loading is an optional dependency
    from . import axonabf
except ImportError:
    # If axonabf fails to import
    # it will simply not be available
    pass
