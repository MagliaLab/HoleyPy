"""
loaders submodule only takes care of
importing the appropriate libraries
and presenting the available loaders
in ext_to_loader
"""
ext_to_loader = {}

from . import csv
try:
    # Axonabf loading is an optional dependency
    from . import axonabf
except ImportError:
    # If axonabf fails to import
    # it will simply not be available
    pass

