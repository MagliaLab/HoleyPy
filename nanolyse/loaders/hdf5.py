import h5py
from ..loaders import ext_to_loader

def load_hdf5(filename):
    return h5py.File(filename)

ext_to_loader[".h5"] = load_hdf5