"""
This file will contain the core object class(es)
"""

def get_extension(datafile):
    """
    Determine the type of data from the datafile

    returns:
        extension (str)
    """
    raise NotImplementedError

def get_reader(datafile, extension):
    """
    Get appropriate reader for datafile and extension

    return:
        file handle? generator?
    """
    raise NotImplementedError

def get_preprocessor(extension, reader):
    """
    Get appropriate preprocessor for data type
    with wrapped reader so that data is automagically preprocessed

    input:
        extension (str)
        reader (generator)

    return:
        generator
    """
    raise NotImplementedError

class Core(np.ndarray):
    """
    We need a "core" object, something that takes care of reading the data,
    maybe preprocesses it according to its type, and then presents
    the data to the user in a user-friendly way.
    Data will be lazily pulled from self._reader.
    New plan, try to subclass from ndarray

    This will do for loading csv files,
    we will have to come up with a different solution for
    hdf5 files
    """
    def __new__(cls, datafile, extension):
        """
        This will initialize our object as a subclasses ndarray
        info @: https://numpy.org/devdocs/user/basics.subclassing.html
        """
        loader = get_loader(extension)
        loader = get_preprocessor(loader, extension)
        data = loader(datafile)
        obj = np.asarray(data).view(cls)
        # Keep the extension and loader just in case
        obj._extension = extension
        obj._loader = loader
        return obj

    def __array_finalize__(self, obj):
        """
        Here will be the actual initialization of new attributes
        """
        if obj is None: return
        self._extension = getattr(obj, '_extension', None)
        self._loader = getattr(obj, '_loader', None)

