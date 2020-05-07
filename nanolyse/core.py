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

class Core():
    """
    We need a "core" object, something that takes care of reading the data,
    maybe preprocesses it according to its type, and then presents
    the data to the user in a user-friendly way.
    Data will be lazily pulled from self._reader
    """
    def __init__(self, datafile):
        extension = get_extension(datafile)
        reader = get_reader(datafile, extension)
        self._reader = get_preprocessor(extension, reader)
        pass
