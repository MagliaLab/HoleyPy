from abc import ABC

from .traceloader import load
import numpy as np

"""
Provides the core object that holds the raw data
"""


class Core:
    def __init__(self, filename, freq):
        self.filename = filename
        self.freq = freq # in kHz
        self.data = load(self.filename)

    def time(self):
        t = 0
        dt = 1/(self.freq*1000)
        for _ in self.data[0]:
            yield t
            t += dt


class AnalysisBase:
    def __init__(self, *args, **kwargs):
        self.core = kwargs.pop("core", None)
        if self.core is None:
            raise Exception
        self.result = None

    def operation(self):
        raise NotImplementedError("Implemented by child")

    def run(self):
        self.operation()
        return self


class AnalysisBaseGui(AnalysisBase, ABC):
    def __init__(self, *args, **kwargs):
        super(AnalysisBaseGui, self).__init__(*args, **kwargs)

    @property
    def to_plot(self):
        try:
            y = np.array(self.result)
            x = np.array(list(self.core.time()))
        except ValueError:
            raise ValueError("result cannot be converted to x,y")
        return x, y


class Raw(AnalysisBaseGui):
    def __init__(self, *args, **kwargs):
        super(Raw, self).__init__(*args, **kwargs)

    def operation(self):
        if len(self.core.data) > 1:
            raise NotImplementedError
        self.result = self.core.data[0]
