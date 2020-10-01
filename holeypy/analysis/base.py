"""
Define an analysis abstract base class
"""
from abc import ABC, abstractmethod


class AnalysisBase(ABC):
    """
    Abstract base class for analysis
    Contains the trace object
    Data can be extracted using trace.as_array()
    """
    def __init__(self, trace):
        self.trace = trace
        self.result = None
        self.success = False

    def __bool__(self):
        return self.success

    def _before(self):
        pass

    @abstractmethod
    def _operation(self):
        pass

    def _after(self):
        pass

    def run(self):
        self._before()
        self._operation()
        self.success = True
        self._after()
        return self.result
