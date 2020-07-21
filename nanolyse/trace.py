from . import loaders
import numpy as np
from functools import partial
import copy


def _unfiltered(x):
    return x


class Trace:
    """
    Core object that acts as the interface between the data backend
    and the graphical frontend
    """
    def __init__(self, *, f):
        self.data = {}
        self.frequency = f
        self.active_trace = None
        self.filter_stack = [_unfiltered]

    def __iter__(self):
        for trace in self.data:
            yield trace

    def __len__(self):
        return len(self.data)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return f"Traces: {','.join([str(i) for i, _ in enumerate(self)])}\n" \
               f"Active trace: {self.active_trace}"

    @classmethod
    def from_csv(cls, csv_file, *, f):
        obj = cls(f=f)
        for trace in loaders.csv(csv_file):
            obj.add_data(trace)
        return obj

    @classmethod
    def from_abf(cls, abf_file):
        signal, sampling_period = loaders.axonabf(abf_file)
        obj = cls(f=sampling_period)
        for trace in signal:
            obj.add_data(trace)
        return obj

    @property
    def n_traces(self):
        return len(self.data)

    @property
    def filtered(self):
        stack = copy.deepcopy(self.filter_stack)
        return self._apply_filter_stack(stack)

    def apply_filter_stack(self, stack):
        """
        Recursively apply the filter stack
        :param stack:
        :return: filtered trace
        """
        _filter = stack.pop()
        if len(stack) > 0:
            return _filter(self.apply_filter_stack(stack))
        else:
            return _filter(self.active_trace)

    def __iter__(self):
        for y in self.active_trace:
            yield y

    def __len__(self):
        return self._len

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]

    def add_data(self, array):
        self[self.n_traces] = np.array(array)

    def add_filter(self, _filter, **kwargs):
        partial_filter = partial(_filter, f=self.frequency, **kwargs)
        self.filter_stack.append(partial_filter)

    @property
    def n_traces(self):
        return len(self.data.keys())

    @classmethod
    def from_csv(cls, csv_file, *, f):
        obj = cls(f=f)
        data = loaders.csv(csv_file)
        obj.add_data(data)
        return obj

    def as_array(self):
        """
        Return the internal (filtered) data and corresponding time as
        a numpy array with shape n, 2
        :return: ndarray
        """
        return np.array([list(self.time), list(self.filtered)])

    @property
    def time(self):
        """
        Frequency in kHz
        :return: ndarray
        """
        t = 0
        dt = 1/(self.frequency*1000)
        for _ in range(len(self)):
            yield t
            t += dt

    def set_active(self, key):
        self.active_trace = self.data[key]
        self._len = len(list(self.active_trace))

    def list_traces(self):
        return list(self.data.keys())
