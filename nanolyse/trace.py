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
        self.data = []
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

    def _apply_filter_stack(self, stack):
        """
        Recursively apply the filter stack
        :param stack:
        :return: filtered trace
        """
        _filter = stack.pop()
        if len(stack) > 0:
            return _filter(self._apply_filter_stack(stack))
        else:
            i = self.active_trace
            return _filter(np.array(self[i]))

    def add_data(self, array):
        self.set_active(len(self))
        self.data.append(np.array(array))

    def add_filter(self, _filter, **kwargs):
        partial_filter = partial(_filter, f=self.frequency, **kwargs)
        self.filter_stack.append(partial_filter)

    def as_array(self):
        """
        Return the internal (filtered) data and corresponding time as
        a numpy array
        :return: ndarray
        """
        return np.array(self.filtered)

    @property
    def time(self):
        """
        Frequency in kHz
        :return: ndarray
        """
        dt = 1/(self.frequency*1000)
        i = self.active_trace
        out = []
        t = 0
        for _ in range(len(self[i])):
            out.append(t)
            t += dt
        return np.array(out)

    def set_active(self, key):
        self.active_trace = key

    def join_traces(self, t0=0, t1=-1):
        signal = self.data
        sampling_period = self.frequency
        if (t0 >= 0) & (t0 <= len(signal[0])):
            t0 = int(t0 / sampling_period)
        else:
            t0 = 0
        if (t1 >= 0) & (t1 <= len(signal[0])):
            t1 = int(t1 / sampling_period)
        else:
            t1 = -1
        Y = np.array([])
        for trace in signal:
            Y = np.concatenate((Y, np.array(trace[t0:t1])))
        return [Y]