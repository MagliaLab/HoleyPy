# -*- coding: utf-8 -*-
from .base import AnalysisBase
from .levels import Levels
import numpy as np
import warnings


def threshold_search(signal: np.ndarray, sampling_period: float, levels: tuple,
                     dwell_time=0, skip=2, trace=0, t0=0, t1=-1) -> tuple:
    """Event detection algorithm using threshold.
    
    This function uses a defined set of cut-off parameters (levels) to determine event locations.
    It utilizes a threshold for shot noise, to prevent preliminary decisions.
    
    Parameters
    ----------
    signal : numpy array
        The signal should be fed as a array of arrays.
        Each top-level array is threated as a trace of a signal allowing easy cross-trace analysis (for e.g. current dependent analysis).
    sampling_period : float
        The sampling_frequency is used in combination with the dwelltime argument exclude short events.
    levels : tuple
        The levels determine the center of the baseline and it's minimal threshold. 
        levels = ( center, threshold ).
    dwell_time : float
        The minimal event length event should have to be included.
    skip : int
        The skip parameter will exclude a number of events between events as shotnoise, and combines them into a single event.
        default = 2.
    trace : int
        The trace to be analysed (n-th number array)
    t0 : int
        The first datapoint to be analysed in the trace.
    t1 : int
        The last datapoint to be analysed in the trace
    
    Returns
    -------
    tuple
        A tuple containing (in order), the baseline events, signal events, baseline events start, basline events end, signal events start and signal events end.
        All start and ends are returned as position of the datapoint in the trace.
        
    """
    # Expand variables
    l0, l1 = levels

    # Trim the current trace
    trace_length = len(signal[trace])
    signal = signal[trace][t0:min(trace_length, t1)]

    # All data points above the threshold
    a = np.where(abs(np.array(signal)) < (abs(l0) - abs(l1)))[0]

    # Get all event starts (e.g. where two data points are maximum 'skip' apart), relative to vector a
    # Also, we need to append the last data point of the series
    level_1_end_index_a = np.where(np.diff(a) > skip)[0]
    level_1_end_index_a = np.append(level_1_end_index_a, len(a)-1)

    # We know that all data in vector a is above the baseline, and we know where each block ends
    # The start of these blocks are one index before the end (except the last index)
    # Instead, the first index should be 0
    level_1_start_index_a = np.delete(np.insert(level_1_end_index_a + 1, 0, 0), -1, 0)

    # While the dwell time suggests that also spikes can be seen, at least 2 data points are required to be an event
    n_filter = max(2, int(dwell_time / float(sampling_period)))

    # Only keep those events that are at least 2 or n_filter data points long
    idx = np.where(level_1_end_index_a - level_1_start_index_a > n_filter)[0]

    # The vector a contains the indices of the signal that are above the threshold
    # The level_1_end_index_a and level_1_start_index_a contain the indices in vector a
    # Which we consider to end or start the block. We also defined idx as the indices where
    # The difference between beginning and end are long enough to count as events.
    # We combine these factors to get the indices (in the signal) that correspond to the beginning
    # and end of the level 1 events.
    level_1_start = a[level_1_start_index_a[idx]]
    level_1_end = a[level_1_end_index_a[idx]]

    # We know that whenever level 1 starts, level 0 just ended one data point ahead.
    # We also know that whenever level 1 ends, level 0 starts one data point after.
    level_0_start = np.delete(np.insert(level_1_end + 1, 0, 0), -1, 0)
    level_0_end = level_1_start - 1

    # Simply insert the times from level_0 and level_1 to get the trace data
    # Also, the type must be object as we have an n-dimensional array of multiple sizes
    level_0 = np.array([signal[i:j] for i, j in zip(level_0_start, level_0_end)], dtype=object)
    level_1 = np.array([signal[i:j] for i, j in zip(level_1_start, level_1_end)], dtype=object)

    # Return the level information
    return level_0, level_1, level_0_start, level_0_end, level_1_start, level_1_end


class Events(AnalysisBase):
    def _before(self):
        if self.trace.levels:
            self.levels = self.trace.levels
        else:
            self.levels = Levels(self.trace).run()

    def _operation(self):
        signal = np.array(self.trace.data)
        sampling_period = self.trace.sampling_period
        trace = self.trace.active_trace
        dwell_time = self.trace.minimal_dwell_time
        skip = self.trace.event_skip
        self.result = threshold_search(signal, sampling_period,
                                       levels=self.levels,
                                       dwell_time=dwell_time,
                                       skip=skip,
                                       trace=trace,
                                       t0=self.trace.t0,
                                       t1=self.trace.t1
                                       )

    def _after(self):
        for array in self.result:
            if len(array) == 0:
                warnings.warn("Found no events")
