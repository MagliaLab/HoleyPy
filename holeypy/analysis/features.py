# -*- coding: utf-8 -*-
from .base import AnalysisBase
from . import Events
import numpy as np


def get_features_threshold_search(event_data: tuple, sampling_period: float) -> tuple:
    """Feature extraction algorithm.

    This function calculated the excluded current, the excluded current variance and dwell time of events found
    with threshold search (Events).

    Parameters
    ----------
    event_data : tuple
        Result from threshold search (Events)
    sampling_period : float
        The time between two data points

    Returns
    -------
    tuple
        A tuple containing (in order), excluded current, excluded current variance and dwell time of all events

    """
    # Expand variables
    level_0, level_1, _, _, _, _ = event_data

    # Calculate the mean current and standard deviation of baseline events (level 0)
    level_0_mean = np.array([np.mean(i) for i in level_0])
    level_0_sd = np.array([np.std(i) for i in level_0])

    # Calculate the mean current and standard deviation of events (level 1)
    level_1_mean = np.array([np.mean(i) for i in level_1])
    level_1_sd = np.array([np.std(i) for i in level_1])

    # Calculate the excluded current, variance and dwell time (in seconds)
    residual_current = level_1_mean / level_0_mean
    excluded_current = (1-residual_current)
    residual_current_sd_2 = ((level_1_mean / level_0_mean)**2) * (((level_1_sd**2) / (level_1_mean**2)) + ((level_0_sd**2) / (level_0_mean**2)))
    dwell_time = np.array([len(i) * sampling_period for i in level_1])

    # Return the excluded current, variance and dwell time (in seconds)
    return excluded_current, residual_current_sd_2, dwell_time


class Features(AnalysisBase):
    def _before(self):
        self.events = Events(self.trace).run()

    def _operation(self):
        sampling_period = self.trace.sampling_period
        self.result = get_features_threshold_search(self.events, sampling_period)
