"""
All unittests and integration tests will go here
"""
import numpy as np
import nanolyse as nl

__sampling_period = 2e-5
__test_filter = 2000
__blank_fname = './data/Blank.csv'
__sample_fname = './data/ProteinDigest.csv'
__levels = (True, -115.7, 1.2)

def load_filt( fname ):
    return nl.filter_gaussian( [np.genfromtxt(__blank_fname, delimiter='')], __sampling_period, __test_filter )

def test_get_levels():
    blank_array = load_filt(__blank_fname)
    levels = nl.get_levels(blank_array)

def test_get_events():
    sample_array = load_filt(__sample_fname)
    event_data = nl.thresholdsearch(sample_array, __sampling_period, __levels)
    nl.get_features_THS(event_data, __sampling_period)

