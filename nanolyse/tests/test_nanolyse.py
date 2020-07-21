"""
All unittests and integration tests will go here
"""
import pytest
import pkg_resources
from pathlib import Path

from .. import Trace
from ..filters import *
from ..analysis import *

_sampling_period = 2e-5
_test_filter = 2000
_sample_fname = './data/ProteinDigest.csv'
_levels = (True, -115.7, 1.2)


@pytest.fixture
def blank_trace():
    blank = Path(
        pkg_resources.resource_filename(
            __name__,
            "data/Blank.csv")
    )
    return Trace.from_csv(blank, f=1000)


@pytest.fixture
def protein_digest():
    pd = Path(
        pkg_resources.resource_filename(
            __name__,
            "data/ProteinDigest.csv")
    )
    return Trace.from_csv(pd, f=1000)


def test_trace(blank_trace):
    assert len(blank_trace.data[0]) > 0


def test_filters(protein_digest):
    protein_digest.add_filter(gaussian, Fs=1000)
    return protein_digest.as_array()


def test_levels(protein_digest):
    return Levels(protein_digest).run()


def test_events(protein_digest):
    return Events(protein_digest).run()


def test_features(protein_digest):
    return Features(protein_digest).run()

