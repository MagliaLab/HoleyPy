# stunning-journey
> Python library for the analysis of amperometric single-channel nanopore recordings.

## Introduction
A python package for analyzing electrophysiology and other things by Florian Lucas.
(Also Matthijs Tadema)

## Main features
* Load data from single-channel nanopore electrophysiology (electrophys)
* Filter electrophys 
* Detect events within electrophys data

## Usage example(s)
Importing module classes:
````python
from nanolyse import Trace, filters
from nanolyse.analysis import Levels, Events, Features
````
Loading signal trace(s) and changing the active trace
````python
sample_fname = './data/ProteinDigest.abf'
signal_data = Trace.from_abf(sample_fname)
signal_data.set_active(1)
````
Filtering signal data
````python
filter_frequency = 1000
signal_data.add_filter(filters.gaussian, Fs=filter_frequency)
````
Finding levels
````python
# Automatically find levels
levels = Levels(signal_data).run()

# Manually setting levels
baseline_current = -200
baseline_error = 2
signal_data.set_levels(baseline_current, baseline_error)
````
Finding events using threshold search
````python
# If levels were not set manually, this will use Levels(signal_data) to determine them.
L0, L1, L0_start, L0_end, L1_start, L1_end = Events(signal_data).run()

# L0 contains an numpy.ndarray with signals from the baseline
# L1 contains an numpy.ndarray with signals from the events
# L0_start contains an numpy.array with the start times of the baseline
# L0_end contains an numpy.array with the end times of the baseline
# L1_start contains an numpy.array with the start times of the events
# L1_end contains an numpy.array with the end times of the events
````
Getting the Excluded current, their standard deviation and dwelltime
````python
# This will run the Events(signal_data) prior to fetching features.
# If levels were not set manually, it will also run Levels(signal_data) to determine them.
Iex, IexSD, dwelltime = Features(signal_data).run()
````

## Documentation
We have to add the documentation here.

## Authors and acknowledgements
#### Main contributors
* [Florian L. R. Lucas](https://www.rug.nl/staff/f.l.r.lucas/ "University of Groningen staff page")
* [Matthijs Tadema](https://www.rug.nl/staff/m.j.tadema/ "University of Groningen staff page")

#### Acknowledgements
* [Maglia lab, University of Groningen](https://sites.google.com/a/rug.nl/maglia-lab-groningen/ "University of Groningen Maglia lab page")

## Changelog

## License
License information.

## How to cite
Citation information.

## Related research papers
* Let's make a nice list here