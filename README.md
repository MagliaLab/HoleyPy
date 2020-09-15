# stunning-journey
> Python library for the analysis of amperometric single-channel nanopore recordings.

## Introduction
A python package for analyzing electrophysiology and other things by Florian Lucas.
(Also Matthijs Tadema)

## Main features
* Load data from single-channel nanopore electrophysiology (electrophys)
* Automatic detection of baseline currents
* Threshold search detection of events
* Determination of Excluded currents (I<sub>ex</sub>), its standard deviation (I<sub>ex</sub>SD) and dwell time

## Installing
Information regarding installation.

## Usage example(s)
Importing module classes:
````python
from nanolyse import Trace, filters
from nanolyse.analysis import Levels, Events, Features
````
Loading signal trace(s) and changing the active trace
````python
# Access sample data
import pkg_resources
from pathlib import Path
sample_fname = Path(
        pkg_resources.resource_filename(
            __name__,
            "data/ProteinDigest.abf")
    )

# Load trace(s) from sample data
signal_data = Trace.from_abf(sample_fname)

# Set active trace
signal_data.set_active(1)
````
Setting a signal cut-off, the sample data contains a protocol of switching voltage which we want to trim
````python
# Set the starting (t0) and end (t1) cut-off in seconds
t0 = 1.8
t1 = 5.8
signal_data.set_trim(t0=t0, t1=t1)
````
Adding filters to signal data
````python
filter_frequency = 1000
signal_data.add_filter(filters.gaussian, Fs=filter_frequency)
````
Finding levels, which can be done automatically using Levels, or manually using set_levels.
*Note*: Levels returns (0, 0) if no levels were found. 
````python
# Automatically find levels
levels = Levels(signal_data).run()

# Manually setting levels
baseline_current = -116
baseline_error = 4
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