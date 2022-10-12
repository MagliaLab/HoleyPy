![HoleyPy](./img/HoleyPy.png)
> Python library for the analysis of amperometric single-channel nanopore recordings.

## Introduction
A python package for analyzing electrophysiology and other things by Florian Lucas.
(Also Matthijs Tadema)

## Main features
* Load data from single-channel nanopore electrophysiology (electrophys)
* Automatic detection of baseline currents
* Threshold search detection of events
* Determination of Excluded currents (I<sub>ex</sub>), its standard deviation (I<sub>ex</sub>SD) and dwell time

## Usage example(s)
Importing module classes:
````python
from holeypy import Trace, filters
from holeypy.analysis import Levels, Events, Features
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
Minimal dwell time cut-offs are important as it is an easy way to discard noise. It is also optional to allow for outliers (default=2 data points)
````python
# Set the minimal dwell time (in seconds) for each event
signal_data.set_dwell_time_cutoff(4e-4)

# (optional) set the number of outlier data points before ending an event (default=2)
signal_data.set_event_skip(2)
````

Finding events and features using threshold search
````python
# If levels were not set manually, this will use Levels(signal_data) to determine them.
event_analysis = Events(signal_data)
event_analysis.run()

# The events are returned as a list of named tuples with fields: 
#   baseline_start: (int) datapoint where baseline started
#   baseline_end: (int) datapoint where baseline ended
#   event_start: (int) datapoint where event started 
#   event_end: (int) datapoint where baseline started 
#   t0: (int) offset
events = event_analysis.events

# The features are returned as a list of named tuples with fields: 
#   baseline_current: (float) baseline current
#   baseline_sd: (float) square root of the variance of the baseline current
#   event_current: (float) event current
#   event_sd: (float) square root of the variance of the event current
#   residual_current: (float) residual current fraction of the event
#   residual_current_sd: (float) square root of the variance of the residual current (error propagated)
#   dwell_time: (float) residence time of the event in seconds
features = event_analysis.features
````

The events and features can be accessed using named tuples, which can be used to, for example, calculate the excluded current.
````python
excluded_current = np.array([1-i.residual_current for i in features]) # excluded current fraction
excluded_current_sd = np.array([i.residual_current_sd for i in features]) # The square root of the variance is equal to the residual
````

For the unbiased data analysis for the parameterization of fast translocation events run the following code. This will fit a function (default generalised Normal Distribution Function) around each event.
````python
results = event_analysis.optimise_events()

# returns a list of named tuples, each element in the list is an optimized event.
# - Method: (str) method used for optimisation
# - Trace: (int) trace number
# - Function: (str) function used for optimisation, e.g. gNDF
# - Fitting_parameters: (tuple) parameters for Function
# - Amplitude_block: (float) amplitude of the block
# - Localisation: (float) centroid of the block in time
# - Sigma: (float) sigma over the localisation
# - Beta: (float) beta parameter
# - Open_current: (float) open pore current
# - Dwell_time: (float) dwell time of the event
# - Fs_event: (float) effective frequency
````

## Authors and acknowledgements
#### Main contributors
* [Florian L. R. Lucas](http://orcid.org/0000-0002-9561-5408 "Orcid page")
* [Matthijs Tadema](https://www.rug.nl/staff/m.j.tadema/ "University of Groningen staff page")

#### Acknowledgements
* [Maglia lab, University of Groningen](https://sites.google.com/a/rug.nl/maglia-lab-groningen/ "University of Groningen Maglia lab page")

## Changelog
12-10-2022: Updated readme page

## How to cite
When using this library, please be kind to cite the following work: 

<strong>Unbiased Data Analysis for the Parameterization of Fast Translocation Events through Nanopores</strong>
Florian L. R. Lucas, Kherim Willems, Matthijs J. Tadema, Katarzyna M. Tych, Giovanni Maglia, and Carsten Wloka
ACS Omega 2022 7 (30), 26040-26046
DOI: 10.1021/acsomega.2c00871 