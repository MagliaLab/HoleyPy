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
__sample_fname = './data/ProteinDigest.abf'
signal_data = Trace.from_abf(__sample_fname)
signal_data.set_active(1)
````
Filtering signal data
````python
filter_frequency = 1000
signal_data.add_filter(filters.gaussian, Fs=filter_frequency)
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