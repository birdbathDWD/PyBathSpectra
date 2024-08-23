## About PyBathSpectra
### Contents
The *PyBathSpectra* repository contains Python code and example files for multimodal postprocessing of Doppler spectra recorded with DWD's operational C-band radar (vertically pointing) birdbath scan, as described by Gergely et al. (2022, 2024).

All postprocessing methods are included in the `pybathspectra` folder. The postprocessing routine can be executed by running `postprocessing.py` with the desired postprocessing settings as selected in `postprocessing_config.yaml`. The `input` folder is a collection of birdbath-scan example data; the `output` folder collects all numerical results as well as plots for visualizing the postprocessing steps.  
### License
Distributed under the MIT License. See `LICENSE` for more information.
## Getting started
Experimental code. Flexibility is king (or queen). Best to download full *PyBathSpectra* repository and run all postprocessing analysis from this directory, **without installing** 'pybathspectra' functionality locally.

Exact version numbers of the Python packages used for the development of `pybathspectra` functionality are listed in `setup.py`. Other versions, especially much newer versions, can cause many compatibility issues, particularly due to the many changes in `NumPy` and `pandas` between different versions. Therefore, it may be best to create a virtual environment with the exact versions of Python packages listed in 'setup.py' and then run the *PyBathSpectra* postprocessing analysis inside this Python environment. 
### Installation
`pybathspectra` package can be installed manually after downloading *PyBathSpectra* repository by executing `python setup.py install`, but to preserve flexibility and avoid (some) compatibility issues, it is best to run postprocessing routine with desired settings directly in downloaded repository structure, i.e., without prior installation of Python package.
## Usage
4 birdbath-scan examples (2x snow and 2x hail) are included in *PyBathSpectra*. The settings for processing the given input files and producing the corresponding outputs are listed as `postprocessing_config_....yaml`. To run one of these examples, the settings filename has to be modified accordingly in line 36 of `postprocessing.py`.    
## Citing the code
Gergely, M., Schaper, M., Toussaint, M., and Frech, M., 2022: Doppler spectra from DWD’s operational C-band radar birdbath scan: sampling strategy, spectral postprocessing, and multimodal analysis for the retrieval of precipitation processes, *Atmos. Meas. Tech.*, 15, 7315–7335, https://doi.org/10.5194/amt-15-7315-2022

Gergely, M., Ockenfuß, P., Kneifel, S., Frech, M., 2024: Postprocessing methods to characterize multimodal precipitation in Doppler spectra from DWD's C-band radar birdbath scan, *ERAD 2024 – 12th European Conference on Radar in Meteorology and Hydrology*, 9–13 September 2024, Rome, Italy. Extended Abstract: LINK_GOES_HERE
## Acknowledegments
The work is supported by the German Research Foundation (DFG) 'PROM' priority program SPP-2115 (https://www2.meteo.uni-bonn.de/spp2115) and the German Meteorological Service (Deutscher Wetterdienst, DWD, https://www.dwd.de/DE/Home/home_node.html).
<!-- ## References -->
