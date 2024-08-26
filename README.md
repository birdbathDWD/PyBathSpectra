## About PyBathSpectra
### Contents
The *PyBathSpectra* repository contains Python code and example files for multimodal postprocessing of Doppler spectra recorded with DWD's operational C-band radar (vertically pointing) birdbath scan, as described by Gergely et al. (2022, 2024).
### License
Distributed under the MIT License. See `LICENSE` for more information.
## Getting started
Experimental code. Flexibility is king (or queen). Best to download full *PyBathSpectra* repository and run all postprocessing analysis from this directory, **without installing** 'pybathspectra' functionality locally.

Exact version numbers of the Python packages used for developing `pybathspectra` modules are listed in `setup.py`. Other versions, especially much newer versions, can (and will) cause compatibility issues, particularly due to the many changes in `NumPy` and `pandas` between different version numbers. Therefore, it may be best to create a virtual environment with the exact versions of Python packages listed in 'setup.py' and then run the *PyBathSpectra* postprocessing analysis inside this Python environment. 
### Installation
The `pybathspectra` Python package can be installed manually after downloading the *PyBathSpectra* repository by executing `python setup.py install`. But to preserve flexibility and avoid (some) compatibility issues, it is best to run the postprocessing routine directly in the downloaded repository structure, i.e., without prior installation of the Python package.
## Usage
All postprocessing methods are collected in the `pybathspectra` folder. The postprocessing routine can be executed by running `postprocessing.py` with the desired postprocessing settings selected in `postprocessing_config.yaml`. The `input` folder is a collection of birdbath-scan example data; the `output` folder collects all numerical results as well as plots for visualizing the postprocessing steps.

4 birdbath-scan examples (2x snow and 2x hail) are included in `input`. The settings for processing the given input files and producing the corresponding outputs are listed in the `postprocessing_config_....yaml` files. To run one of these examples, the 'settings_file' has to be modified accordingly in line 36 of `postprocessing.py`.    
## Citing the code
Gergely, M., Schaper, M., Toussaint, M., and Frech, M., 2022: Doppler spectra from DWD’s operational C-band radar birdbath scan: sampling strategy, spectral postprocessing, and multimodal analysis for the retrieval of precipitation processes, *Atmos. Meas. Tech.*, 15, 7315–7335, https://doi.org/10.5194/amt-15-7315-2022

Gergely, M., Ockenfuß, P., Kneifel, S., Frech, M., 2024: Postprocessing methods to characterize multimodal precipitation in Doppler spectra from DWD's C-band radar birdbath scan, *ERAD 2024 – 12th European Conference on Radar in Meteorology and Hydrology*, 9–13 September 2024, Rome, Italy. Extended Abstract: LINK_GOES_HERE
## Acknowledegments
The work is supported by the German Research Foundation (DFG) 'PROM' priority program SPP-2115 (https://www2.meteo.uni-bonn.de/spp2115) and the German Meteorological Service (Deutscher Wetterdienst, DWD, https://www.dwd.de/DE/Home/home_node.html).
<!-- ## References -->
