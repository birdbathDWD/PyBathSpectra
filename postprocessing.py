"""
Running full spectral postprocessing of DWD birdbath-scan Doppler spectra
with the settings specified in the settings_file.

Copyright (c) 2024 Mathias Gergely, DWD

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import yaml

from  pybathspectra import reader
from pybathspectra.birdbathscan import BirdBathScan


# Find directory of this postprocessing.py script
this_directory = os.path.dirname(os.path.abspath(__file__))
# Path to config file containing the settings for postprocessing routine
settings_file = os.path.join(this_directory, 'postprocessing_config.yaml')


###########################################################################
## Pre-settings, if desired: Python environment and working directory     #
###########################################################################
## Use (uninstalled) pybathspectra modules from subfolder of this directory
#import sys
#if sys.path[0] == this_directory:
#    pass
#else:
#    sys.path.insert(0, this_directory)
## Explicitly set current working directory to resolve relative paths
## for plotting and saving postprocessing data and results
#os.chdir(this_directory)


###########################################################################
# From here on, no interaction with code; set all options in config file  #
###########################################################################

# Load current postprocessor settings from .yaml config file
with open(settings_file, 'r') as config_file:
    settings = yaml.safe_load(config_file)

# Split settings into groups for different postprocessing steps
settings_groups = reader.split_settings(settings)
reader_settings, filter_settings, modes_settings = settings_groups

# Birdbath scan timestamp
# could be used to loop over multiple birdbath scans
birdbath_timestamp = settings['birdbath_time'][0]

# Load birdbath scan data (= radar output or DWD database files)
birdbath_scan = BirdBathScan()
birdbath_scan.load(birdbath_timestamp, reader_settings)

# Isolate weather signal in Doppler spectra
# i.e. filter out clutter and background, if specified in settings
birdbath_scan.isolate(filter_settings)

# Multimodal analysis of isolated weather Doppler spectra
birdbath_scan.analyze(modes_settings)

# Plot radar outputs and postprocessing results, if selected
if settings['plot_all']:
    birdbath_scan.plot(
        birdbath_timestamp, settings['rgb_scaling'], settings['plot_dir'])

# Save postprocessing results, if selected
if settings['save_results']:
    birdbath_scan.save(
        birdbath_timestamp, settings['results_dir'])
