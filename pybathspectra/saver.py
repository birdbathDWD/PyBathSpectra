"""
Functions for saving postprocessing results of DWD birdbath scans.

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
import datetime as dt
import numpy as np


def save_multimodal(multimodal_analysis_data, time, resdir='./results/'):
    """
    Save results from multimodal analysis.
    
    Save multimodal analysis results (modal properties incl. flag,
    multimodal properties, and their uncertainties) at time=datetime_object
    in .txt files in resdir directory after reshaping results arrays to 2D. 
    """
    
    # Create directory for results if it does not exist
    os.makedirs(resdir, exist_ok=True)
    
    # Extract individual datasets before reshaping and saving
    height_vector = multimodal_analysis_data['range_heights']
    modal = multimodal_analysis_data['modal_properties']
    multimodal = multimodal_analysis_data['multimodal_properties']
    uncertain_modal = multimodal_analysis_data['modal_uncertainties']
    uncertain_multi = multimodal_analysis_data['multimodal_uncertainties']
    
    # Reshape results to 2-D
    modal_reshaped= modal.reshape(
        (modal.shape[0]*modal.shape[1], modal.shape[2]), order='F')
    multimodal_reshaped = multimodal.reshape(
        (multimodal.shape[0]*multimodal.shape[1], multimodal.shape[2]),
        order='F')
    uncertainty_modal_reshaped = uncertain_modal.reshape(
        (uncertain_modal.shape[0]*uncertain_modal.shape[1], uncertain_modal.shape[2]),
        order='F')
    uncertainty_multimodal_reshaped = uncertain_multi.reshape(
        (uncertain_multi.shape[0]*uncertain_multi.shape[1], uncertain_multi.shape[2]),
        order='F')
    
    # All height bin levels [m] above radar
    np.savetxt(resdir + 'heights.txt', height_vector)
    # Modal properties, data and descriptions 
    f_modal = resdir + 'modal_%Y%m%d_%H%M%S.txt' 
    modal_file = time.strftime(f_modal)
    np.savetxt(modal_file, modal_reshaped)
    # Multimodal properties
    f_multimodal = resdir + 'multimodal_%Y%m%d_%H%M%S.txt' 
    multimodal_file = time.strftime(f_multimodal)
    np.savetxt(multimodal_file, multimodal_reshaped)
    # Uncertainties of modal properties
    f_modal_unc = resdir + 'uncertainty_modal_%Y%m%d_%H%M%S.txt' 
    modal_unc_file = time.strftime(f_modal_unc)
    np.savetxt(modal_unc_file, uncertainty_modal_reshaped)
    # Uncertainties of multimodal properties
    f_multimodal_unc = resdir + 'uncertainty_multimodal_%Y%m%d_%H%M%S.txt' 
    multimodal_unc_file = time.strftime(f_multimodal_unc)
    np.savetxt(multimodal_unc_file, uncertainty_multimodal_reshaped)  
            
    # Print confirmation for saved results
    saved_str = '%Y%m%d_%H%M%S'
    saved_time = time.strftime(saved_str)
    print('Multimodal analysis of DWD birdbath scan from '
          + saved_time + ' saved to directory ' + resdir + '.')
    
    
def save_reflectivity(spectral_reflectivities_data, time, resdir='./results/'):
    """
    Save results for spectrally resolved reflectivity data.
    
    Save data arrays (reflectivity spectra, power spectra, velocity vectors,
    mode indices) at time=datetime_object in .txt files in resdir directory. 
    """
    
    # Create directory for results if it does not exist
    os.makedirs(resdir, exist_ok=True)
    
    # Save individual datasets
    np.savetxt(
        resdir + 'reflectivity_spectra_' + time.strftime('%Y%m%d_%H%M%S') + '.txt',
        spectral_reflectivities_data['reflectivity_spectra'])
    np.savetxt(
        resdir + 'power_spectra_' + time.strftime('%Y%m%d_%H%M%S') + '.txt',
        spectral_reflectivities_data['power_spectra'])
    np.savetxt(
        resdir + 'velocity_vectors_' + time.strftime('%Y%m%d_%H%M%S') + '.txt',
        spectral_reflectivities_data['velocity_spectra'])
    np.savetxt(
        resdir + 'mode_indices_' + time.strftime('%Y%m%d_%H%M%S') + '.txt',
        spectral_reflectivities_data['mode_indices'])
            
    # Print confirmation for saved results
    saved_str = '%Y%m%d_%H%M%S'
    saved_time = time.strftime(saved_str)
    print('Spectral reflectivities of DWD birdbath scan from '
          + saved_time + ' saved to directory ' + resdir + '.')
    
    
def save_results(data, time, output_path='./results/'):
    """
    Save postprocessing results for DWD birdbath scan as .txt files.
    
    Save results for multimodal analysis 
    (and spectrally resolved reflectivity data, if selected in settings). 
    
    Args:
        data (dict):
            Dictionary of 2 dictionaries: multimodal analysis 
            and spectral reflectivities.
            
        time (str):
            Time string of birdbath scan.
            
        output_path (Path):
            Relative path of directory to save postprocessing results.
            
        
    Returns:
        Postprocessing results summarized in .txt files in output_path.
    """
    
    print('saving postprocessing results...')
    
    # Get time of birdbath scan
    birdbath_time = dt.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    
    # 1) Extract and save results of multimodal analysis
    multimodal_data = data['multimodal_analysis']
    save_multimodal(multimodal_data, birdbath_time, resdir=output_path)
    
    # 2) Extract and save results of spectral reflectivities, if they exist
    if data['spectral_zh'] is not None:
        reflectivity_data = data['spectral_zh']
        save_reflectivity(reflectivity_data, birdbath_time, resdir=output_path)
        