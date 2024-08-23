"""
Functions for identifying relevant modes in Doppler spectra.

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

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

from pybathspectra import clustering


def find_peakintervals(
        dfth_isolated, dfth_raw, height_level, filter_element=[3, 7],
        powerscale='dBlin', peak_prominence=8):
    """
    Find peak intervals in Doppler spectrum.
    
    Find velocity intervals where the peak(s) of individual mode(s) within the
    Doppler spectrum at a single height are located, using UniDip based on the
    Dip Test, or using a fixed peak_prominence.
    
    Args:
        dfth_isolated (DataFrame): 
            DataFrame of DFTh values [dB]. Isolated meteorological signal;
            invalid bins = np.nan.
            
        dfth_raw (DataFrame): 
            DataFrame of DFTh values [dB] before isolating weather signal.
        
        height_level (scalar):
            Height [m] above radar of analyzed Doppler spectrum. 
            
        filter_element (list):
            Number of [height bins, velocity bins] 
            used for filtering dfth_raw to obtain dfth_isolated.
            
        powerscale (str):
            Find peak intervals for Doppler spectra on linear or dB scale;
            'linear' is linear scale, 'dBlin' dB scale.
            Use 'dBlin' generally, as 'linear' option is not always valid.
            
        peak_prominence (int or float or None):
            Minimum peak-to-trough difference in Doppler spectra 
            for identifying a new peak.
            None if 'peak_finder' is not 'fixed'.
            
    Returns:
        output (tuple of various types):
            Output required fro further analysis, 
            e.g. peak intervals, velocity and dfth selections.
    """ 
    
    print('...finding peaks...')
    
    # Doppler spectra for analysis, where weather signal has been isolated
    dfth_interp = dfth_isolated.values
    velocity = dfth_isolated.index.astype(float)
    height = dfth_isolated.columns.astype(float)
    
    # Doppler spectra where weather signal has not been isolated
    dfth_mean = dfth_raw
    
    # Analysis per individual height bin
    height_anaspec = height_level
    
    # Window size used also for filtering and interpolation of spectra
    windowsize = filter_element[1]
    
    # 1) Use given peak prominence for 'fixed' peak finder, if selected
    if peak_prominence is not None:
        
        # Background noise, here minimum of values in isolated signal
        dfth_noise = np.nanmin(dfth_interp, axis=0)
        
        # Pick Doppler spectrum at selected height bin
        dfth_anaspec = dfth_interp[:, height==height_anaspec].squeeze()
        
        # Replace filtered measurements with noise estimate
        nan_dfth = np.isnan(dfth_anaspec)
        dfth_anaspec[nan_dfth] = dfth_noise[height == height_anaspec]
        # Unfiltered Doppler spectra; for scaling of reflectivity later on
        dfth_anaspec_II = dfth_mean.values[:, height==height_anaspec].squeeze()
        
        # Transform dB values to linear scale
        dfth_anaspec_lin = 10 ** (0.1 * dfth_anaspec)
        dfth_noise_lin = 10 ** (0.1 * dfth_noise)
        dfth_anaspec_II_lin = 10 ** (0.1 * dfth_anaspec_II)
        
        # Velocities as array, for code compatibility
        velocities_full = velocity.values
        
        # Remove irrelevant part of spectrum
        # check if all noise: if so, then no further analysis
        noise_check = np.argwhere(
            dfth_anaspec > dfth_noise[height==height_anaspec])
        if noise_check.size < 1:
            dfth_selspec = np.nan
            dfth_selspec_lin = np.nan
        # general case for further analysis
        else:
            idx_start = noise_check[0][0] - 1
            idx_end = noise_check[-1][-1] + 1
            dfth_selspec = dfth_anaspec[idx_start:idx_end+1]
            dfth_selspec_lin = dfth_anaspec_lin[idx_start:idx_end+1]
            
        # Smooth valid signals with Savitzky-Golay filter
        # (SG-polyorder > 1: better reproduces real peaks)
        # a) invalid case 1 (all noise, no valid data)
        if np.isnan(dfth_selspec).any() or np.isnan(dfth_selspec_lin).any():
            output = -999
        # b) invalid case 2 (not enough valid data)
        elif (len(dfth_selspec) < windowsize+2 or 
              len(dfth_selspec_lin) < windowsize+2):
            output = -999
        # c) valid case for further analysis
        else:
            if powerscale == 'dBlin':
                dfth_selspec_sg = savgol_filter(
                    dfth_selspec[1:-1], window_length=windowsize, polyorder=2)
                # Replace values < previously determined minimum
                minimum_condition = (dfth_selspec_sg < dfth_selspec.min())
                dfth_selspec_sg[minimum_condition] = dfth_selspec.min()
                # Linear-scale dfth values for spectrum sg-filtered in dB-space
                dfth_selspec_lin_sg = 10 ** (0.1 * dfth_selspec_sg)
            else:
                raise ValueError('Invalid powerscale in find_peakintervals().')
        
            # Velocity ranges of Doppler spectra selected for analysis
            velocity_selspec = velocities_full[idx_start:idx_end+1]
            velocity_selspec_sg = velocities_full[idx_start+1:idx_end]
            
            # Find peak indices based on prescribed peak prominence
            peaks, _ = find_peaks(dfth_selspec_sg, prominence=peak_prominence)
            
            # Cleanup: Exclude peaks at border of spectra
            for number, peak in enumerate(peaks):
                if (peak < 2) or (peak > len(dfth_selspec_sg)-3):
                    np.delete(peaks, number)
                
            # Catch special case: no mode -> choose max value as peak
            if peaks.shape[0] < 1:
                peaks = np.argwhere(dfth_selspec_sg==dfth_selspec_sg.max())[0]
            
            output = (peaks, velocity_selspec, velocity_selspec_sg,
                      dfth_selspec_lin, dfth_selspec_lin_sg,
                      dfth_noise_lin, dfth_anaspec_II_lin)
        
    # 2) Use 'adaptive' peak finder via UniDip clustering, if selected
    else:
        output = clustering.unidip_peaks(
            dfth_interp, dfth_mean, height, velocity,
            height_anaspec, windowsize, powerscale)
    
    return output

    
def identify_modes(
        velocity_selection, dfth_selection, unidip_intervals=[], 
        velocity_distribution=[], peaks=np.array([])):
    """
    Identify modes in Doppler spectrum.
    
    Derive mode(s) limits in Doppler spectrum at single height
    based on minima between neighboring peaks.
    
    Args:
        velocity_selection (array):
            Velocity values [m/s] of analyzed part of Doppler spectrum.
            
        dfth_selection (array):
            DFTh values [linear units] of analyzed part of Doppler spectrum
            after filtering, interpolation + cutting.
            
        unidip_intervals (array): 
            Indices of the different peak intervals found with UniDip algorithm. 
            Empty if 'peak_finder' was not set to 'adaptive'.
            
        velocity_distribution (array):
            Array of velocity values [m/s] for UniDip clustering.
            Empty if 'peak_finder' was not set to 'adaptive'.
            
        peaks (array):
            Array of indices where peak is located.
            Empty when 'peak_finder' is not 'fixed'.
            
    Returns:
        modes output (tuple):
            2 or 3 results, depending on 'peak_finder' setting.
    """   
    
    print('...identifying modes...')
    
    # 1) Modes for output from 'fixed' peak finder, if selected
    if peaks.size > 0:
        
        # Velocity range of meteorological signal
        velocity_selspec = velocity_selection
        # DFTh values corresponding to velocity range
        dfth_selspec_lin = dfth_selection
        
        # Identify modes for later calculations
        n_intervals = peaks.size
        dfth_max = np.ones((n_intervals,))
        vel_int_idx = np.ones((n_intervals, 2), dtype='int')
        if n_intervals < 1:
            raise ValueError('Number of peak intervals cannot'
                             ' be < 1 in identify_modes().')
        for count, peak_idx in enumerate(peaks):
            dfth_max_meas_lin = dfth_selspec_lin[peak_idx].squeeze()
            dfth_max[count] = dfth_max_meas_lin
        
        # a) Case 1: only single peak interval
        if n_intervals == 1:
            dfth_left = dfth_selspec_lin[:peaks[0]+1]
            vel_left_idx = np.argwhere(dfth_left == dfth_left.min())[-1][0]
            dfth_right = dfth_selspec_lin[peaks[0]:]
            r_con = (dfth_right == dfth_right.min())
            vel_right_idx = np.argwhere(r_con)[0][0] + peaks[0]
            # Peak intervals for later calculations of moments and parameters
            vel_int_idx[0,:] = np.array([vel_left_idx, vel_right_idx])
        
        # b) Case 2: multiple peak intervals
        elif n_intervals > 1:
            for count, peak_idx in enumerate(peaks):
                
                # Sub-case 1: left-most peak interval
                if count == 0:
                    dfth_left = dfth_selspec_lin[:peaks[count]+1]
                    l_con = (dfth_left == dfth_left.min())
                    p_cou = peaks[count]
                    p1_cou = peaks[count+1]
                    vel_left_idx = np.argwhere(l_con)[-1][0]
                    dfth_right = dfth_selspec_lin[p_cou:p1_cou+1]
                    r_con = (dfth_right == dfth_right.min())
                    vel_right_idx = np.argwhere(r_con)[0][0] + p_cou
                # Sub-case 2: right-most peak interval   
                elif count == n_intervals-1:
                    dfth_left = dfth_selspec_lin[peaks[count-1]:peaks[count]+1]
                    l_con = (dfth_left == dfth_left.min())
                    vel_left_idx = np.argwhere(l_con)[-1][0] + peaks[count-1]
                    dfth_right = dfth_selspec_lin[peaks[count]:]
                    r_con = (dfth_right == dfth_right.min())
                    vel_right_idx = np.argwhere(r_con)[0][0] + peaks[count]
                # Sub-case 3: peak interval between two other peak intervals
                else:
                    dfth_left = dfth_selspec_lin[peaks[count-1]:peaks[count]+1]
                    l_con = (dfth_left == dfth_left.min())
                    p_cou = peaks[count]
                    p1_cou = peaks[count+1]
                    vel_left_idx = np.argwhere(l_con)[-1][0] + peaks[count-1]
                    dfth_right = dfth_selspec_lin[p_cou:p1_cou+1]
                    r_con = (dfth_right == dfth_right.min())
                    vel_right_idx = np.argwhere(r_con)[0][0] + p_cou
                    
                # Peak intervals for later calculations
                vel_int_idx[count,:] = np.array([vel_left_idx, vel_right_idx])
        
        # If no intervals are found, which should not happen        
        else:
            raise ValueError('Number of peak intervals cannot'
                             ' be < 1 in identify_modes().')
        
        modes_output = n_intervals, vel_int_idx
        
    # 2) Then look at other case: 'adaptive' peak finder
    else:
    
        # Previously determined peak intervals with UniDip
        unidip_finals = unidip_intervals
        # Previously derived velocity distribution for application of UniDip
        vel_distrib_lin = velocity_distribution
        # Velocity range of meteorological signal
        velocity_selspec = velocity_selection
        # DFTh values corresponding to velocity range
        dfth_selspec_lin = dfth_selection
        
        
        # Identify peak boundaries and peaks for later calculations
        multi_shape = unidip_finals.shape
        n_intervals = multi_shape[0]
        vel_int_idx_unidip = np.ones((n_intervals, 2), dtype='int')
        dfth_max_idx = np.ones((n_intervals,), dtype='int')
        dfth_max = np.ones((n_intervals,))
        if n_intervals < 1:
            raise ValueError('Number of peak intervals cannot'
                             ' be < 1 in identify_modes().')
        # Identify peak values, indices of cutout spectra, and peak intervals
        for int_idx, int_vals in enumerate(unidip_finals):
            vel_min = vel_distrib_lin[int_vals[0]]
            vel_max = vel_distrib_lin[int_vals[1]-1]
            veldiff_min = np.abs(velocity_selspec - vel_min)
            veldiff_max = np.abs(velocity_selspec - vel_max)
            vel_min_bin = np.where(veldiff_min == veldiff_min.min())
            vel_max_bin = np.where(veldiff_max == veldiff_max.min())
            peak_int_idx = np.array([vel_min_bin[0][0], vel_max_bin[0][0]])
            d_range = dfth_selspec_lin[peak_int_idx[0]:peak_int_idx[1]+1]
            max_con = (dfth_selspec_lin == d_range.max())
            dfth_max_idx_lin = np.argwhere(max_con).squeeze()
            if dfth_max_idx_lin.size > 1:
                dfth_max_idx_lin = dfth_max_idx_lin[-1]
            dfth_max_meas_lin = dfth_selspec_lin[dfth_max_idx_lin].squeeze()
            vel_int_idx_unidip[int_idx, :] = peak_int_idx
            dfth_max_idx[int_idx] = dfth_max_idx_lin
            dfth_max[int_idx] = dfth_max_meas_lin
        # UniDip peak intervals
        vel_int_idx = np.ones((n_intervals, 2), dtype='int')
        
        # a) Case 1: only single peak interval
        if n_intervals == 1:
            dfth_left = dfth_selspec_lin[:dfth_max_idx[0]+1]
            vel_left_idx = np.argwhere(dfth_left == dfth_left.min())[-1][0]
            dfth_right = dfth_selspec_lin[dfth_max_idx[0]:]
            r_con = (dfth_right == dfth_right.min())
            vel_right_idx = np.argwhere(r_con)[0][0] + dfth_max_idx[0]
                            
            # Better peak intervals for later calculations
            vel_int_idx[0,:] = np.array([vel_left_idx, vel_right_idx])
        
        # b) Case 2: multiple peak intervals
        elif n_intervals > 1:
            for int_idx, int_vals in enumerate(unidip_finals):
                
                # Sub-case 1: left-most peak interval
                if int_idx == 0:
                    dfth_left = dfth_selspec_lin[:dfth_max_idx[int_idx]+1]
                    l_con = (dfth_left == dfth_left.min())
                    vel_left_idx = np.argwhere(l_con)[-1][0]
                    max_idx = dfth_max_idx[int_idx]
                    max1_idx = dfth_max_idx[int_idx+1]
                    dfth_right = dfth_selspec_lin[max_idx:max1_idx+1]
                    r_con = (dfth_right == dfth_right.min())
                    vel_right_idx = np.argwhere(r_con)[0][0] + max_idx
                # Sub-case 2: right-most peak interval   
                elif int_idx == n_intervals-1:
                    max_idx = dfth_max_idx[int_idx]
                    max1m_idx = dfth_max_idx[int_idx-1]
                    dfth_left = dfth_selspec_lin[max1m_idx:max_idx+1]
                    l_con = (dfth_left == dfth_left.min())
                    vel_left_idx = np.argwhere(l_con)[-1][0] + max1m_idx
                    dfth_right = dfth_selspec_lin[max_idx:]
                    r_con = (dfth_right == dfth_right.min())
                    vel_right_idx = np.argwhere(r_con)[0][0] + max_idx
                # Sub-case 3: peak interval between two other peak intervals
                else:
                    max_idx = dfth_max_idx[int_idx]
                    max1m_idx = dfth_max_idx[int_idx-1]
                    max1_idx = dfth_max_idx[int_idx+1]
                    dfth_left = dfth_selspec_lin[max1m_idx:max_idx+1]
                    l_con = (dfth_left == dfth_left.min())
                    vel_left_idx = np.argwhere(l_con)[-1][0] + max1m_idx
                    dfth_right = dfth_selspec_lin[max_idx:max1_idx+1]
                    r_con = (dfth_right == dfth_right.min())
                    vel_right_idx = np.argwhere(r_con)[0][0] + max_idx
                
                # Better peak intervals for later calculations
                vel_int_idx[int_idx,:] = np.array([vel_left_idx,
                                                   vel_right_idx])
                    
        # If no intervals are found with UniDip, which should not happen        
        else:
            raise ValueError('Number of peak intervals cannot'
                             ' be < 1 in identify_modes().')
        
        modes_output = n_intervals, vel_int_idx, vel_int_idx_unidip
    
    return modes_output


def find_modes(find_peakintervals_output):
    """
    Identify relevant modes in weather signal at single height.
    
    Args:
        find_peakintervals_output (tuple):
            Output from find_peakintervals().
            Length of the tuple distinguishes different selected settings. 
            
            
    Returns:
        modes, modes_savgol (??):
            Modes found in unsmoothed Doppler spectrum and in 
            (2nd order Savitzky-Golay) smoothed Doppler spectrum.      
    """
    
    print('...finding modes in birdbath spectra...')
    
    
    # Extract correct input data and determine modes
    # 1) 'fixed' peak finder
    if len(find_peakintervals_output) == 7:
        (peaks_collection, velocity_selspec, velocity_selspec_sg,
         dfth_selspec_lin, dfth_selspec_lin_sg, dfth_noise_lin,
         dfth_anaspec_II_lin) = find_peakintervals_output
        modes = identify_modes(
            velocity_selspec, dfth_selspec_lin, peaks=peaks_collection+1)
        modes_savgol = identify_modes(
            velocity_selspec_sg, dfth_selspec_lin_sg, peaks=peaks_collection)
    
    # 2) 'adaptive' peak finder
    elif len(find_peakintervals_output) == 8:   
        (unidip_finals, vel_distrib, velocity_selspec, velocity_selspec_sg,
         dfth_selspec_lin, dfth_selspec_lin_sg, dfth_noise_lin,
         dfth_anaspec_II_lin) = find_peakintervals_output
        modes = identify_modes(
            velocity_selspec, dfth_selspec_lin,
            unidip_intervals=unidip_finals, velocity_distribution=vel_distrib)
        modes_savgol = identify_modes(
            velocity_selspec_sg, dfth_selspec_lin_sg,
            unidip_intervals=unidip_finals, velocity_distribution=vel_distrib)
        
    else:
        raise ValueError('Error in find_peakintervals() output.')
    
    return modes, modes_savgol


def mode_properties(
        mode_indices, height_level, height_vector, df_moments,
        velocity_selection, dfth_selection, dfth_raw_avglin,
        background_noise, powerscale='dBlin', reflectivity_scale='Zh',
        thresholding_type='fixed'):
    """
    Calculate modal and multimodal characteristic properties.
    
    Calculate moments and other characteristics of individual mode(s)
    and determine characteristics that describe the relation between multiple
    modes if multiple modes are found. Do this either for signal as is
    or signal smoothed by Savitzky Golay filter: the difference between the two
    should give a rough uncertainty estimate in each determined characteristic.
    
    Args:
        mode_indices (array): 
            Indices of individual modes found in Doppler spectrum 
            (rows: different mode(s); cols: [start, end] index). 
        
        height_level (scalar):
            Height [m] above radar of analyzed Doppler spectrum.
            
        height_vector (DataFrame columns):
            Vector of all height bin values [m] of all Doppler spectra data
            (different from all height bins for df_moments below).  
        
        df_moments (DataFrame):
            Birdbath moments from load_birdbath(), for selecting 
            reflectivity_scale moment.
        
        velocity_selection (array):
            Velocity values (m/s) of analyzed part of Doppler spectrum.
            
        dfth_selection (array):
            DFTh values (linear units) of analyzed part of Doppler spectrum 
            after filtering, interpolation + cutting.
            
        dfth_raw_avglin (array):
            DFTh spectra (linear units) without filtering, i.e., 
            averaged output from signal processor before filtering. 
            
        background_noise (array):
            DFTh residual signal (linear units) away from 
            meteorological signal, for all heights.
            
        powerscale (str):
             Calculate mode properties from Doppler spectra based on 
             only linear ('linear', 'dBlin') units.
        
        reflectivity_scale (str):
            Reflectivity 'Zh' or 'UZh' for transforming power to reflectivity.
            From given df_moments.
        
        thresholding_type (str):
            Type of thresholding applied. 'adaptive' thresholding by HDBSCAN
            clustering, 'fixed' preselected thresholds, 
            or 'none_expanded', where there is no thresholding applied 
            but Doppler velocity range is expanded for manual dealiasing.
            Here needed for scaling power-to-reflectivity transform correctly.
            
            
    Returns:
        each_mode, multi_modes (tuples):
            2 tuples of 11 modal and 7 multimodal properties.
    """   
    
    # Number of individual modes found in Doppler spectrum
    n_intervals = mode_indices.shape[0]
    # Intervals of velocity indices for individual mode(s)
    vel_int_idx = mode_indices
    # All heights above radar for all Doppler spectra
    height = height_vector
    # Height bin value selected for analysis
    height_anaspec = height_level
    # Velocity range of meteorological signal
    velocity_selspec = velocity_selection
    # DFTh values corresponding to velocity range
    dfth_selspec_lin = dfth_selection
    # DFTh full spectrum is averaged signal processor output
    dfth_anaspec_II_lin = dfth_raw_avglin
    # Background noise = residual signal away from weather signal
    dfth_noise_lin = background_noise
    
    # Initialize results, one value for each mode
    # Moments
    Zh_lin = np.zeros((n_intervals,))
    dBZh = np.zeros((n_intervals,))
    v_mean = np.zeros((n_intervals,))
    v_std = np.zeros((n_intervals,))
    v_skew = np.zeros((n_intervals,))
    v_kurt = np.zeros((n_intervals,))
    # Other characteristics
    v_mode = np.zeros((n_intervals,))
    v_med = np.zeros((n_intervals,))
    v_skew_mode = np.zeros((n_intervals,))
    v_skew_med = np.zeros((n_intervals,))
    dfth_vec_max = np.zeros((n_intervals,))
    mode_power_sca = np.zeros((n_intervals,))
    
    # Estimate background noise level in DFTh signal
    noise_level = dfth_noise_lin[height == height_anaspec]
    # Get Zh in dB units for full spectrum at correct height
    Zh_scaling = pd.IndexSlice[:,reflectivity_scale]
    valid_height = (df_moments.columns == height_anaspec)
    Zh_tot_dBZ = df_moments.loc[Zh_scaling, valid_height].values.squeeze()
    Zh_tot_lin = 10 ** (0.1 * Zh_tot_dBZ)
    #Calculate non-normalized power (= (area) sum) for unfiltered spectrum
    dfth_anaspec_full_lin = dfth_anaspec_II_lin.copy() - noise_level
    dfth_anaspec_full_lin[dfth_anaspec_full_lin < 0] = 0
    full_power_sca = np.sum(dfth_anaspec_full_lin)
    
    # Analysis based on linear units of Doppler power
    if powerscale == 'linear' or powerscale == 'dBlin':
        
        for mode in range(n_intervals):
            
            # Index range for later calculations, see below
            slice_start = vel_int_idx[mode,0]
            slice_end = vel_int_idx[mode,1]
            # Velocity spectrum vectors (m/s)
            vels_vec = velocity_selspec[slice_start:slice_end]
            # DFTh spectrum (linear units), corrected for residual noise
            dfth_vec = dfth_selspec_lin[slice_start:slice_end] - noise_level
            dfth_vec[dfth_vec < 0] = 0
                
            
            # Mode maxima
            dfth_vec_max[mode] = dfth_vec.max()
            # Power of spectral mode = non-normalized 0th order moment
            mode_power_sca[mode] = np.sum(dfth_vec)
            
            # derived from 0th order moment: Zh
            # e.g.: Zh[mode] = dft_power(mode) / dft_power(total) * Zh(total)
            # Different scaling due to expansion of Doppler spectra
            if thresholding_type == 'none_expanded':
                Zh_lin[mode] = (mode_power_sca[mode] / full_power_sca
                                * Zh_tot_lin * 3)
            else:
                Zh_lin[mode] = (mode_power_sca[mode] / full_power_sca
                                * Zh_tot_lin)
            dBZh[mode] = 10 * np.log10(Zh_lin[mode])
            
            # 1st order moment = power-weighted mean velocity v_mean (m/s)
            v_mean_int = dfth_vec * vels_vec
            v_mean[mode] = np.sum(v_mean_int) / mode_power_sca[mode]
            
            # 2nd order moment = standard deviation = width of mode (m/s)
            momdiff = (vels_vec - v_mean[mode]) ** 2
            v_var_int = dfth_vec * momdiff
            v_var = np.sum(v_var_int) / mode_power_sca[mode] 
            v_std[mode] = v_var ** 0.5
            
            # 3rd order moment = normalized skewness (-)
            momdiff = (vels_vec - v_mean[mode]) ** 3
            v_skew_int = dfth_vec * momdiff
            v_skew[mode] = (np.sum(v_skew_int) / mode_power_sca[mode]
                            / v_std[mode]**3)
            
            # 4th order moment = normalized kurtosis (-) 
            momdiff = (vels_vec - v_mean[mode]) ** 4
            v_kurt_int = dfth_vec * momdiff
            v_kurt[mode] = (np.sum(v_kurt_int) / mode_power_sca[mode]
                            / v_std[mode]**4)
        
            # (Power) mode = velocity at maximum power (m/s)
            mode_vels = vels_vec[dfth_vec == dfth_vec.max()]
            if mode_vels.size > 1:
                mode_vels = mode_vels[-1]
            v_mode[mode] = mode_vels
            
            
            # (Power) median = velocity (m/s) splitting power in equal halfs
            dfth_cumsum = np.cumsum(dfth_vec)
            dfth_power_half = dfth_cumsum[-1] / 2
            dfth_power_diff = np.abs(dfth_cumsum - dfth_power_half)
            median_vels = vels_vec[dfth_power_diff == dfth_power_diff.min()]
            if median_vels.size > 1:
                median_vels = median_vels[-1]
            v_med[mode] = median_vels
        
            # (Pearson) mode skewness (-)
            v_skew_mode[mode] = (v_mean[mode] - v_mode[mode]) / v_std[mode]
            
            # Median skewness = Nonparametric skew (-)
            v_skew_med[mode] = (v_mean[mode] - v_med[mode]) / v_std[mode]
        
        # Percentage of total reflectivity contained in weather signal [%]
        Zh_percentage = np.array([100 * Zh_lin.sum() / Zh_tot_lin])
    
    
        # Calculate characteristics relating neighboring spectral modes
        if n_intervals > 1:
            
            # (Multi)modal ratio (dB) gives dominant regime
            vrp_log = np.log10(mode_power_sca) - np.log10(mode_power_sca.max())
            vrm_log = np.log10(dfth_vec_max) - np.log10(dfth_vec_max.max())
            v_rat_pow = 10 * vrp_log
            v_rat_mode = 10 * vrm_log
            
            # Other characteristics, here only one value for 2 successive modes
            v_sep_mean = np.zeros((n_intervals-1,))
            v_sep_mean_norm = np.zeros((n_intervals-1,))
            v_sep_mode = np.zeros((n_intervals-1,))
            v_amp_mode = np.zeros((n_intervals-1,))
            v_amp_mode_alt = np.zeros((n_intervals-1,))
            
            for dip in range(n_intervals-1):
                # Modes to left and right of the trough
                pre = dip
                nex = dip + 1
                # (Bi)modal separation (-) or (m/s)
                v_sep_mean[dip] = v_mean[nex] - v_mean[pre]
                v_sep_scaling = 2*v_std[pre] + 2*v_std[nex]
                v_sep_mean_norm[dip] = v_sep_mean[dip] / v_sep_scaling
                v_sep_mode[dip] = v_mode[nex] - v_mode[pre]
                # (Bi)modal amplitude (-) or modified version (dB) 
                # This gives effectiveness of mixing of physical processes 
                amp_lower = np.min([dfth_vec_max[pre], dfth_vec_max[nex]])
                amp_antimode = (dfth_selspec_lin[vel_int_idx[pre,1]]
                                - noise_level)
                if amp_antimode <= 0:
                    amp_antimode = noise_level
                    amp_lower_noise = amp_lower + noise_level
                else:
                    amp_lower_noise = amp_lower
                v_amp_mode[dip] = (amp_lower_noise - amp_antimode) / amp_lower
                vam_log = np.log10(amp_antimode) - np.log10(amp_lower_noise)
                v_amp_mode_alt[dip] = 10 * vam_log
    
        else:
            v_rat_pow = np.nan
            v_rat_mode = np.nan
            v_sep_mean = np.nan
            v_sep_mean_norm = np.nan
            v_sep_mode = np.nan
            v_amp_mode = np.nan
            v_amp_mode_alt = np.nan
        
    else:
        raise ValueError('Invalid powerscale entered for mode_properties().')
    
    
    each_mode = (
        dBZh, v_mean, v_std, v_skew,
        v_kurt, v_skew_med, Zh_lin, v_mode,
        v_med, v_skew_mode, Zh_percentage,
    )
    
    multi_modes = (
        v_rat_mode, v_sep_mean_norm, v_amp_mode,
        v_amp_mode_alt, v_rat_pow, v_sep_mean,
        v_sep_mode,
    )
    
    return each_mode, multi_modes


def calculate_mode_properties(
        find_modes_output, find_peakintervals_output, height_level,
        height_bins, birdbath_moments, power='dBlin', reflectivity='Zh',
        thresholding='fixed', uncertainty_savgol=False):
    """
    Calculate modal and multimodal properties for identified modes.
    
    Args:
        find_modes_output (tuple):
            Output from find_modes(). Either smoothed or unsmoothed spectra.
            Length of the tuple distinguishes different selected settings. 
            
        find_peakintervals_output (tuple):
            Output from find_peakintervals().
            Length of the tuple distinguishes different selected settings.
        
        height_level (float):
            Height (above radar) [m] for current analyzed radar bin.
            
        height_bins (DataFrame columns):
            Vector of all height bin values [m] of all Doppler spectra data 
            (different from all height bins for birdbath_moments below).
            
        birdbath_moments (DataFrame):
            DataFrame of DWD birdbath scan moment data from load_birdbath().
            
        power (str):
            Power mode for analysis. Calculate mode properties from Doppler
            spectra based only on linear ('linear', 'dBlin') units.
            
        reflectivity (str):
            Reflectivity 'Zh' or 'UZh' for transforming power to reflectivity.
        
        thresholding (str):
            Type of thresholding applied. 'adaptive' thresholding by HDBSCAN
            clustering, 'fixed' preselected thresholds, 
            or 'none_expanded', where there is no thresholding applied 
            but Doppler velocity range is expanded for manual dealiasing.
            Here needed for scaling power-to-reflectivity transform correctly.
            
        uncertainty_savgol (bool):
            If 'True', mode properties are calculated for Savitzky-Golay
            smoothed spectra, which is needed if an uncertainty estimate
            is desired.
            
            
    Returns:
        modal, multimodal (tuples):
            2 tuples of 11 modal and 7 multimodal properties.
    """
    
    print('...calculating modal and multimodal properties...')
    
    
    # Extract correct input data
    # 1) 'fixed' peak finder
    if len(find_peakintervals_output) == 7:
        (peaks_collection, velocity_selspec, velocity_selspec_sg,
         dfth_selspec_lin, dfth_selspec_lin_sg, dfth_noise_lin,
         dfth_anaspec_II_lin) = find_peakintervals_output
        n_intervals, vel_int_idx = find_modes_output
    # 2) 'adaptive' peak finder
    elif len(find_peakintervals_output) == 8:   
        (unidip_finals, vel_distrib, velocity_selspec, velocity_selspec_sg,
         dfth_selspec_lin, dfth_selspec_lin_sg, dfth_noise_lin,
         dfth_anaspec_II_lin) = find_peakintervals_output
        n_intervals, vel_int_idx, vel_int_idx_unidip = find_modes_output
    else:
        raise ValueError('Error in find_modes() output.')
        
    # Calculate mode properties either of smoothed or unsmoothed spectra
    if uncertainty_savgol:
        modal, multimodal = mode_properties(
            vel_int_idx, height_level, height_bins, birdbath_moments,
            velocity_selspec_sg, dfth_selspec_lin_sg, dfth_anaspec_II_lin,
            dfth_noise_lin, powerscale=power, reflectivity_scale=reflectivity,
            thresholding_type=thresholding)
    else:
        modal, multimodal = mode_properties(
            vel_int_idx, height_level, height_bins, birdbath_moments,
            velocity_selspec, dfth_selspec_lin, dfth_anaspec_II_lin,
            dfth_noise_lin, powerscale=power, reflectivity_scale=reflectivity,
            thresholding_type=thresholding)
        
    return modal, multimodal


def substitute_results(
        heightlevel_results, height_level, full_results, height_vector,
        dimension=3):
    """
    Substitute results determined at single height into full array.
    
    Insert results (modal incl. peaks_flag, multimodal, uncertainties)
    from analysis at a specific height level into full array of final
    results at all height levels.
    Depends on the dimension of the full results array. 
    """
    
    final_results = full_results.copy()
    
    if dimension == 3:
        for idx_result, result in enumerate(heightlevel_results):
            height_found = (height_vector == height_level)
            length = np.size(result)
            final_results[idx_result, height_found, :length] = result        
    elif dimension == 2:
        height_found = (height_vector == height_level)
        length = np.size(heightlevel_results)
        final_results[height_found, :length] = heightlevel_results        
    else:
         raise ValueError('Dimension in substitute_results() not supported')  
            
    return final_results


def estimate_uncertainties(
        modal, multimodal, modal_sg, multimodal_sg, 
        velocity_resolution, mode='absolute'):
    """
    Estimate uncertainties of calculated modal and multimodal properties.
    
    Estimate uncertainties of calculated characteristics of individual
    modes ('modal') and of multimodal characteristics ('multimodal') as
    difference between properties calculated for radar spectrum as is (-)
    and smoothed by Savitzky Golay filter ('sg').
    Output uncertainties in units of calculated characteristics ('absolute').
    """
    
    # Differences between characteristics for signal as is and smoothed signal
    uncertain_modal = tuple(x-y for x,y in zip(modal,modal_sg))
    uncertain_multimodal = tuple(x-y for x,y in zip(multimodal,multimodal_sg))
    # At least for v_mode and v_median -> effect of finite velocity resolution
    mode_res = (np.abs(uncertain_modal[7]) < velocity_resolution)
    median_res = (np.abs(uncertain_modal[8]) < velocity_resolution)
    uncertain_modal[7][mode_res] = velocity_resolution
    uncertain_modal[8][median_res] = velocity_resolution
    
    # Absolute uncertainties
    if mode == 'absolute':
        uncertain_modal = tuple(abs(x) for x in uncertain_modal)
        uncertain_multimodal = tuple(abs(x) for x in uncertain_multimodal)
    else:
        raise ValueError('Invalid mode entered for estimate_uncertainty().')
    
    return uncertain_modal, uncertain_multimodal


def power_to_reflectivity(
        mode_indices, height_level, velocity_selection, dfth_selection,
        dfth_raw_avglin, df_moments, reflectivity_scale='Zh',
        thresholding_type='fixed'):
    """
    Estimate spectrally resolved reflectivity from Doppler power spectrum.
    
    Transform analyzed Doppler (power) spectrum to reflectivity spectrum
    at analyzed height_level (and collect some relevant data for later).
    
    Args:
        mode_indices: 
            Indices of individual modes found in Doppler spectrum 
            (rows: different mode(s); cols: [start, end] index). 
        
        height_level:
            Height [m] above radar of analyzed Doppler spectrum.
            
        velocity_selection (array):
            Velocity values (m/s) of analyzed part of Doppler spectrum.
            
        dfth_selection (array):
            DFTh values (LINEAR UNITS) of analyzed part of Doppler spectrum 
            after filtering, interpolation + cutting.
            
        dfth_raw_avglin (array):
            DFTh spectra (LINEAR UNITS) without own clutter filter, i.e., 
            averaged output from signal processor BEFORE filtering. 
            Full velocity range, not only analyzed part.
        
        df_moments (DataFrame):
            Birdbath moments from load_birdbath(), for selecting 
            reflectivity_scale moment.
        
        reflectivity_scale (str):
            Reflectivity 'Zh' or 'UZh' for transforming power to reflectivity.
            From given df_moments.
        
        thresholding_type (str):
            Type of thresholding applied. 'adaptive' thresholding by HDBSCAN
            clustering, 'fixed' preselected thresholds, 
            or 'none_expanded', where there is no thresholding applied 
            but Doppler velocity range is expanded for manual dealiasing.
            Here needed for scaling power-to-reflectivity transform correctly. 
            
            
    Returns:
        tuple of 4 outputs:
            Reflectivity spectrum, power spectrum, velocity vector,
            mode indices.
    """   
    
    # Intervals of velocity indices for individual mode(s)
    vel_int_idx = mode_indices
    # Height bin value selected for analysis
    height_anaspec = height_level
    # Velocity range of meteorological signal
    velocity_selspec = velocity_selection
    # DFTh values corresponding to velocity range
    dfth_selspec_lin = dfth_selection
    # DFTh full spectrum is averaged signal processor output
    dfth_anaspec_II_lin = dfth_raw_avglin
    
    # Get Zh in dB units for full spectrum at correct height
    Zh_scaling = pd.IndexSlice[:,reflectivity_scale]
    valid_height = (df_moments.columns == height_anaspec)
    Zh_tot_dBZ = df_moments.loc[Zh_scaling, valid_height].values.squeeze()
    Zh_tot_lin = 10 ** (0.1 * Zh_tot_dBZ)
    #Calculate non-normalized power (= (area) sum) for full spectrum
    dfth_anaspec_full_lin = dfth_anaspec_II_lin.copy()
    dfth_anaspec_full_lin[dfth_anaspec_full_lin < 0] = 0
    full_power_sca = np.sum(dfth_anaspec_full_lin)
        
    # Calculate linear reflectivities for each bin in selcted power spectrum
    dfth_vec = dfth_selspec_lin.copy()
    dfth_vec[dfth_vec < 0] = 0
    # Different scaling due to expansion of Doppler spectra
    if thresholding_type == 'none_expanded':
        Zh_lin_spectrum = dfth_vec / full_power_sca * Zh_tot_lin * 3
    else:
        Zh_lin_spectrum = dfth_vec / full_power_sca * Zh_tot_lin
    
    # Collect results here
    reflectivity_spectrum = Zh_lin_spectrum
    power_spectrum = dfth_vec
    velocity_vector = velocity_selspec
    mode_idx = vel_int_idx.ravel()
    
    return reflectivity_spectrum, power_spectrum, velocity_vector, mode_idx


def determine_reflectivity_spectrum(
        find_modes_output, find_peakintervals_output, height_level,
        birdbath_moments, reflectivity='Zh', thresholding='fixed'):
    """
    Estimate spectrally resolved reflectivities from Doppler power spectrum.
    
    Args:
        find_modes_output (tuple):
            Output from find_modes(). Either smoothed or unsmoothed spectra.
            Length of the tuple distinguishes different selected settings. 
            
        find_peakintervals_output (tuple):
            Output from find_peakintervals().
            Length of the tuple distinguishes different selected settings.
        
        height_level (float):
            Height (above radar) [m] for current analyzed radar bin.
            
        birdbath_moments (DataFrame):
            DataFrame of DWD birdbath scan moment data from load_birdbath().
            
        reflectivity (str):
            Reflectivity 'Zh' or 'UZh' for transforming power to reflectivity.
        
        thresholding (str):
            Type of thresholding applied. 'adaptive' thresholding by HDBSCAN
            clustering, 'fixed' preselected thresholds, 
            or 'none_expanded', where there is no thresholding applied 
            but Doppler velocity range is expanded for manual dealiasing.
            Here needed for scaling power-to-reflectivity transform correctly.
            
            
    Returns:
        reflectivity_output (tuple):
            Tuple of 4 outputs; reflectivity spectrum, power spectrum,
            velocity vector, mode indices.
    """
    
    print('...estimating reflectivity spectrum...')
    
    
    # Extract correct input data
    # 1) 'fixed' peak finder
    if len(find_peakintervals_output) == 7:
        (peaks_collection, velocity_selspec, velocity_selspec_sg,
         dfth_selspec_lin, dfth_selspec_lin_sg, dfth_noise_lin,
         dfth_anaspec_II_lin) = find_peakintervals_output
        n_intervals, vel_int_idx = find_modes_output
    # 2) 'adaptive' peak finder
    elif len(find_peakintervals_output) == 8:   
        (unidip_finals, vel_distrib, velocity_selspec, velocity_selspec_sg,
         dfth_selspec_lin, dfth_selspec_lin_sg, dfth_noise_lin,
         dfth_anaspec_II_lin) = find_peakintervals_output
        n_intervals, vel_int_idx, vel_int_idx_unidip = find_modes_output
    else:
        raise ValueError('Error in find_modes() output.')
        
    # Calculate reflectivity spectra from uncalibrated power spectra
    reflectivity_output = power_to_reflectivity(
        vel_int_idx, height_level, velocity_selspec,
        dfth_selspec_lin, dfth_anaspec_II_lin, birdbath_moments,
        reflectivity_scale=reflectivity, thresholding_type=thresholding)
    
    return reflectivity_output


def multimodal_analysis(weather_spectra, birdbath_data, modes_settings):
    """
    Multimodal analysis of weather signal of birdbath Doppler spectra.
    
    Args:
        weather_spectra (dict):
            Dictionary of two DataFrames: 'isolated_weather' contains 
            filtered Doppler spectra, 'radar_measured' unfiltered spectra 
            (but averaged over all 15 rays), and 1 metadata key.
        
        birdbath_data (dict):
            Dictionary of 2 DataFrames: 'birdbath_spectra' and 
            'birdbath_moments', and 2 metadata keys included directly
            for easier further processing.
        
        modes_settings (dict):
            8 keys for setting options of multimodal analysis,
            in addition to 1+2 metadata keys in data dictionaries.
            
            
    Returns:
        postprocessing_results (dict):
            Dictionary of results from multimodal analysis (dict) and 
            results for spectrally resolved reflectivities (dict).
    """
    
    print('multimodal analysis of birdbath spectra...')
    
    # Extract individual settings required for multimodal analysis
    thresholding_flavor = weather_spectra['thresholding']
    box_roi = birdbath_data['structuring_element']
    peak_finder = modes_settings['peak_finder']
    peak_to_trough = modes_settings['peak_to_trough']
    maximum_modes = modes_settings['maximum_modes']
    reflectivity_spectra = modes_settings['reflectivity_spectra']
    maximum_height = modes_settings['analysis_maxheight']
    power_mode = modes_settings['power_mode']
    reflectivity_moment = modes_settings['reflectivity_scale']
    uncertainty_estimate = modes_settings['estimate_uncertainty']
    
    # Select method for finding peaks in Doppler spectra
    if peak_finder == 'fixed':
        peak_condition = peak_to_trough
    elif peak_finder == 'adaptive':
        peak_condition = None 
    else:
        raise ValueError('Invalid peak_finder in multimodal_analysis().')
    
    # Doppler data for identifying and analyzing modes at each height bin
    birdbath_moments = birdbath_data['birdbath_moments']
    isolated_weather = weather_spectra['isolated_weather']
    radar_measured = weather_spectra['radar_measured']
    height_bins = isolated_weather.columns
    velocity_bins = isolated_weather.index
    velocity_resolution = velocity_bins[1] - velocity_bins[0]
    
    # Initialize final results
    most_values = maximum_modes
    # 11 (+ 1 flag) characteristics for each mode and height
    modal_final = np.full((12, len(height_bins), most_values),
                          np.nan)
    # 7 multimodal characteristics
    multimodal_final = np.full((7, len(height_bins), most_values),
                               np.nan)
    # Uncertainty estimates for modal and multimodal characteristics
    uncertainty_modal_final = np.full((11, len(height_bins), most_values),
                                      np.nan)
    uncertainty_multimodal_final = np.full_like(multimodal_final, np.nan)
    # Initialize spectrally resolved reflectivity data, if selected
    if reflectivity_spectra:
        # Full Doppler (power) spectra (all heights * spectra length)
        power_spectra_final = np.full((isolated_weather.T.shape), np.nan)
        # Full reflectivity spectra estimated from power spectra
        reflectivity_spectra_final = np.full(power_spectra_final.shape, np.nan)
        # Doppler velocity vectors for reflectivity spectra
        velocity_vectors_final = np.full(power_spectra_final.shape, np.nan)
        # Start and end index of each mode at all heights, flattened
        mode_idx_final = np.full((len(height_bins), most_values*2), np.nan)
    
    # Loop over all heights for multimodal analysis
    for level in height_bins[height_bins <= maximum_height]:
        
        # Find peaks and extract peak info needed for further processing
        peaks_output = find_peakintervals(
            isolated_weather, radar_measured, level, filter_element=box_roi,
            powerscale=power_mode, peak_prominence=peak_condition)
        
        # Find modes in unsmoothed and smoothed spectra (-> uncertainty)
        # case a) Not enough valid velocity bins: no further processing
        if np.size(peaks_output) == 1:
            continue
        # case b) Valid Doppler spectrum: analyze modes in postprocessing
        modes, modes_savgol = find_modes(peaks_output)

        # Calculate modal and multimodal properties
        modal, multimodal = calculate_mode_properties(
            modes, peaks_output, level, height_bins, 
            birdbath_moments, power=power_mode,
            reflectivity=reflectivity_moment,
            thresholding=thresholding_flavor,
            uncertainty_savgol=False)
        # Substitute current results at correct height of full arrays
        modal_final = substitute_results(
            modal, level, modal_final,
            height_bins, dimension=3)
        multimodal_final = substitute_results(
            multimodal, level, multimodal_final,
            height_bins, dimension=3)
        
        # Uncertainty estimates, if selected
        if uncertainty_estimate:
            modal_savgol, multimodal_savgol = calculate_mode_properties(
                modes_savgol, peaks_output, level, height_bins,
                birdbath_moments, power=power_mode,
                reflectivity=reflectivity_moment,
                thresholding=thresholding_flavor,
                uncertainty_savgol=True)
            uncertainty_modal, uncertainty_multimodal = estimate_uncertainties(
                modal, multimodal, modal_savgol, multimodal_savgol,
                velocity_resolution, mode='absolute')
            # Substitute current results at correct height of full arrays
            uncertainty_modal_final = substitute_results(
                uncertainty_modal, level, uncertainty_modal_final,
                height_bins, dimension=3)
            uncertainty_multimodal_final = substitute_results(
                uncertainty_multimodal, level, uncertainty_multimodal_final,
                height_bins, dimension=3)
        
        # Full reflectivity spectra etc, if selected
        if reflectivity_spectra:
            spectral_reflectivity_data = determine_reflectivity_spectrum(
                modes, peaks_output, level, birdbath_moments,
                reflectivity=reflectivity_moment,
                thresholding=thresholding_flavor)
            # Substitute current results at correct height of full arrays
            (reflectivity_spectrum, power_spectrum,
             velocity_vector, mode_idx) = spectral_reflectivity_data
            reflectivity_spectra_final = substitute_results(
                reflectivity_spectrum, level, reflectivity_spectra_final,
                height_bins, dimension=2)
            power_spectra_final = substitute_results(
                power_spectrum, level, power_spectra_final,
                height_bins, dimension=2)
            velocity_vectors_final = substitute_results(
                velocity_vector, level, velocity_vectors_final,
                height_bins, dimension=2)
            mode_idx_final = substitute_results(
                mode_idx, level, mode_idx_final,
                height_bins, dimension=2)
        
        # Combine results in dictionaries for compact output
        modal_property_names = [
            '(U)Zh [dBZ]', 'Vh mean [m/s]', 'Vh std [m/s]', 'Vh skewness [-]',
            'Vh kurtosis [-]', 'Vh median skewness [-]', '(U)Zh [mm6 m-3]',
            'Vh mode [m/s]', 'Vh median [-]', 'Vh mode skewness [-]',
            'fractional (U)Zh [%]', 'flag [-]',
        ]
        
        multimodal_property_names = [
            'multimodal ratio, max power [dB]',
            'bimodal separation, normalized [-]',
            'bimodal amplitude [-]', 'bimodal amplitude, dB scale [dB]',
            'multimodal ratio, integrated power [dB]',
            'bimodal separation [m/s]',
            'bimodal separation (of Vh mode) [m/s]',
        ]
        
        multimodal_results = dict(
            modal_properties=modal_final,
            multimodal_properties=multimodal_final,
            modal_uncertainties=uncertainty_modal_final,
            multimodal_uncertainties=uncertainty_multimodal_final,
            range_heights=height_bins.values,
            modal_names=modal_property_names,
            multimodal_names=multimodal_property_names,
            analysis_maxheight=maximum_height,
            estimate_uncertainty=uncertainty_estimate)
        if reflectivity_spectra:
            reflectivity_results = dict(
                reflectivity_spectra=reflectivity_spectra_final,
                power_spectra=power_spectra_final,
                velocity_spectra=velocity_vectors_final,
                mode_indices=mode_idx_final)
        else:
            reflectivity_results = None
            
        # Put everything together in one final dictionary
        postprocessing_results = dict(
            multimodal_analysis=multimodal_results,
            spectral_zh=reflectivity_results)
    
    return postprocessing_results
