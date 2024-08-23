"""
Functions for isolating weather signal in radar output of Dppler spectra.

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
import scipy.ndimage as nd

from pybathspectra import clustering


def smooth_average(df, filter_element=[3, 7]):
    """
    Calculate mean h-pol power, h-v pol power difference, and variability.
    
    Calculate per Doppler velocity and range bin the mean of Doppler spectra
    over all 15 rays in one birdbath scan, the power difference between h- and
    v-pol channels, and the variability of this power difference; 
    and smoothen results before filtering out non-meteorological contributions
    to the Doppler spectra later on. 
    
    Args:
        df (DataFrame): 
            Dataframe with all DFTh,v [dB] spectra read in by read_spectra. 
        
        filter_element (list):
            Number of [height, velocity] bins employed in filtering. 
            A good choice is [3, 7], in general.
            
            
    Returns:
        dfth_mean (DataFrame):
           Values of averaged Doppler spectra 
           (1 value for every velocity and height bin, i.e. only 1 ray
           instead of originally 15 rays).
           
        dfthv_difference, dfthv_variability (arrays):
            smoothened average power difference between h and v polarization,
            and the smoothed variability (standard deviation over all 15 rays)
            for the birdbath spectra.
    """
    
    # Difference DFTh - DFTv
    dfth = df.loc[pd.IndexSlice[:,:,'DFTh'],:]
    dftv_data = df.loc[pd.IndexSlice[:,:,'DFTv'],:].values
    diff_dft = dfth - dftv_data
    # Mean of h-pol channel and h-v differencess
    dfth_mean = dfth.groupby('velocity').mean()
    diff_dft_mean = diff_dft.groupby('velocity').mean()
    diff_dft_mean_abs = diff_dft_mean.abs()
    
    # Smoothing of filter data
    # In 1D (applied to each spectrum)
    windowsize = filter_element[1]
    diff_dft_rolling_std = diff_dft_mean.rolling(
        window=windowsize, min_periods=1, axis='rows', center=True).std(ddof=0)
    diff_dft_rolling_mean_abs = diff_dft_mean_abs.rolling(
        window=windowsize, min_periods=1, axis='rows', center=True).mean()
    # In 2D (combining multiple spectra at successive heights)
    strucel = filter_element
    dfthv_variability = nd.grey_closing(
        diff_dft_rolling_std.values, size=strucel)
    dfthv_difference = nd.grey_closing(
        diff_dft_rolling_mean_abs.values, size=strucel)
    
    return dfth_mean, dfthv_difference, dfthv_variability


def isolate_spectra(
        df, filter_element=[3, 7], filter_thresholds=[1.4, 1.4, 60.0],
        min_height=625, region_area=None, mask_high=[0.38, 0.5],
        mask_low=[0.38, 11], mask_type='h(v)', use_interpolated=True):
    """
    Separate weather signal from non-weather in Doppler spectra.
    
    Isolate the meteorological signal of the weather radar Doppler spectra.
    Includes individual steps of filtering static clutter, background noise,
    near-field ground clutter; interpolating filtered spectra to fill holes;
    and cutting out relevant region(s) of interest in height-velocity space. 
    
    Args:
        df (DataFrame): 
            Dataframe with all DFTh,v [dB] spectra read in by read_spectra. 
        
        filter_element (list):
            Number of [height, velocity] bins employed in filtering. 
            A good choice is [3, 7], in general.
            
        filter_thresholds (list or None):
            The 3 threshold values that should be used for filtering later on. 
            [threshold value for absolute difference between DFTh and DFTv, 
            the standard deviation of the difference, 
            DFTh threshold for selecting connected regions of interest]. 
            None means no filtering applied to isolate weather signal.
            
        min_height (int or float):
            Height [m] above radar of first usable range bin 
            (higher than excessive ground clutter).
            
        region_area (int): 
            Threshold for connected regions [pixel] in filtered dataframe
            to be kept for analysis. 
            Default is None, where only the largest connected region is kept.
            
        mask_high and mask_low (lists):
            Parameters for cutting spectra after expansion of velocity range;
            see mask_type (determined manually before this analysis).
        
        mask_type (str):
            mask type 'h(v)': height = a*velocity + b or 
            'v(h)': velocity = a*height + b.
        
        use_interpolated (bool):
            After spectra are isolated, use final interpolated values
            of isolated spectra as output or simply use isolated spectra
            as ROI to mask mean of measurement data (i.e. no interpolation).
            
            
    Returns:
        isolated_spectra (DataFrame):
             DataFrame of DFTh [dB] after filtering, interpolation,
             and cutting regions of intererst (invalid bins = np.nan),
             including corresponding velocty and height.
            
        measured_spectra (DataFrame):
            Dataframe of mean DFTh [dB] of radar measurements,
            i.e. before filtering, interpolation, and cutting regions of
            intererst, including corresponding velocty and height.
    """
    
    print('...isolating weather signal...')
    
    # Average h-pol Doppler spectra over 15 rays + calculate smoothend 
    # power difference h-v pol and the variability
    dfthv_vars = smooth_average(df, filter_element=[3, 7])
    dfth_mean, dfthv_difference, dfthv_variability = dfthv_vars
    
    # Extract velocities and heights
    velocity = dfth_mean.index.astype(float)
    height = dfth_mean.columns.astype(float)
    
    
    # If thresholds are given as input, use them 
    # for trying to filter out non-meteorological contributions
    if isinstance(filter_thresholds, list):
        
        # Filter condition
        condition = (dfthv_difference > filter_thresholds[0]) | \
                    (dfthv_variability > filter_thresholds[1])
        
        # Filter dfth_mean Doppler spectra
        dfth_filtered = dfth_mean.copy().values
        dfth_filtered[condition] = np.nan
        
        # Remove ground clutter at low heights above radar
        height_min = min_height
        dfth_filtered[:, height<height_min] = np.nan
        
        # Interpolation of filtered dfth to recover potentially deleted parts
        # of weather signal (e.g. extrapolate weather across clutter region)
        dfth_filtered_df = pd.DataFrame(data=dfth_filtered)
        dfth_interp = dfth_filtered_df.interpolate(
            method='linear', axis=0, limit=filter_element[1],
            limit_direction='both', limit_area='inside')
        dfth_interp = dfth_interp.interpolate(
            method='linear', axis=1, limit=filter_element[0],
            limit_direction='both', limit_area='inside')
        dfth_interp = dfth_interp.values
    
        # Label and cut spectra
        dfth_zeroed = dfth_interp.copy()
        dfth_zeroed[dfth_zeroed == 0] = 0.00001
        dfth_zeroed[np.isnan(dfth_zeroed)] = 0
        dfth_labeled, label_num = nd.label(dfth_zeroed)
        dfth_binary = dfth_labeled.copy()
        dfth_binary[dfth_binary != 0] = 1
        # Find (pixel) areas and DFTh maxima of labeled regions
        region_areas = nd.sum(
            dfth_binary, dfth_labeled, range(label_num + 1))
        region_dfthmax = nd.maximum(
            dfth_interp, dfth_labeled, range(label_num + 1))
        region_dfthmax[~np.isfinite(region_dfthmax)] = -999
        # Only keep labeled region(s) with certain area and DFTh max
        if region_area is None:
            area_threshold = region_areas.max()
        else:
            area_threshold = region_area
        dfthmax_threshold = filter_thresholds[-1]
        area_dfthmax_mask = (region_areas < area_threshold) | \
                            (region_dfthmax < dfthmax_threshold)
        dfth_binary[area_dfthmax_mask[dfth_labeled]] = 0
        # Cut selection
        dfth_labeled[dfth_binary == 0] = 0
        dfth_interp[dfth_binary == 0] = np.nan
        
        # Interpolate labeled and cut regions across velocities only
        labels_valid = np.unique(dfth_labeled)
        for region_label in labels_valid[labels_valid > 0]:
            # Mask all other (labeled) regions of interest 
            region_mask = (dfth_labeled != region_label)
            dfth_interp_region = dfth_interp.copy()
            dfth_interp_region[region_mask] = np.nan
            # Interpolate current (labeled) region of interest 
            dfth_interp_region_df = pd.DataFrame(dfth_interp_region)
            dfth_interp_region_df.interpolate(
                method='linear', axis=0, limit_area='inside', inplace=True)
            # Update full data of all Doppler spectra with interpolated values
            valids = ~np.isnan(dfth_interp_region_df.values)
            dfth_interp[valids] = dfth_interp_region_df.values[valids]
            
        # Output originally measured mean values if selected 
        if not use_interpolated:
            no_nan = ~np.isnan(dfth_interp)
            dfth_interp[no_nan] = dfth_mean.values[no_nan]
    
    # Thresholding flavor of 'none_expanded' 
    # (no isolation and expand velocity range, for e.g. strong convection)        
    else:
            
        # Base data for expanding and later cutting of Doppler spectra 
        power_spectra = dfth_mean.values
        heights_spectra = height.values
        velocities_spectra = velocity.values
        
        # Extend Doppler velocity to more negative downward velocities 
        # and to higher (positive) upward velocities
        velocity_delta = np.abs(
            velocities_spectra[-1] - velocities_spectra[-2])
        v_downextend = (velocities_spectra
                        - 2 * velocities_spectra.max()
                        - velocity_delta)
        v_upextend = (velocities_spectra
                      + 2 * velocities_spectra.max()
                      + velocity_delta)
        velocities_full = np.concatenate(
            (v_downextend, velocities_spectra, v_upextend))
        
        # Extend radar output assuming fully folded signal both down and up
        power_spectra_ext = np.tile(power_spectra, (3, 1))
        # Redefine dfth_mean for consistent data output
        dfth_mean = pd.DataFrame(
            power_spectra_ext, index=velocities_full, columns=height,
            dtype='float')
        
        
        # Build mask for deleting spurious data in expanded Doppler spectra 
        # i.e. for manual dealiasing
        
        # Grid for which to calculate mask
        xx, yy = np.meshgrid(velocities_full, heights_spectra * 1e-3)
        
        # Lines to separate visually appealing spectra from spurious data
        if mask_type == 'h(v)':
            mask_up = (yy.T < mask_high[0] * xx.T + mask_high[1])
            mask_down = (yy.T > mask_low[0] * xx.T + mask_low[1])
        elif mask_type == 'v(h)':
            mask_up = (xx.T > mask_high[0] * yy.T + mask_high[1])
            mask_down = (xx.T < mask_low[0] * yy.T + mask_low[1])
        else:
            raise ValueError('Invalid mask_type in isolate_spectra().')
        full_mask = mask_up | mask_down
        
        # Filter/Mask extended Doppler spectra
        dfth_filtered_ext = power_spectra_ext.copy()
        dfth_filtered_ext[full_mask] = np.nan
        
        # Remove ground clutter at low heights above radar
        height_min = min_height
        dfth_filtered_ext[:, height<height_min] = np.nan
        
        # Re-name for consistency
        dfth_interp = dfth_filtered_ext.copy()
        velocity = velocities_full
        
    # Collect output in 2 DataFrames pre- and post-isolation step(s)
    measured_spectra = dfth_mean
    isolated_spectra = pd.DataFrame(
        dfth_interp, index=velocity, columns=height, dtype='float')

    return isolated_spectra, measured_spectra


def isolate_weather(data, filter_settings):
    """
    Isolate weather signal in birdbath Doppler spectra.
    
    Args:
        data (dict): 
            Dictionary of 2 DataFrames: 'birdbath_spectra' and 
            'birdbath_moments', and 2 metadata keys needed for
            further processing.
        
        filter_settings (dict):
            5 keys for setting filtering options of postprocessing,
            in addition to 2 metadata keys in data.
         
            
    Returns:
        weather_spectra (dict):
            Dictionary of two DataFrames: 'isolated_weather' contains
            filtered Doppler spectra, 'radar_measured' unfiltered spectra 
            (but averaged over all 15 rays), and 1 metadata key.
    """
    
    print('filtering birdbath spectra...')
    
    # Extract data and individual settings required for filtering steps
    birdbath_spectra = data['birdbath_spectra']
    minimum_height = data['analysis_minheight']
    box_roi = data['structuring_element']
    thresholding_flavor = filter_settings['thresholding']
    thresholding_presets = filter_settings['fixed_thresholds']
    minimum_dftharea = filter_settings['dfth_minarea']
    interpolation = filter_settings['interpolate_isolated']
    mask = filter_settings['mask']
    
    # Thresholding for clutter and background filter, if selected
    if thresholding_flavor == 'adaptive':
        thresholds = clustering.hdbscan_thresholds(
            birdbath_spectra,
            filter_element=box_roi,
            threshold_type='met_clut') 
    elif thresholding_flavor == 'fixed':
        thresholds = list(thresholding_presets.values())  
    else:
        thresholds = None
    
    # Isolate weather (or use measured spectra)
    if (thresholding_flavor == 'adaptive') or (thresholding_flavor == 'fixed'):
        # use thresholds for filtering if thresholds are not None
        isolated_spectra, measured_spectra = isolate_spectra(
            birdbath_spectra, filter_element=box_roi,
            filter_thresholds=thresholds, min_height=minimum_height,
            region_area=minimum_dftharea, use_interpolated=interpolation)
    
    elif thresholding_flavor == 'none_expanded':
        # no thresholding, but expand velocity range and cut out mask
        isolated_spectra, measured_spectra = isolate_spectra(
            birdbath_spectra, filter_element=box_roi,
            filter_thresholds=thresholds, min_height=minimum_height,
            mask_high=mask['mask_up'], mask_low=mask['mask_down'],
            mask_type=mask['mask_flavor'])
    
    else:
        raise ValueError('Illegal thresholding_flavor in isolate_weather().')
        
    # Combine into a dictionary for compact output
    weather_spectra = dict(
        isolated_weather=isolated_spectra,
        radar_measured=measured_spectra,
        thresholding=thresholding_flavor)
    
    return weather_spectra
