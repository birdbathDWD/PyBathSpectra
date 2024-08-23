"""
Clustering algorithms used for DWD birdbath scan postprocessing. 'hdbscan' for
adaptively finding filter thresholds and 'unidip' for adaptively finding peaks.
Not relevant for options other than 'adaptive' selected in settings file.

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

import numpy as np
from scipy.signal import savgol_filter
from hdbscan import HDBSCAN
from unidip import UniDip

from pybathspectra import filtering


def hdbscan_thresholds(df, filter_element=[3, 7], threshold_type='met_clut'):
    """
    Determine thresholds for isolating weather signal by HDBSCAN clustering.
    
    Find threshold values that are used later on for isolating the 
    meteorological signal of the weather radar Doppler spectra, using HDBSCAN
    clustering. This yields the threshold values for difference between DFTh
    and DFTv in dB and for its variability (standard deviation) to filter out
    background signal and ground clutter, as well as a threshold value for DFTh
    in dB that can be used to select different connected regions of interest
    besides main ROI (the latter functionality is not used, so far). 
    
    Args:
        df (DataFrame): 
            Dataframe that contains all DFTh,v [dB] spectra read in 
            by read_spectra. 
        
        filter_element (list):
            Structuring element of [height, velocity] bins used for filtering.
            A good choice is [3, 7], in general.
            
        threshold_type (str):
            Method to determine final threshold values from clusters,
            as max or min value of cluster representing meteorological signal
            ('met_only') or also accounting for min or max of other clusters 
            representing different clutter regimes ('met_clut').
            
            
    Returns:
        filter_thresholds (list):
            The 3 threshold values that should be used for filtering later on. 
            [threshold value for difference between DFTh and DFTv, 
            the standard deviation of the difference, 
            DFTh threshold for selecting connected regions of interest 
            in further analysis]. 
    """
    
    print('...finding threshold values...')
    
    # Average h-pol Doppler spectra over 15 rays + calculate smoothend 
    # power difference h-v pol and the variability
    averages = filtering.smooth_average(df, filter_element=[3, 7])
    dfth_mean, dfthv_difference, dfthv_variability = averages
    
    # Data for clustering     
    data_dfth_flat = dfth_mean.copy().values.ravel()
    data_diff_dft_flat = dfthv_difference.ravel()
    data_diff_dft_std_flat = dfthv_variability.ravel()
    cluster_data = np.stack(
        (data_diff_dft_flat, data_diff_dft_std_flat, data_dfth_flat), axis=1)
    
    # HDBSCAN clustering
    # Scale data (improves clustering results, sometimes)
    hdbscan_data = cluster_data / cluster_data.max(axis=0)
    # Clustering
    hdbscan_clusterer = HDBSCAN(
        min_cluster_size=500,
        min_samples=50,
        cluster_selection_epsilon=0.0,
        prediction_data=True).fit(hdbscan_data)
    hdbscan_labels = hdbscan_clusterer.labels_
    
    # Find suitable filter thresholds quantitatively
    # Have labels start at 0 instead of -1
    labels = hdbscan_labels + 1
    labels_max = labels.max()
    # Calculate some characteristic properties of individual clusters 
    mean_hdbscan_clusters = np.full((labels_max+1, 3), np.nan)
    median_hdbscan_clusters = np.full((labels_max+1, 3), np.nan)
    max_hdbscan_clusters = np.full((labels_max+1, 3), np.nan)
    min_hdbscan_clusters = np.full((labels_max+1, 3), np.nan)
    # scaled version of the above
    mean_hdbscan_clusters_scaled = np.full((labels_max+1, 3), np.nan)
    median_hdbscan_clusters_scaled = np.full((labels_max+1, 3), np.nan)
    max_hdbscan_clusters_scaled = np.full((labels_max+1, 3), np.nan)
    min_hdbscan_clusters_scaled = np.full((labels_max+1, 3), np.nan)
    for label in range(labels_max+1):
        mean_hdbscan_clusters[label,:] = np.mean(
            cluster_data[labels==label,:], axis=0)
        median_hdbscan_clusters[label,:] = np.median(
            cluster_data[labels==label,:], axis=0)
        max_hdbscan_clusters[label,:] = np.max(
            cluster_data[labels==label,:], axis=0)
        min_hdbscan_clusters[label,:] = np.min(
            cluster_data[labels==label,:], axis=0)
        # scaled version of the above
        mean_hdbscan_clusters_scaled[label,:] = np.mean(
            hdbscan_data[labels==label,:], axis=0)
        median_hdbscan_clusters_scaled[label,:] = np.median(
            hdbscan_data[labels==label,:], axis=0)
        max_hdbscan_clusters_scaled[label,:] = np.max(
            hdbscan_data[labels==label,:], axis=0)
        min_hdbscan_clusters_scaled[label,:] = np.min(
            hdbscan_data[labels==label,:], axis=0)
        
    # Find meteorological signal = weather signal
    met_criterion = (mean_hdbscan_clusters_scaled[:,0]**2
                     + mean_hdbscan_clusters_scaled[:,1]**2)
    met_label = np.argwhere(met_criterion == met_criterion.min()).squeeze()
    
    # Estimate reasonable thresholds for isolating the meterorological signal
    min_met = min_hdbscan_clusters[met_label,:]
    max_met = max_hdbscan_clusters[met_label,:]
    min_rest = np.delete(min_hdbscan_clusters, [0,met_label], axis=0)
    max_rest = np.delete(max_hdbscan_clusters, [0,met_label], axis=0)
    mean_rest = np.delete(mean_hdbscan_clusters, [0,met_label], axis=0)
    # Two different flavors of how to determine appropriate threshold values
    if threshold_type == 'met_only':
        thresh_diff = max_met[0]
        thresh_std = max_met[1]
        thresh_dfth = min_met[2]
    elif threshold_type == 'met_clut':
        cluster_index = np.argwhere(mean_rest[:,2] == mean_rest[:,2].max())
        maxDFTh_cluster_ind = cluster_index.squeeze()
        thresh_diff = 0.5 * (max_met[0] + min_rest[maxDFTh_cluster_ind,0])
        thresh_std = 0.5 * (max_met[1] + min_rest[maxDFTh_cluster_ind,1])
        weakest_condition = (mean_rest[:,2] == mean_rest[:,2].min())
        thresh_double = min_met[2] + max_rest[weakest_condition,2].squeeze()
        thresh_dfth = 0.5 * thresh_double
        if thresh_dfth < max_rest[weakest_condition,2].squeeze():
            thresh_dfth = max_rest[weakest_condition,2].squeeze()
    else:
         raise ValueError('Invalid threshold_type entered: find_thresholds().')
    
    # Combine results to be used for filtering later
    filter_thresholds = [
        thresh_diff.item(),
        thresh_std.item(),
        thresh_dfth.item(),
    ]
    
    return filter_thresholds


def unidip_peaks(
        dfth_interp, dfth_mean, height, velocity,
        height_anaspec, windowsize, powerscale):
    """
    Find peaks in Doppler spectrum by UniDip clustering.
    
    Only used if 'adaptive' peak_finder is set in postprocessing settings file.
    UniDip algorithm finds 'statistically significant' peaks automatically,
    but does not really work for multiple peaks that are not close to
    typical distribution shapes and not well separated.
    
    Args:
        dfth_interp (array): 
            Array of DFTh values [dB]. Isolated meteorological signal;
            invalid bins = np.nan.
            
        dfth_mean (DataFrame):
            DataFrame of DFTh values [dB] before isolating weather signal.
            
        height (DataFrame column):
            Vector of all heights [m] above radar corresponding to dfth values.
            
        velocity (DataFrame index):
            Velocities [m/s] corresponding to dfth values.
            
        height_anaspec (scalar):
            One specific height bin [m]; current height level of analysis.
            
        windowsize (scalar):
            Number of velocity bins used for filtering dfth_mean.
            Derived from structuring_element in settings file.
            
        powerscale (str):
            Find peak intervals for Doppler spectra on linear or dB scale;
            'linear' is linear scale, 'dBlin' is dB scale. 
            Use 'dBlin' generally, as 'linear' option is not always valid.
            
            
    Returns:
        unidip_output (tuple or -999):
            UniDip clustering outputs of peak intervals required for then
            identifying all separate modes in Doppler spectrum.      
    """
        
    # Reverse mask of filtered and interpolated spectra dfth_interp
    dfth_rev = dfth_interp.copy()
    dfth_rev[~np.isnan(dfth_rev)] = -999
    dfth_rev[np.isnan(dfth_rev)] = dfth_mean.values[np.isnan(dfth_rev)]
    dfth_rev[dfth_rev==-999] = np.nan
    # Background noise estimate per height bin
    dfth_noise = np.nanmedian(dfth_rev, axis=0)
    
    # Pick selected height bin
    dfth_anaspec = dfth_interp[:, height==height_anaspec].squeeze()
    
    # Replace filtered/removed measurements with noise estimate at height bin
    dfth_anaspec[np.isnan(dfth_anaspec)] = dfth_noise[height==height_anaspec]
    # Measurements without filter, i.e., averaged output from signal processor
    dfth_anaspec_II = dfth_mean.values[:, height==height_anaspec].squeeze()    
    
    # Transform dB values to linear scale
    dfth_anaspec_lin = 10 ** (0.1 * dfth_anaspec)
    dfth_noise_lin = 10 ** (0.1 * dfth_noise)
    dfth_anaspec_II_lin = 10 ** (0.1 * dfth_anaspec_II)
    
    # Remove irrelevant part of spectrum
    # check if all noise (i.e. if no meteorological signal)
    noise_check = np.argwhere(
        dfth_anaspec > dfth_noise[height==height_anaspec])
    if noise_check.size < 1:
        dfth_selspec = np.nan
        dfth_selspec_lin = np.nan
    # general case is analyzed further
    else:
        idx_start = noise_check[0][0] - 1
        idx_end = noise_check[-1][-1] + 1
        dfth_selspec = dfth_anaspec[idx_start:idx_end+1]
        dfth_selspec_lin = dfth_anaspec_lin[idx_start:idx_end+1]
    
    # Try to smooth valid signals (without end points=background noise) 
    # using Savitzky-Golay filter (polyorder > 1: better reproduces real peaks)
    # a) invalid case 1 from above (all noise, no valid data)
    if np.isnan(dfth_selspec).any() or np.isnan(dfth_selspec_lin).any():
        unidip_output = -999
    # b) invalid case 2 (not enough valid data)
    elif (len(dfth_selspec) < windowsize+2 or 
          len(dfth_selspec_lin) < windowsize+2):
        unidip_output = -999
    # c) valid case for further analysis
    else:
        if powerscale == 'linear':
            dfth_selspec_lin_sg = savgol_filter(
                dfth_selspec_lin[1:-1], window_length=windowsize, polyorder=2)
            # Replace values < previously determined minimum, due to SG filter
            minimum_condition = (dfth_selspec_lin_sg < dfth_selspec_lin.min())
            dfth_selspec_lin_sg[minimum_condition] = dfth_selspec_lin.min()
        elif powerscale == 'dBlin':
            dfth_selspec_sg = savgol_filter(
                dfth_selspec[1:-1], window_length=windowsize, polyorder=2)
            # Replace values < previously determined minimum, due to SG filter
            minimum_condition = (dfth_selspec_sg < dfth_selspec.min())
            dfth_selspec_sg[minimum_condition] = dfth_selspec.min()
            # Linear-scale dfth values for spectrum savgol-filtered in dB-space
            dfth_selspec_lin_sg = 10 ** (0.1 * dfth_selspec_sg)
        else:
            raise ValueError('Invalid powerscale for find_peakintervals().')
    
        # Construct explicit velocity distribution from dfth values for UniDip
        if powerscale == 'linear':
            pick_dfth_lin_full = dfth_selspec_lin_sg
            # Rescale values 
            lin_range = [0, 1]
            pick_top = pick_dfth_lin_full - pick_dfth_lin_full.min()
            pick_bottom = pick_dfth_lin_full.max() - pick_dfth_lin_full.min()
            scaling_factor_lin = pick_top / pick_bottom
            range_diff = lin_range[1] - lin_range[0]
            pick_dfth_lin = lin_range[0] + scaling_factor_lin * range_diff
            # Transform dfth values to number of velocity values per bin
            overall_scaling = 200
            number_dfth = pick_dfth_lin * overall_scaling
            dfth_as_int_lin = np.rint(number_dfth).astype(int)
            # Velocity distribution
            velocity_selspec = velocity.values[idx_start:idx_end+1]
            velocity_selspec_sg = velocity.values[idx_start+1:idx_end]
            velocity_res = velocity_selspec_sg[1] - velocity_selspec_sg[0]
            vel_distrib = np.array([])
            for vel in velocity_selspec_sg:
                vel_num = dfth_as_int_lin[velocity_selspec_sg==vel]
                new_vels = np.linspace(
                    vel - velocity_res/2,
                    vel + velocity_res/2,
                    num=vel_num[0],
                    endpoint=False)
                new_vels.sort()
                vel_distrib = np.append(vel_distrib, new_vels)
            # Find peaks with UniDip
            peak_intervals_veldistrib = UniDip(
                vel_distrib,
                is_hist=False,
                alpha=0.05,
                ntrials=100,
                mrg_dst=1).run()
        
        elif powerscale == 'dBlin':
            pick_dfth = dfth_selspec_sg
            # Rescale values
            lin_range = [0, 1]
            pick_top = pick_dfth - pick_dfth.min()
            pick_bottom = pick_dfth.max() - pick_dfth.min()
            scaling_factor_dB = pick_top / pick_bottom
            range_diff = lin_range[1] - lin_range[0]
            pick_dfth = lin_range[0] + scaling_factor_dB * range_diff
            # Transform dfth values to number of velocity values per bin
            overall_scaling = 200
            dfth_as_int = np.rint(pick_dfth * overall_scaling).astype(int)
            # Velocity distribution
            velocity_selspec = velocity.values[idx_start:idx_end+1]
            velocity_selspec_sg = velocity.values[idx_start+1:idx_end]
            velocity_res = velocity_selspec_sg[1] - velocity_selspec_sg[0]
            vel_distrib = np.array([])
            for vel in velocity_selspec_sg:
                vel_num = dfth_as_int[velocity_selspec_sg==vel]
                new_vels = np.linspace(
                    vel - velocity_res/2,
                    vel + velocity_res/2,
                    num=vel_num[0],
                    endpoint=False)
                new_vels.sort()
                vel_distrib = np.append(vel_distrib, new_vels)
            # Find peaks with UniDip
            peak_intervals_veldistrib = UniDip(
                vel_distrib,
                is_hist=False,
                alpha=0.05,
                ntrials=100,
                mrg_dst=1).run()
            
        else:
            raise ValueError('Invalid powerscale in find_peakintervals().')
        
      
        # Cleanup: exclude peak intervals w. 'peak' at edge of UniDip interval
        unidip_intervals = np.array(peak_intervals_veldistrib)
        unidip_finals = np.array([], dtype='int32')
        for peak_int in unidip_intervals:
            vel_min = vel_distrib[peak_int[0]]
            vel_max = vel_distrib[peak_int[1] - 1]
            veldiff_min = np.abs(velocity_selspec_sg - vel_min)
            veldiff_max = np.abs(velocity_selspec_sg - vel_max)
            vel_min_bin = np.where(veldiff_min == veldiff_min.min())
            vel_max_bin = np.where(veldiff_max == veldiff_max.min())
            min_valid = vel_min_bin[0][0]
            max_valid = vel_max_bin[0][0] + 1
            dfth_max = dfth_selspec_lin_sg[min_valid:max_valid].max()
            if (dfth_max != dfth_selspec_lin_sg[vel_min_bin[0][0]]) & \
                (dfth_max != dfth_selspec_lin_sg[vel_max_bin[0][0]]):
                # Add valid interval(s) to final array of interval(s)
                unidip_finals = np.append(unidip_finals, peak_int)
        unidip_finals = unidip_finals.reshape((-1,2))
        
        # Catch special case: no mode -> choose entire interval as mode
        if unidip_finals.shape[0] < 1:
            unidip_finals = np.array([[0, len(vel_distrib)]])
        
        # Collect analysis output
        unidip_output = (
            unidip_finals, vel_distrib,
            velocity_selspec, velocity_selspec_sg,
            dfth_selspec_lin, dfth_selspec_lin_sg,
            dfth_noise_lin, dfth_anaspec_II_lin,
        )
        
    return unidip_output
