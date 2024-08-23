"""
Functions for plotting DWD birdbath scan data and postprocessing results.

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
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from pybathspectra import filtering


def xydata_pcolormesh(x_vector, y_vector):
    """
    Build correct x,y data needed for pcolormesh plots of birdbath spectra.
    
    Args:
        x_vector, y_vector (arrays):
            Arrays of x and y data at bin centers for Doppler spectra.
            x = velocities, y = heights.
        
        
    Returns:
        xdata, ydata (arrays):
            Arrays of x and y data at bin edges, for pcolormesh plot.
    """
    
    xdata_offset = (x_vector[1] - x_vector[0]) / 2
    ydata_offset = (y_vector[1] - y_vector[0]) / 2
    xdata = x_vector - xdata_offset
    xdata = np.append(xdata, x_vector[-1] + xdata_offset)
    ydata = y_vector - ydata_offset
    ydata = np.append(ydata, y_vector[-1] + ydata_offset)
    
    return xdata, ydata


def velocity_minmax(isolated_spectra):
    """
    Derive appropriate velocity range for plotting isolated spectra.
    
    Args:
        isolated_spectra (DataFrame):
            Isolated weather spectra, i.e. part of output from 
            filtering.isolate_weather().
        
        
    Returns:
        velocity_min, velocity_max (tuple):
            Values of min and max velocity for compact
            plot of isolated weather spectra.
    """
    
    # All velocities of weather spectra
    velocities = isolated_spectra.index.values
    isolated_weather = isolated_spectra.values
    
    # Determine minimum and maximum valid velocity
    valid = np.argwhere(~np.isnan(isolated_weather))
    min_valid_min = valid[:,0].min()
    max_valid_max = valid[:,0].max()
    velocity_min = velocities[min_valid_min]
    velocity_max = velocities[max_valid_max]
    
    return velocity_min, velocity_max


def height_max(isolated_spectra):
    """
    Derive potential maximum height for plotting isolated spectra.
    
    Args:
        isolated_spectra (DataFrame):
            Isolated weather spectra, i.e. part of output from 
            filtering.isolate_weather().
        
        
    Returns:
        height_max (scalar):
            Max height of spectra with valid Doppler power,
            for compact plot of isolated weather spectra.
    """
    
    # All heights of weather spectra
    heights = isolated_spectra.columns.values
    isolated_weather = isolated_spectra.values
    
    # Determine maximum valid height
    valid = np.argwhere(~np.isnan(isolated_weather))
    max_valid_max = valid[:,1].max()
    height_max = heights[max_valid_max]
    
    return height_max


def extract_multimodal(multimodal_data):
    """
    Set up plots of results of multimodal analysis.
    
    Args:
        multimodal_data (dict):
            Dictionary of multimodal analysis results and metadata
            (i.e. no spectral reflectivity data).
        
        
    Returns:
        plot_setup (tuple of various data):
            many parameters and variables for plotting results of
            multimodal analysis.
    """
    
    # Separate results from multimodal analysis
    modal = multimodal_data['modal_properties']
    multimodal = multimodal_data['multimodal_properties']
    modal_uncertainty = multimodal_data['modal_uncertainties']
    multimodal_uncertainty = multimodal_data['multimodal_uncertainties']
    heights = multimodal_data['range_heights']
    # Mean Doppler veloctiy
    v_mean = modal[1,:,:]
    # Colors for individual modes (number of colors should = v_mean.shape[1])
    modecolors = np.array([
            'tab:blue', 'tab:olive', 'tab:green', 'tab:cyan', 'tab:pink',
            'tab:orange', 'tab:gray', 'lime', 'tab:purple', 'tab:brown',
            'tab:blue', 'tab:olive', 'tab:green', 'tab:cyan', 'tab:pink',
            'tab:orange', 'tab:gray', 'lime', 'tab:purple', 'tab:brown'])
    if (len(modecolors) != v_mean.shape[1]):
        raise ValueError(
            'Number of different mode colors and number of maximum modes does'
            ' not agree in plotting.extract_multimodal(). Change maximum_modes'
            ' setting in config file to ' + str(len(modecolors)) + 
            ' for safely plotting modal and multimodal properties.'
        )
    # Colors for peaks flag
    peaks_flag = modal[-1,:,:]
    flag_color = 'red'
    # Mode widths
    v_std = modal[2,:,:]
    # Reflectivities
    dBZh = modal[0,:,:]
    # (Linear) reflectivity percent of full signal
    Zh_percentage = modal[-2,:,0]
    # (Normalized) skewness moment
    v_skew = modal[3,:,:]
    # (Normalized) kurtosis
    v_kurt = modal[4,:,:]
    # Median skewness
    v_skew_med = modal[5,:,:]
    # Mode skewness
    v_skew_mode = modal[9,:,:]
    # Multimodal ratio [dB difference]
    v_rat_mode = multimodal[0,:,:]
    # (Normalized) Bimodal separation
    v_sep_mean_norm = multimodal[1,:,:]
    # Bimodal amplitude ratio (linear units and alt(ernative) in dB)
    v_amp_mode = multimodal[2,:,:]
    v_amp_mode_alt = multimodal[3,:,:]
    # Relative uncertainty [%] of skewness moment, median, mode
    unc_skew = np.abs(modal_uncertainty[3,:,:] / v_skew) * 100
    unc_skew_med = np.abs(modal_uncertainty[5,:,:] / v_skew_med) * 100
    unc_skew_mode = np.abs(modal_uncertainty[9,:,:] / v_skew_mode) * 100
    # Relative uncertainty [%] of mean
    unc_mean = np.abs(modal_uncertainty[1,:,:] / v_mean) * 100
    # Relative uncertainty [%] of std
    unc_std = np.abs(modal_uncertainty[2,:,:] / v_std) * 100
    # Relative uncertainty [%] of kurtosis
    unc_kurt = np.abs(modal_uncertainty[4,:,:] / v_kurt) * 100
    # Uncertainty [dB] of dBZh
    unc_dBZh = np.abs(modal_uncertainty[0,:,:])
    # Uncertainty [% absolute] of (linear) reflectivity percent of full signal
    unc_Zhperc = np.abs(modal_uncertainty[-1,:,:])
    # Uncertainties of some relevant multimodal properties
    unc_rat = np.abs(multimodal_uncertainty[0,:,:])  # [dB]
    unc_sep = np.abs(multimodal_uncertainty[1,:,:] / v_sep_mean_norm) * 100
    unc_amp = np.abs(multimodal_uncertainty[2,:,:] / v_amp_mode) * 100
    unc_amp_alt = np.abs(multimodal_uncertainty[3,:,:])  # [dB]
    # Flag dubious modes based on peaks_flag (not used here)
    flag_modes = True
    
    # Combine data
    plot_setup = (
        modecolors, flag_color, flag_modes, peaks_flag, heights, v_mean, v_std,
        dBZh, Zh_percentage, v_skew, v_kurt, v_skew_med, v_skew_mode,
        v_rat_mode, v_sep_mean_norm, v_amp_mode, v_amp_mode_alt, unc_skew,
        unc_skew_med, unc_skew_mode, unc_mean, unc_std, unc_kurt, unc_dBZh,
        unc_Zhperc, unc_rat, unc_sep, unc_amp, unc_amp_alt,
    )
    
    return plot_setup

    
def plot_birdbath(data, time, plot_path='./plots/'):
    """
    Plot overview of radar output data from DWD birdbath scan.
    
    Plot Doppler spectra, spectrally resolved polarimetric parameters for
    clutter and background filter and (some) radar variables from moment file.
    
    Args:
        data (dict):
            Dictionary of 2 DataFrames: 'birdbath_spectra' and 
            'birdbath_moments', and 2 metadata keys included directly.
        
        time (str):
            Time string of birdbath scan (could be extracted from data).
        
        box_roi (list):
            Number of [height, velocity] bins employed in filtering. 
            Pick same values that were used for loading birdbath data. 
            A good choice is [3, 7], in general.
            
        
    Returns:
        Figures.
    """
    
    print('plotting birdbath scan data...')
    
    # Create plotting directory if it does not exist
    os.makedirs(plot_path, exist_ok=True)
    
    # 1) Extract and plot all radar moments first
    moments = data['birdbath_moments'].droplevel(0)
    #birdbath_time = data['birdbath_moments'].droplevel(1).index[0]
    birdbath_time = dt.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    min_height = data['analysis_minheight']
    # Heights above radar [km]
    heights = moments.columns.values / 1000
    
    # Create one figure for plotting all loaded moments
    fig = plt.figure(figsize=(8, 6), dpi=None, tight_layout=True)
    figname = (plot_path + 'birdbath_moments_' 
               + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
    ax = fig.add_subplot(1, 1, 1)
    
    # Loop over moments for plotting, and scale moments
    for moment_name in moments.index:
        moment_data = moments.loc[moment_name].values
        moment_label = moment_name
        if moment_name in ['Zh', 'UZh']:
            moment_data = moment_data / 10
            moment_label = moment_name + ' / 10 [dBZ]'
        elif moment_name in ['Vh', 'UVFh', 'UnVh']:
            moment_data = moment_data / 4
            moment_label = moment_name + ' / 4 [m/s]'
        elif moment_name in ['SNRh']:
            moment_data = moment_data / 20
            moment_label = moment_name + ' / 20 [dB]'
        elif moment_name in ['RHOHV', 'URHOHV']:
            moment_data = (moment_data - 1) * 10
            moment_label = '(' + moment_name + ' - 1) * 10 [$-$]'
        ax.plot(moment_data, heights, '-', lw=2, label=moment_label)
    
    if min_height > 0:
        ax.fill_between([-10, 10], min_height / 1000, alpha=0.5,
                        color='tab:gray', label='< minimum height bin')
    ax.set_xlabel('Scaled value', fontsize=13)
    ax.set_ylabel('Height (above radar) [km]', fontsize=13)
    ax.grid(linestyle=':', linewidth=0.5)
    ax.set_xlim([-6, 6])
    ax.set_ylim([0, 14])
    ax.legend(loc='upper right', numpoints=1, ncol=1, borderpad=0.2,
              borderaxespad=0.3, labelspacing=0.1, handlelength=2.0,
              framealpha=1, fontsize=13)
    fig.savefig(figname)
    
    # 2) Extract and plot dft spectral data
    spectra = data['birdbath_spectra']
    box_roi = data['structuring_element']
    dfth_vars = filtering.smooth_average(spectra, filter_element=box_roi)
    dfth_mean, dfthv_difference, dfthv_variability = dfth_vars
    # Heights above radar [km] and velocities [m/s]
    heights = dfth_mean.columns.values / 1000
    velocities = dfth_mean.index.values
    # Compile x and y data for plots with pcolormesh
    xdata, ydata = xydata_pcolormesh(velocities, heights)
    
    # 2a) Plot Doppler (power) spectra in h-polarization channel
    fig = plt.figure(figsize=(8, 6), dpi=None, tight_layout=True)
    figname = (plot_path + 'birdbath_Doppler_spectra_'
               + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
    ax = fig.add_subplot(1, 1, 1)
    spec = ax.pcolormesh(xdata, ydata, dfth_mean.values.T,
                         vmin=40, vmax=120, cmap='jet', zorder=1)
    cb = plt.colorbar(spec, ax=ax, orientation='vertical', extend='both')
    cb.set_label('Uncalibrated power [dB]', fontsize=13)
    cb.ax.tick_params(labelsize=12)
    #ax.set_ylim([0, 13.2])
    ax.grid(linestyle=':', linewidth=0.5)
    ax.set_xlabel(r'Doppler velocity [m s$^{-1}$]', fontsize=13)
    ax.set_ylabel(r'Height (above radar) [km]', fontsize=13)
    fig.savefig(figname)
    
    # 2b) Plot spectral polarimetric parameters
    fig = plt.figure(figsize=(8, 6), dpi=None, tight_layout=True)
    figname = (plot_path + 'birdbath_polarimetric_spectra_'
               + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
    # h-v polarization channel differences
    ax = fig.add_subplot(1, 2, 1)
    spec = ax.pcolormesh(xdata, ydata, dfthv_difference.T,
                         vmin=0, vmax=4, cmap='inferno', zorder=1)
    cb = plt.colorbar(spec, ax=ax, orientation='horizontal', extend='max')
    cb.set_label(r'Power difference $H,V$ polarization [dB]', fontsize=13)
    cb.ax.tick_params(labelsize=12)
    ax.grid(linestyle=':', linewidth=0.5)
    ax.set_xlabel(r'Doppler velocity [m s$^{-1}$]', fontsize=13)
    ax.set_ylabel(r'Height (above radar) [km]', fontsize=13)
    # h-v variabilities
    ax2 = fig.add_subplot(1, 2, 2)
    spec2 = ax2.pcolormesh(xdata, ydata, dfthv_variability.T,
                           vmin=0, vmax=2, cmap='viridis', zorder=1)
    cb2 = plt.colorbar(spec2, ax=ax2, orientation='horizontal', extend='max')
    cb2.set_label(r'Standard deviation [dB]', fontsize=13)
    cb2.ax.tick_params(labelsize=12)
    ax2.grid(linestyle=':', linewidth=0.5)
    ax2.set_xlabel(r'Doppler velocity [m s$^{-1}$]', fontsize=13)
    fig.savefig(figname)
    
    
def plot_birdbath_rgb(data, time, scaling=[1, 1.8, 1.8], plot_path='./plots/'):
    """
    Summarize DWD birdbath spectra in RGB and RB pseudo-color plots.
    
    Args:
        data (dict):
            Dictionary of 2 DataFrames: 'birdbath_spectra' and 
            'birdbath_moments', and 2 metadata keys included directly.
            
        time (str):
            Time string of birdbath scan (could be extracted from data).
        
        scaling (list):
            Scaling factors for R, G, and B color channels.
            Generally, boost G and B channels a bit.
            R = h-pol power, G = h-v power difference, B = h-v variability.
            
        plot_path (Path):
            Relative path of directory where output plots should be saved.
            
        
    Returns:
        Figures of 1 RGB and 1 RB pseudo color plot each.
    """
    
    print('plotting birdbath spectra in RGB pseudo color...')
    
    # Create plotting directory if it does not exist
    os.makedirs(plot_path, exist_ok=True)
    
    # Get time of birdbath scan
    #birdbath_time = data['birdbath_moments'].droplevel(1).index[0]
    birdbath_time = dt.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    
    # Extract and plot dft spectral data
    spectra = data['birdbath_spectra']
    box_roi = data['structuring_element']
    dfth_vars = filtering.smooth_average(spectra, filter_element=box_roi)
    dfth_mean, dfthv_difference, dfthv_variability = dfth_vars
    # Heights above radar [km] and velocities [m/s]
    heights = dfth_mean.columns.values / 1000
    velocities = dfth_mean.index.values
    # Compile x and y data for plots with pcolormesh
    xdata, ydata = xydata_pcolormesh(velocities, heights)
    
    # Normalize and scale data for plotting
    dfth_flat = dfth_mean.values.ravel()
    dfthv_diff_flat = dfthv_difference.ravel()
    dfthv_var_flat = dfthv_variability.ravel()
    dfth_mean_scaled = dfth_mean.values / dfth_flat.max()
    dfthv_diff_scaled = dfthv_difference / dfthv_diff_flat.max()
    dfthv_var_scaled = dfthv_variability / dfthv_var_flat.max()
    # Transform normalized values to RGB values and rescale/boost Blue,Green
    dfth_mean_R = scaling[0] * dfth_mean_scaled
    dfthv_diff_G = scaling[1] * dfthv_diff_scaled
    dfthv_var_B = scaling[2] * dfthv_var_scaled
    # Summarize all RGB channels for RGB and RB plots
    zdata_output_RGB_T = np.stack(
        (dfth_mean_R.T, dfthv_diff_G.T, dfthv_var_B.T), axis=2)
    zdata_output_BG_mod = dfthv_diff_G + dfthv_var_B
    zdata_output_R_BG = dfth_mean_R - zdata_output_BG_mod
    # Limit values to max value of 1 for RGB plot
    zdata_output_RGB_T[zdata_output_RGB_T > 1.0] = 1.0
    
    # 1) Plot radar output in RGB-pseudo color (incl. polarimetric parameters)
    fig = plt.figure(figsize=(8, 6), dpi=None, tight_layout=True)
    figname = (plot_path + 'birdbath_RGB_spectra_' 
               + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
    ax = fig.add_subplot(1, 1, 1)
    spec = ax.pcolormesh(xdata, ydata, zdata_output_RGB_T[:,:,0],
                         facecolors=zdata_output_RGB_T.reshape(-1,3))
    spec.set_array(None)
    ax.grid(linestyle=':', linewidth=0.5)
    ax.set_xlabel(r'Doppler velocity [m s$^{-1}$]', fontsize=13)
    ax.set_ylabel(r'Height (above radar) [km]', fontsize=13)
    fig.savefig(figname)
    
    # 2) Plot radar output in RB-pseudo color (incl. polarimetric parameters)
    fig = plt.figure(figsize=(8, 6), dpi=None, tight_layout=True)
    figname = (plot_path + 'birdbath_RminusBG_spectra_'
               + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
    ax = fig.add_subplot(1, 1, 1)
    spec = ax.pcolormesh(xdata, ydata, zdata_output_R_BG.T, cmap='bwr',
                         vmin=-1, vmax=1, zorder=1)
    cb = plt.colorbar(spec, ax=ax, orientation='vertical', extend='both')  
    cb.set_label('RB pseudo color [$-$]', fontsize=13)
    cb.ax.tick_params(labelsize=12)
    ax.grid(linestyle=':', linewidth=0.5)
    ax.set_xlabel(r'Doppler velocity [m s$^{-1}$]', fontsize=13)
    ax.set_ylabel(r'Height (above radar) [km]', fontsize=13)
    fig.savefig(figname)
        

def plot_weather(data, time, plot_path='./plots/'):
    """
    Plot overview of (isolated) weather spectra from DWD birdbath scan.
    
    Plot Doppler spectra before and after isolation step 
    (if selected in postprocessing settings).
    
    Args:
        data (dict):
            Dictionary of 2 DataFrames: isolated weather spectra and radar
            measured spectra, and metadata included.
            
        time (str):
            Time string of birdbath scan.
            
        plot_path (Path):
            Relative path of directory where output plots should be saved.
            
        
    Returns:
        Figures.
    """
    
    print('plotting (isolated) weather spectra...')
    
    # Create plotting directory if it does not exist
    os.makedirs(plot_path, exist_ok=True)
    
    # Get time of birdbath scan
    birdbath_time = dt.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    
    # 1) Extract and plot Doppler spectra before isolation
    spectra = data['radar_measured']
    # Heights above radar [km] and velocities [m/s]
    heights = spectra.columns.values / 1000
    velocities = spectra.index.values
    # Compile x and y data for plots with pcolormesh
    xdata, ydata = xydata_pcolormesh(velocities, heights)
    
    # Plot
    fig = plt.figure(figsize=(8, 6), dpi=None, tight_layout=True)
    figname = (plot_path + 'weather_measured_spectra_'
               + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
    ax = fig.add_subplot(1, 1, 1)
    spec = ax.pcolormesh(xdata, ydata, spectra.values.T,
                         vmin=40, vmax=120, cmap='jet', zorder=1)
    cb = plt.colorbar(spec, ax=ax, orientation='vertical', extend='both')
    cb.set_label('Uncalibrated power [dB]', fontsize=13)
    cb.ax.tick_params(labelsize=12)
    ax.grid(linestyle=':', linewidth=0.5)
    ax.set_xlabel(r'Doppler velocity [m s$^{-1}$]', fontsize=13)
    ax.set_ylabel(r'Height (above radar) [km]', fontsize=13)
    fig.savefig(figname)
    
    # 2) Extract and plot isolated weather spectra
    weather_spectra = data['isolated_weather']
    # Heights above radar [km] and velocities [m/s]
    heights = weather_spectra.columns.values / 1000
    velocities = weather_spectra.index.values
    # Compile x and y data for plots with pcolormesh
    xdata, ydata = xydata_pcolormesh(velocities, heights)
    
    # Plot
    fig = plt.figure(figsize=(8, 6), dpi=None, tight_layout=True)
    figname = (plot_path + 'weather_isolated_spectra_'
               + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
    ax = fig.add_subplot(1, 1, 1)
    spec = ax.pcolormesh(xdata, ydata, weather_spectra.values.T,
                         vmin=40, vmax=120, cmap='jet', zorder=1)
    cb = plt.colorbar(spec, ax=ax, orientation='vertical', extend='both')
    cb.set_label('Uncalibrated power [dB]', fontsize=13)
    cb.ax.tick_params(labelsize=12)
    ax.grid(linestyle=':', linewidth=0.5)
    ax.set_xlabel(r'Doppler velocity [m s$^{-1}$]', fontsize=13)
    ax.set_ylabel(r'Height (above radar) [km]', fontsize=13)
    fig.savefig(figname)
    

def plot_multimodal(multimodal_analysis, time, plot_path='./plots/'):
    """
    Plot (many) results from multimodal analysis of DWD birdbath scan.
    
    After postprocessing, plot modal properties, multimodal properties,
    and their uncertainties.
    
    Args:
        multimodal_analysis (dict):
            Dictionary of multimodal analysis results and metadata included
            (i.e. no spectral reflectivity data).
            
        time (str):
            Time string of birdbath scan.
            
        plot_path (Path):
            Relative path of directory where output plots should be saved.
            
        
    Returns:
        Figures.
    """
    
    print('plotting detailed results of multimodal analysis...')
    
    # Create plotting directory if it does not exist
    os.makedirs(plot_path, exist_ok=True)
    
    # Get time of birdbath scan
    birdbath_time = dt.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    
    # Extract data and set up plotting options
    data_setup = extract_multimodal(multimodal_analysis)
    (modecolors, flag_color, flag_modes, peaks_flag, heights, v_mean, v_std,
     dBZh, Zh_percentage, v_skew, v_kurt, v_skew_med, v_skew_mode, v_rat_mode,
     v_sep_mean_norm, v_amp_mode, v_amp_mode_alt, unc_skew, unc_skew_med,
     unc_skew_mode, unc_mean, unc_std, unc_kurt, unc_dBZh, unc_Zhperc, unc_rat,
     unc_sep, unc_amp, unc_amp_alt) = data_setup
    # Uncertainties included in data (and then to be plotted)?
    uncertainty_plots = multimodal_analysis['estimate_uncertainty']
    
    # 1) Modal properties
    fig = plt.figure(figsize=(20, 7), dpi=None, tight_layout=True)
    figname = (plot_path + 'modes_results_'
               + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax4 = ax2.twiny()
    # plot multiple modes at each height level in different colors
    for height in heights:
        condition = (heights == height)
        vels = v_mean[condition,:]
        valid = ~np.isnan(vels)
        vels_valid = vels[valid]
        if vels_valid.size == 0:
            continue
        color_vector = modecolors.copy()
        # Option to flag dubious modes (if used in multimodal analysis)
        if flag_modes:
            flag_condition = (peaks_flag[heights==height,:].squeeze() == 1)
            color_vector[flag_condition] = flag_color
        # Mean velocities
        color_vector = color_vector[:len(vels_valid)]
        height_km = np.repeat(height, len(vels_valid)) / 1000
        ax1.scatter(vels_valid, height_km, marker='o', s=3, c=color_vector)
        # Increase marker size for mode with maximum peak power
        modalratio = v_rat_mode[condition,:]
        modalratio_valid = modalratio[valid]
        if np.isnan(modalratio_valid).any():
            vel_maxpeak = vels_valid[0]
        else:
            vel_maxpeak = vels_valid[modalratio_valid == 0]
        ax1.scatter(vel_maxpeak, height_km[0], marker='o', s=16,
                    c=color_vector[vels_valid==vel_maxpeak])
        # Mode widths
        widths = v_std[condition,:]
        widths_valid = widths[valid]
        widths_range = np.array([vels_valid - widths_valid,
                                 vels_valid + widths_valid])
        ax1.hlines(height_km, widths_range[0,:], widths_range[1,:],
                   lw=1, color=color_vector)
        # Reflectivities in new panel
        refls = dBZh[condition,:]
        refls_valid = refls[valid]
        ax2.scatter(refls_valid, height_km, marker='+', s=35, c=color_vector)
        # Skewness (and others in another panel)
        skew = v_skew[condition,:]
        skew_valid = skew[valid]
        ax3.scatter(skew_valid, height_km, marker='o', s=9, c='tab:cyan')
        # Mode skewness
        skewmode = v_skew_mode[condition,:]
        skewmode_valid = skewmode[valid]
        ax3.scatter(skewmode_valid, height_km, marker='s',
                    s=11, c='tab:orange')
        # Median skewness
        skewmed = v_skew_med[condition,:]
        skewmed_valid = skewmed[valid]
        ax3.scatter(skewmed_valid, height_km, marker='x', s=20, c='k')
        # Kurtosis
        kurt = v_kurt[condition,:]
        kurt_valid = kurt[valid]
        ax3.scatter(kurt_valid / 5, height_km, marker='|',
                    s=22, c=color_vector)
    # Plot percent of overall signal contained in analyzed signal    
    ax4.plot(Zh_percentage, heights/1000, '-', lw=1.5, c='tab:orange')
    # Settings for plots
    ax4.set_xlabel('Total Zh fraction [%]', c='tab:orange', fontsize=12)
    ax4.tick_params(axis='x', labelcolor='tab:orange', labelsize=12)
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel('Velocity [m s$^{-1}$]', fontsize=13)
    ax1.set_ylabel('Height [km]', fontsize=13)
    # Set up legend
    ax1.plot([], [], 'o-', ms=5, c='tab:blue', label=r'mean $\pm$ std')
    ax1.legend(loc='upper left', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=0.2, labelspacing=0.1, handlelength=2.0,
               framealpha=1, fontsize=13)
    ax2.grid(linestyle=':', linewidth=0.5)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xlabel('Reflectivity [dBZ]', fontsize=13)
    ax3.grid(linestyle=':', linewidth=0.5)
    ax3.tick_params(axis='both', labelsize=12)
    ax3.set_xlabel('Skewness; Kurtosis [$-$]', fontsize=13)
    # Set up legend
    ax3.plot([], [], 'o', ms=5, c='tab:cyan', label='skewness moment')
    ax3.plot([], [], 's', ms=6, c='tab:orange', label='mode skewness')
    ax3.plot([], [], 'x', ms=6, c='k', label='median skewness')
    ax3.plot([], [], '|', ms=9, c='tab:blue', label='kurtosis / 5')
    ax3.legend(loc='upper right', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=0.2, labelspacing=0.1, handlelength=2.0,
               framealpha=1, fontsize=13)
    fig.savefig(figname)
    
    # 2) Multimodal properties
    figname2 = (plot_path + 'multimodal_results_'
                + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
    fig = plt.figure(figsize=(20, 7), dpi=None, tight_layout=True)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax4 = ax3.twiny()
    # plot different modes in different colors and check for peaks_flag
    for height in heights:
        condition = (heights == height)
        ratmode = v_rat_mode[condition,:]
        valid = ~np.isnan(ratmode)
        ratmode_valid = ratmode[valid]
        if ratmode_valid.size == 0:
            continue
        color_vector = modecolors.copy()
        # Option to flag dubious modes
        if flag_modes:
            flag_condition = (peaks_flag[heights==height,:].squeeze() == 1)
            color_vector[flag_condition] = flag_color
        # Multimodal ratio plot
        color_vector = color_vector[:len(ratmode_valid)]
        height_km = np.repeat(height, len(ratmode_valid)) / 1000
        ax1.scatter(ratmode_valid, height_km, marker='o',
                    s=3, c=color_vector)
        # Increase marker size for mode with maximum peak power
        ratmode_maxpeak = ratmode_valid[ratmode_valid==0]
        ax1.scatter(ratmode_maxpeak, height_km[0], marker='o',
                    s=16, c=color_vector[ratmode_valid==ratmode_maxpeak])
        # Bimodal separation
        sepmean = v_sep_mean_norm[condition,:]
        valid = ~np.isnan(sepmean)
        sepmean_valid = sepmean[valid]
        ax2.scatter(sepmean_valid, height_km[:-1], marker='o',
                    s=9, c=color_vector[:-1])
        # Bimodal amplitude
        ampmode = v_amp_mode[condition,:]
        ampmode_valid = ampmode[valid]
        ax3.scatter(ampmode_valid, height_km[:-1], marker='o',
                    s=9, c=color_vector[:-1])
        ampmode_alt = v_amp_mode_alt[condition,:]
        ampmode_alt_valid = ampmode_alt[valid]
        ax4.scatter(ampmode_alt_valid, height_km[:-1], marker='x',
                    s=20, c='tab:orange')
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel('Multimodal ratio [dB]', fontsize=13)
    ax1.set_ylabel('Height [km]', fontsize=13)
    ax2.grid(linestyle=':', linewidth=0.5)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xlabel('Normalized bimodal separation [$-$]', fontsize=13)
    ax3.grid(linestyle=':', linewidth=0.5)
    ax3.tick_params(axis='both', labelsize=12)
    ax3.set_xlabel('Bimodal amplitue [$-$]', fontsize=13)
    ax4.invert_xaxis()
    ax4.set_xlabel('Bimodal amplitude [dB] alternative',
                   c='tab:orange', fontsize=12)
    ax4.tick_params(axis='x', labelcolor='tab:orange', labelsize=12)
    fig.savefig(figname2)
    
    if uncertainty_plots:
        # 3) Uncertainties of modal properties
        fig = plt.figure(figsize=(20, 7), dpi=None, tight_layout=True)
        figname3 = (plot_path + 'modes_uncertainties_'
                    + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax4 = ax2.twiny()
        # plot different modes in different colors and check for peaks_flag
        for height in heights:
            condition = (heights == height)
            unc = unc_mean[condition,:]
            valid = ~np.isnan(unc)
            unc_valid = unc[valid]
            if unc_valid.size == 0:
                continue
            color_vector = modecolors.copy()
            # Option to flag dubious modes
            if flag_modes:
                flag_condition = (peaks_flag[heights==height,:].squeeze() == 1)
                color_vector[flag_condition] = flag_color
            # Uncertainty of mean velocities
            color_vector = color_vector[:len(unc_valid)]
            height_km = np.repeat(height, len(unc_valid)) / 1000
            ax1.scatter(unc_valid, height_km, marker='o',
                        s=3, c=color_vector)
            # Increase marker size for mode with maximum peak power
            modalratio = v_rat_mode[condition,:]
            modalratio_valid = modalratio[valid]
            if np.isnan(modalratio_valid).any():
                unc_maxpeak = unc_valid[0]
            else:
                unc_maxpeak = unc_valid[modalratio_valid==0]
            ax1.scatter(unc_maxpeak, height_km[0], marker='o',
                        s=16, c=color_vector[unc_valid==unc_maxpeak])
            # Uncertainty of mode widths
            uncwidths = unc_std[condition,:]
            uncwidths_valid = uncwidths[valid]
            ax1.scatter(uncwidths_valid, height_km, marker='x',
                        s=20, c=color_vector)
            # Uncertainty of reflectivities in new panel
            uncrefls = unc_dBZh[condition,:]
            uncrefls_valid = uncrefls[valid]
            ax2.scatter(uncrefls_valid, height_km, marker='+',
                        s=35, c=color_vector)
            # Uncertainty of skewness (and others) in another panel
            uncskew = unc_skew[condition,:]
            uncskew_valid = uncskew[valid]
            ax3.scatter(uncskew_valid, height_km, marker='o',
                        s=9, c='tab:cyan')
            # Uncertainty of mode skewness
            uncskewmode = unc_skew_mode[condition,:]
            uncskewmode_valid = uncskewmode[valid]
            ax3.scatter(uncskewmode_valid, height_km, marker='s',
                        s=11, c='tab:orange')
            # Uncertainty of median skewness
            uncskewmed = unc_skew_med[condition,:]
            uncskewmed_valid = uncskewmed[valid]
            ax3.scatter(uncskewmed_valid, height_km, marker='x',
                        s=20, c='k')
            # Uncertainty of kurtosis
            unckurt = unc_kurt[condition,:]
            unckurt_valid = unckurt[valid]
            ax3.scatter(unckurt_valid, height_km, marker='|',
                        s=22, c=color_vector)
        # Plot percent of overall signal contained in analyzed signal    
        ax4.plot(unc_Zhperc, heights/1000, '-', lw=1.5, c='tab:orange')
        # Settings for plots
        ax4.set_xscale('log')
        ax4.set_xlabel('Uncertainty total Zh fraction [% absolute]',
                       c='tab:orange', fontsize=12)
        ax4.tick_params(axis='x', labelcolor='tab:orange', labelsize=12)
        ax1.set_xscale('log')
        ax1.grid(linestyle=':', linewidth=0.5)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.set_xlabel('Velocity uncertainty [%]', fontsize=13)
        ax1.set_ylabel('Height [km]', fontsize=13)
        # Set up legend
        ax1.plot([], [], 'o', ms=5, c='tab:blue', label='mean')
        ax1.plot([], [], 'x', ms=6, c='tab:blue', label='std')
        ax1.legend(loc='upper left', numpoints=1, ncol=1, borderpad=0.2,
                   borderaxespad=0.2, labelspacing=0.1, handlelength=2.0,
                   framealpha=1, fontsize=13)
        ax2.grid(linestyle=':', linewidth=0.5)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.set_xlabel('Reflectivity uncertainty [dB]', fontsize=13)
        ax3.set_xscale('log')
        ax3.grid(linestyle=':', linewidth=0.5)
        ax3.tick_params(axis='both', labelsize=12)
        ax3.set_xlabel('Uncertainty in skewness; kurtosis [%]', fontsize=13)
        # Set up legend
        ax3.plot([], [], 'o', ms=5, c='tab:cyan', label='skewness moment')
        ax3.plot([], [], 's', ms=6, c='tab:orange', label='mode skewness')
        ax3.plot([], [], 'x', ms=6, c='k', label='median skewness')
        ax3.plot([], [], '|', ms=9, c='tab:blue', label='kurtosis')
        ax3.legend(loc='upper right', numpoints=1, ncol=1, borderpad=0.2,
                   borderaxespad=0.2, labelspacing=0.1, handlelength=2.0,
                   framealpha=1, fontsize=13)
        fig.savefig(figname3)
        
        # 4) Uncertainties of multimodal properties
        fig = plt.figure(figsize=(20, 7), dpi=None, tight_layout=True)
        figname4 = (plot_path + 'multimodal_uncertainties_'
                    + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax4 = ax3.twiny()
        # plot different modes in different colors and check for peaks_flag
        for height in heights:
            condition = (heights == height)
            unc_ratmode = unc_rat[condition,:]
            valid = ~np.isnan(unc_ratmode)
            uncrat_valid = unc_ratmode[valid]
            if uncrat_valid.size == 0:
                continue
            color_vector = modecolors.copy()
            # Option to flag dubious modes
            if flag_modes:
                flag_condition = (peaks_flag[heights==height,:].squeeze() == 1)
                color_vector[flag_condition] = flag_color
            # Uncertainty of multimodal ratio
            color_vector = color_vector[:len(uncrat_valid)]
            height_km = np.repeat(height, len(uncrat_valid)) / 1000
            ax1.scatter(uncrat_valid, height_km, marker='o',
                        s=3, c=color_vector)
            # Increase marker size for mode with maximum peak power
            modalratio = v_rat_mode[condition,:]
            modalratio_valid = modalratio[valid]
            if np.isnan(modalratio_valid).any():
                uncrat_maxpeak = uncrat_valid[0]
            else:
                uncrat_maxpeak = uncrat_valid[modalratio_valid==0]
            ax1.scatter(uncrat_maxpeak, height_km[0], marker='o',
                        s=16, c=color_vector[uncrat_valid==uncrat_maxpeak])
            # Uncertainty of bimodal separation
            uncsep = unc_sep[condition,:]
            valid = ~np.isnan(uncsep)
            uncsep_valid = uncsep[valid]
            ax2.scatter(uncsep_valid, height_km[:-1], marker='o',
                        s=9, c=color_vector[:-1])
            # Uncertainty of bimodal amplitude
            uncamp = unc_amp[condition,:]
            uncamp_valid = uncamp[valid]
            ax3.scatter(uncamp_valid, height_km[:-1], marker='o',
                        s=9, c=color_vector[:-1])
            uncamp_alt = unc_amp_alt[condition,:]
            uncamp_alt_valid = uncamp_alt[valid]
            ax4.scatter(uncamp_alt_valid, height_km[:-1], marker='x',
                        s=20, c='tab:orange')
        ax1.grid(linestyle=':', linewidth=0.5)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.set_xlabel('Uncertainty in multimodal ratio [dB]', fontsize=13)
        ax1.set_ylabel('Height [km]', fontsize=13)
        ax2.grid(linestyle=':', linewidth=0.5)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.set_xlabel('Uncertainty in bimodal separation [%]', fontsize=13)
        ax3.grid(linestyle=':', linewidth=0.5)
        ax3.tick_params(axis='both', labelsize=12)
        ax3.set_xlabel('Uncertainty in bimodal amplitue [%]', fontsize=13)
        ax4.set_xlabel('Uncertainty alternative bimodal amplitude [dB]',
                       c='tab:orange', fontsize=12)
        ax4.tick_params(axis='x', labelcolor='tab:orange', labelsize=12)
        fig.savefig(figname4)
    
    
def plot_summary(
        multimodal_analysis, isolated_spectra,
        time, plot_path='./plots/'):
    """
    Plot summary of results from multimodal analysis of DWD birdbath scan.
    
    Plot (isolated) weather spectra and major results from corrsponding
    multimodal analysis.
    
    Args:
        multimodal_analysis (dict):
            Dictionary of multimodal analysis results and metadata included
            (i.e. no spectral reflectivity data).
        
        isolated_spectra (DataFrame):
            Isolated weather spectra, i.e. part of output from 
            filtering.isolate_weather().
            
        time (str):
            Time string of birdbath scan.
            
        plot_path (Path):
            Relative path of directory where output figure should be saved.
            
        
    Returns:
        Figure.
    """
    
    print('plotting summary of postprocessing and multimodal analysis...')
    
    # Create plotting directory if it does not exist
    os.makedirs(plot_path, exist_ok=True)
    
    # Get time of birdbath scan
    birdbath_time = dt.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    
    # Extract data and setup plotting options
    isolated_weather = isolated_spectra.values
    # Heights above radar [km] and velocities [m/s]
    heights = isolated_spectra.columns.values / 1000
    velocities = isolated_spectra.index.values
    max_height_set = multimodal_analysis['analysis_maxheight'] / 1000
    # Compile x and y data for plots with pcolormesh
    xdata, ydata = xydata_pcolormesh(velocities, heights)
    # find minimum and maximum velocity to plot valid Doppler spectra
    velocity_min, velocity_max = velocity_minmax(isolated_spectra)
    # find maximum height to plot for valid Doppler spectra
    max_height_adapt = height_max(isolated_spectra) / 1000
    max_height = np.array([max_height_set, max_height_adapt]).min()
    
    # Figure
    fig = plt.figure(figsize=(21, 7), dpi=None, tight_layout=True)
    figname = (plot_path + 'summary_'
               + birdbath_time.strftime('%Y%m%d_%H%M%S') + '.png')
    ax0 = fig.add_subplot(141)
    ax1 = fig.add_subplot(142)
    ax2 = ax1.twiny()
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    ax5 = ax4.twiny()
    
    # 1) Isolated weather spectra
    spec = ax0.pcolormesh(xdata, ydata, isolated_weather.T,
                          vmin=40, vmax=120, cmap='jet', zorder=1)
    cbaxes = fig.add_axes([0.05, 0.2, 0.01, 0.6]) 
    cb = plt.colorbar(spec, ax=ax0, cax=cbaxes, orientation='vertical',
                      extend='both')
    cb.set_label('Uncalibrated power [dB]', fontsize=13)
    cb.ax.tick_params(labelsize=12)
    ax0.set_xlim([velocity_min, velocity_max])
    ax0.set_ylim([0, max_height])
    ax0.grid(linestyle=':', linewidth=0.5)
    ax0.tick_params(axis='both', labelsize=12)
    ax0.set_xlabel(r'Doppler velocity [m s$^{-1}$]', fontsize=13)
    ax0.set_ylabel(r'Height (above radar) [km]', fontsize=13)
    
    # 2) Extract data and set up plotting options for multimodal analysis
    data_setup = extract_multimodal(multimodal_analysis)
    (modecolors, flag_color, flag_modes, peaks_flag, heights, v_mean, v_std,
     dBZh, Zh_percentage, v_skew, v_kurt, v_skew_med, v_skew_mode, v_rat_mode,
     v_sep_mean_norm, v_amp_mode, v_amp_mode_alt, unc_skew, unc_skew_med,
     unc_skew_mode, unc_mean, unc_std, unc_kurt, unc_dBZh, unc_Zhperc, unc_rat,
     unc_sep, unc_amp, unc_amp_alt) = data_setup
    # Estimate suitable min,max values for multimodal plot
    sepmean_lims = [np.nanmin(v_sep_mean_norm), np.nanmax(v_sep_mean_norm)]
    ampmode_lims = [np.nanmin(v_amp_mode), np.nanmax(v_amp_mode)]
    # plot different modes in different colors and check for peaks_flag
    for height in heights:
        condition = (heights == height)
        vels = v_mean[condition,:]
        valid = ~np.isnan(vels)
        vels_valid = vels[valid]
        if vels_valid.size == 0:
            continue
        color_vector = modecolors.copy()
        # Option to flag dubious modes
        if flag_modes:
            flag_condition = (peaks_flag[heights==height,:].squeeze() == 1)
            color_vector[flag_condition] = flag_color
        # Mean velocities
        color_vector = color_vector[:len(vels_valid)]
        height_km = np.repeat(height, len(vels_valid)) / 1000
        # Mode widths
        widths = v_std[condition,:]
        widths_valid = widths[valid]
        widths_range = np.array([vels_valid - widths_valid,
                                 vels_valid + widths_valid])
        ax1.hlines(height_km, widths_range[0,:], widths_range[1,:],
                   lw=1, color=color_vector)
        # Reflectivities in same panel
        refls = dBZh[condition,:]
        refls_valid = refls[valid]
        ax2.scatter(refls_valid, height_km, marker='o',
                    s=12, c=color_vector)
        # Skewness (and others in another panel)
        skew = v_skew[condition,:]
        skew_valid = skew[valid]
        ax3.scatter(skew_valid, height_km, marker='o',
                    s=12, c=color_vector)
        # Median skewness
        skewmed = v_skew_med[condition,:]
        skewmed_valid = skewmed[valid]
        ax3.scatter(skewmed_valid, height_km, marker='x',
                    s=15, c='k')
        # Kurtosis
        kurt = v_kurt[condition,:]
        kurt_valid = kurt[valid]
        ax3.scatter(kurt_valid / 5, height_km, marker='|',
                    s=30, c=color_vector)
        # Multimodal parameter: bimodal separation
        sepmean = v_sep_mean_norm[condition,:]
        valid = ~np.isnan(sepmean)
        sepmean_valid = sepmean[valid]
        ax4.scatter(sepmean_valid, height_km[:-1], marker='o',
                    s=12, c=color_vector[:-1])
        # Multimodal parameter: Bimodal amplitude
        ampmode = v_amp_mode[condition,:]
        ampmode_valid = ampmode[valid]
        ax5.scatter(ampmode_valid, height_km[:-1], marker='x',
                    s=25, c=color_vector[:-1])
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.set_ylim(ax0.get_ylim())
    ax1.set_yticklabels([])
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel('Doppler velocity [m s$^{-1}$]', fontsize=13)
    # Set up legend
    ax2.plot([], [], 'o', ms=3, c='tab:blue',
             label=r'radar reflectivity')
    ax2.plot([], [], '-', lw=1.2, c='tab:blue',
             label=r'mean velocity $\pm$ SD')
    ax2.legend(bbox_to_anchor=(0.47, 0.9), numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=0.2, labelspacing=0.1, handlelength=2.0,
               framealpha=1, facecolor='w', fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xlabel('Radar reflectivity [dBZ]', fontsize=12)
    ax3.grid(linestyle=':', linewidth=0.5)
    ax3.set_ylim(ax1.get_ylim())
    ax3.set_yticklabels([])
    ax3.tick_params(axis='both', labelsize=12)
    ax3.set_xlabel('Skewness; Kurtosis [$-$]', fontsize=13)
    # Set up legend
    ax3.plot([], [], 'o', ms=3, c='tab:blue', label='skewness')
    ax3.plot([], [], 'x', ms=6, c='k', label='median skewness')
    ax3.plot([], [], '|', ms=9, c='tab:blue', label='kurtosis / 5')
    ax3.legend(bbox_to_anchor=(0.47, 0.9), numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=0.2, labelspacing=0.1, handlelength=2.0,
               framealpha=1, fontsize=12)
    ax4.grid(linestyle=':', linewidth=0.5)
    ax4.set_xlim(sepmean_lims)
    ax4.set_ylim(ax1.get_ylim())
    ax4.tick_params(axis='both', labelsize=12)
    ax4.set_xlabel('Normalized bimodal separation [$-$]', fontsize=13)
    ax4.set_yticklabels([])
    ax5.set_xlabel('Bimodal amplitude [$-$]', fontsize=12)
    ax5.set_xlim(ampmode_lims)
    ax5.tick_params(axis='x', labelsize=12)
    ax5.plot([], [], 'o', ms=3, c='tab:blue', label=r'bimodal separation')
    ax5.plot([], [], 'x', ms=6, c='tab:blue', label=r'bimodal amplitude')
    ax5.legend(bbox_to_anchor=(0.60, 0.9), numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=0.2, labelspacing=0.1, handlelength=2.0,
               framealpha=1, facecolor='w', fontsize=12)
    fig.savefig(figname) 
    