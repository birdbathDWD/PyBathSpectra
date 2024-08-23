"""
Functions for loading DWD birdbath scan data.

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
import h5py
import pandas as pd
import numpy as np
import datetime as dt


def split_settings(all_settings):
    """
    Split all postprocessing settings into 3 groups for different use cases.
    
    Use outputs as inputs to the reader-, filter-, and multimodal functions.
    
    Args:
        all_settings (dict):
            Dictionary of all settings collected from config.yaml file.
            
            
    Returns:
        3 dictionaries of groups of settings to use as input to reader-,
        filter-, and multimodal routine, respectively.
    """
    
    # For reader module
    reader_settings = dict(
        birdbath_pattern=all_settings['birdbath_pattern'],
        data_dir=all_settings['data_dir'],
        structuring_element=all_settings['structuring_element'],
        analysis_minheight=all_settings['analysis_minheight'])
    # Attach 'structuring_element' and 'analysis_minheight' to reader output
    
    # For filtering module
    filter_settings = dict(
        thresholding=all_settings['thresholding'],
        fixed_thresholds=all_settings['fixed_thresholds'],
        dfth_minarea=all_settings['dfth_minarea'],
        interpolate_isolated=all_settings['interpolate_isolated'],
        mask=all_settings['mask'])
    # Attach 'thresholding' to filtering output
    
    # For modes module
    modes_settings = dict(
        peak_finder=all_settings['peak_finder'],
        peak_to_trough=all_settings['peak_to_trough'],
        analysis_maxheight=all_settings['analysis_maxheight'],
        power_mode=all_settings['power_mode'],
        reflectivity_scale=all_settings['reflectivity_scale'],
        maximum_modes=all_settings['maximum_modes'],
        estimate_uncertainty=all_settings['estimate_uncertainty'],
        reflectivity_spectra=all_settings['reflectivity_spectra'])
    # Attach 'analysis_maxheight' and 'estimate_uncertainty' to modes output
                 
    return reader_settings, filter_settings, modes_settings


def read_spectra(path, moment_list=None):
    """
    Load Doppler spectra from DWD birdbath scan.
    
    This function reads DFT spectra from GAMIC hdf5 radar files and operational
    ras20 hdf5 files into a pandas DataFrame. The moments (e.g. 'DFTh' spectral
    coefficients) that should be read can be specified, or all moments 
    (so far, only DFTh and DFTv?) included in the hdf5 data files are read.
    
    Args:
        path (str or list):
            Path to the data. One can also provide a list of paths. 
            Each path should specify exactly one file that should be read.
            Note: The data from all files is aggregated into one DataFrame.
            
        moment_list (str or list):
            List of moment names (here: spectral coefficients) that should be
            extracted from the specified files. One can also provide a single
            string specifying only one moment to be read. 
            If "None" is provided (default) all moments will be read.
            
            
    Returns:
        df (DataFrame):
            All spectral coefficients (for all 1024 Doppler velocities) 
            for all radar rays (15 for 1 birdbath scan) at all height bins.
            
        fh (h5py file):
            hdf5 file object that contains radar output, settings, metadata.
    """
    
    if not isinstance(path, list):
        path = [path]

    if moment_list is not None and not isinstance(moment_list, list):
        moment_list = [moment_list]

    df = None
    for p in path:
        fh = h5py.File(p, 'r')
        # Skip file if it is empty
        if len(fh.keys()) == 0:
            print('skipped empty file {}...'.format(p))
            continue
        # Extract scan counts
        num_scans = fh['what'].attrs['sets']
        for ns in range(num_scans):
            scan = fh['scan{}'.format(ns)]
            print('...loading dft spectra...')
            # Extract range in meters
            attr = scan['how']['extended'].attrs
            r = 1e3 * np.linspace(
                start=np.float(attr['DFTRangeStart'].decode()),
                stop=np.float(attr['DFTRangeStop'].decode()),
                num=scan['moment_0'].shape[1]
            )
            # Calculate height above radar
            el = pd.DataFrame(np.array(scan['ray_header']))
            el = el.loc[:, ['elevation_start', 'elevation_stop']]
            el = el.mean().mean()
            h = r * np.sin(el / 180 * np.pi)
            # Extract moments count
            num_moments = scan['what'].attrs['descriptor_count']
            # Go through all contained moments
            for nm in range(num_moments):
                # Extract moment
                moment = scan['moment_{}'.format(nm)]
                # Get moment name
                moment_name = moment.attrs['moment'].decode()
                # Skip moment if moment name is not in moment list
                if moment_list is not None:
                    if moment_name not in moment_list:
                        continue

                # Extract data
                df_part = pd.DataFrame(np.array(moment))
               
                # Collect info needed to compute correct data values
                dyn_range = [
                    moment.attrs['dyn_range_min'],
                    moment.attrs['dyn_range_max'],
                ]
                if moment.dtype == np.uint8:
                    cur_range_max = 255
                elif moment.dtype == np.uint16:
                    cur_range_max = 65535
                elif moment.dtype == np.uint32:
                    cur_range_max = 4294967295
                elif moment.dtype == np.uint64:
                    cur_range_max = 18446744073709551615
                # Data values
                if moment.attrs['format'].decode() == 'F':
                    df_part = df_part
                else:
                    # If GAMIC value is 0, then thresholded 
                    # (should not happen for Doppler spectra)
                    df_part = df_part.astype(float)
                    df_part[df_part==0] = np.nan
                    # Calculate decoded moment values by correct scaling
                    range_diff = dyn_range[1] - dyn_range[0]
                    factor_mult = range_diff / (cur_range_max - 1)
                    factor_add = dyn_range[0]
                    df_part = (df_part - 1) * factor_mult + factor_add
                
                # Compute data range
                prf = scan['how'].attrs['PRF']
                lam = scan['how'].attrs['radar_wave_length']
                vr_max = prf * lam / 4
                # Aggregate dft (adds all dfts to a multi-index DataFrame)
                rh = pd.DataFrame(np.array(scan['ray_header']))
                if 'dft_item_row_idx' in rh and 'dft_item_row_count' in rh:
                    dft_starts = rh['dft_item_row_idx']
                    dft_row_counts = rh['dft_item_row_count']
                else:
                    raise KeyError('File is missing dft information.')
                df_rays = None
                dft_num = 0
                for start, num_rows in zip(dft_starts, dft_row_counts):
                    df_ray = df_part.iloc[start:start + num_rows, :]
                    df_ray = np.reshape(df_ray.values, (len(r), num_rows))
                    df_ray = pd.DataFrame(df_ray.T)
                    df_ray['dft_num'] = dft_num
                    dft_num += 1
                    df_ray['velocity'] = np.linspace(
                        -vr_max, vr_max, num=df_ray.shape[0]
                    )
                    df_ray['moment'] = moment_name
                    df_ray = df_ray.set_index(
                        ['dft_num', 'velocity', 'moment']
                    )
                    if df_rays is None:
                        df_rays = df_ray
                    else:
                        df_rays = pd.concat([df_rays, df_ray])
                df_part = df_rays
                    
                # Add the new data to the DataFrame
                df_part.columns = np.round(h, 1)
                df_part.columns = df_part.columns.set_names('height')
                if df is None:
                    df = df_part
                else:
                    df = pd.concat([df, df_part])             
                    
    return df, fh


def read_moments(path, moment_list=['Zh'], stat='mean', moment_format='GAMIC'):
    """
    Load radar moments from DWD birdbath scan.
    
    This function reads radar variables from operational ras11 and ras13
    ODIM hdf5 files or GAMIC hdf5 files into a pandas DataFrame. 
    The moments that should be read have to be specified.
    
    Args:
        path (str or list):
            Path to the data. One can also provide a list of paths. 
            Each path should specify exactly one file that should be read.
            Note: The data from all files is aggregated into one dataframe.
            
        moment_list (str or list):
            List of radar variable (GAMIC) names that should be extracted from
            the specified files. One can also provide a single string 
            specifying only 1 moment to be read. "None" is not a valid choice.
            
        stat (str):
            Indicates what statistic is applied to summarize moments over the
            full azimuthal range of 15 rays:
            averaging over all azimuths (stat="mean"),
            standard deviation (stat="std"),
            mean of absolute values of differences from mean (stat="absdiff"), 
            median (stat="median").
            
        moment_format (str):
            Specifies the type of radar hdf5 moment file to read. 
            Either 'GAMIC' or anything else that is 'ODIM'-compatible, 
            e.g. ras11 or ras13 files. 
            
            
    Returns:
        df (DataFrame):
            All specified moments at all height bins 
            (given as averages over all 15 rays, thresholded values ignored).
            
        fh (h5py file):
            hdf5 file object that contains radar output, settings, metadata.
    """
    
    if not isinstance(path, list):
        path = [path]
        
    if moment_list is None:
        raise KeyError('Specify moment_list. None is not supported here')
        
    if moment_list is not None and not isinstance(moment_list, list):
        moment_list = [moment_list]
    
    # Allow switching between (some) GAMIC and ODIM moment names
    gamic_list = moment_list.copy()
    odim_list = gamic_list.copy()
    gamic_different = [
        'Zh', 'Zv', 'UZh', 'UZv', 'Vh',
        'UVFh', 'SNRh', 'SNRv', 'Wh',
    ]
    odim_different = [
        'DBZH', 'DBZV', 'TH', 'TV', 'VRADH',
        'UFVRADH', 'DBSNRH', 'DBSNRV', 'WRADH',
    ]
    for index_all, gamic_name in enumerate(gamic_list):
        if gamic_name in gamic_different:
            index_different = gamic_different.index(gamic_name)
            odim_list[index_all] = odim_different[index_different]
    
    # If file is in GAMIC file format
    if moment_format == 'GAMIC':
        df = None
        for p in path:
            fh = h5py.File(p, 'r')
            # Skip file if it is empty
            if len(fh.keys()) == 0:
                print('skipped empty file {}...'.format(p))
                continue
            # Extract timestamp
            timestamp = pd.to_datetime(fh['what'].attrs['date'].decode())
            # Extract scan counts
            num_scans = fh['what'].attrs['sets']
            for ns in range(num_scans):
                scan = fh['scan{}'.format(ns)]
                # Compute height to use as column labels
                print('...loading radar moments...')
                # Extract range in meters
                r = (
                    scan['how'].attrs['range_start']
                    + np.arange(scan['how'].attrs['bin_count'])
                    * scan['how'].attrs['range_step']
                )
                # Calculate height
                el = pd.DataFrame(np.array(scan['ray_header']))
                el = el.loc[:, ['elevation_start', 'elevation_stop']]
                el = el.mean().mean()
                h = r * np.sin(el / 180 * np.pi)
                # Extract moments count
                num_moments = scan['what'].attrs['descriptor_count']
                # Go through all contained moments
                for nm in range(num_moments):
                    # Extract moment
                    moment = scan['moment_{}'.format(nm)]
                    # Get moment name
                    moment_name = moment.attrs['moment'].decode()
                    # Skip moment if moment name is not in moment list
                    if moment_list is not None:
                        if moment_name not in moment_list:
                            continue
    
                    # Extract data
                    df_part = pd.DataFrame(np.array(moment))
                    
                    # Collect info needed to compute correct data values
                    dyn_range = [
                        moment.attrs['dyn_range_min'],
                        moment.attrs['dyn_range_max'],
                    ]
                    if moment.dtype == np.uint8:
                        cur_range_max = 255
                    elif moment.dtype == np.uint16:
                        cur_range_max = 65535
                    elif moment.dtype == np.uint32:
                        cur_range_max = 4294967295
                    elif moment.dtype == np.uint64:
                        cur_range_max = 18446744073709551615
                    else:
                        raise ValueError('Unexpected data type.')
                    # If GAMIC moment value is 0, then thresholded
                    df_part = df_part.astype(float)
                    df_part[df_part==0] = np.nan
                    # Calculate decoded moment values by correct scaling 
                    range_diff = dyn_range[1] - dyn_range[0]
                    factor_mult = range_diff / (cur_range_max - 1)
                    factor_add = dyn_range[0]
                    df_part = (df_part - 1) * factor_mult + factor_add
                    valid = pd.notna(df_part).astype(bool) 
                    
                    # Aggregate sweepdata for one timestamp
                    # Different summary statistics
                    dB_moments = [
                        'Zh', 'Zv', 'UZh', 'UZv',
                        'SNRh', 'SNRv', 'ZDR', 'UZDR',
                    ]
                    if stat == 'mean':
                        df_part = df_part[valid].mean(axis=0)
                    elif stat == 'std':
                        if moment_name in dB_moments:
                            df_part = df_part[valid].std(axis=0, ddof=1)
                        else:
                            df_part_mean = df_part[valid].mean(axis=0)
                            df_part_std = df_part[valid].std(axis=0, ddof=1)
                            df_part = df_part_std / df_part_mean
                    elif stat == 'absdiff':
                        df_part_mean = df_part[valid].mean(axis=0)
                        df_part_diffs = df_part[valid] - df_part_mean
                        if moment_name in dB_moments:
                            df_part = np.abs(df_part_diffs).mean()
                        else:
                            df_part = np.abs(df_part_diffs / df_part_mean)
                            df_part = df_part.mean()
                    elif stat == 'median':
                        df_part = np.median(df_part[valid], axis=0)
                    else:
                        raise KeyError('Illegal summary statistic of'
                                       ' moments specified.')
                    # Reset to pd.DataFrame to add additional information
                    df_part = pd.DataFrame(df_part).T
                    df_part['date'] = timestamp
                    df_part['moment'] = moment_name
                    df_part = df_part.set_index(['date', 'moment'])
                        
                    # Add the new data to the DataFrame
                    df_part.columns = np.round(h, 1)
                    df_part.columns = df_part.columns.set_names('height')
                    if df is None:
                        df = df_part
                    else:
                        df = pd.concat([df, df_part])
    
    # If not 'GAMIC' file format, then it should be ODIM ras11 or ras13
    elif moment_format == 'ODIM':
        df = None
        for p in path:
            fh = h5py.File(p, 'r')
            # Skip file if it is empty
            if len(fh.keys()) == 0:
                print('skipped empty file {}...'.format(p))
                continue
            # Extract timestamp
            timestamp = pd.to_datetime(
                fh['what'].attrs['date'].decode()
                + fh['dataset1']['what'].attrs['endtime'].decode())
            # Extract scan counts (not given in ODIM files, so find it)
            fh_list = list(fh.keys())
            fh_substring = 'dataset'
            fh_stringlist = list(filter(lambda x: fh_substring in x, fh_list))
            num_scans = len(fh_stringlist)
            for ns in range(1,num_scans+1):
                scan = fh['dataset{}'.format(ns)]
                # Compute height to use as column labels
                print('...loading radar moments...')
                # Extract range in meters
                r = (
                    scan['where'].attrs['rstart']
                    + np.arange(scan['where'].attrs['nbins'])
                    * scan['where'].attrs['rscale']
                )
                # Calculate height
                el = scan['where'].attrs['elangle']
                h = r * np.sin(el / 180 * np.pi)
                # Extract moments count (not given in ODIM files, so find it)
                key_list = list(scan.keys())
                substring = 'data'
                string_list = list(filter(lambda x: substring in x, key_list))
                num_moments = len(string_list)
                # Go through all contained moments
                for nm in range(1,num_moments+1):
                    # Extract moment
                    moment = scan['data{}'.format(nm)]
                    # Get moment name
                    moment_name = moment['what'].attrs['quantity'].decode()
                    # Skip moment if moment name is not in the moment list
                    if moment_list is not None:
                        if moment_name not in odim_list:
                            continue
    
                    # Extract data
                    df_part = pd.DataFrame(np.array(moment['data']))
                    
                    # Collect info needed to compute correct data values
                    # threshold undetected and nodata values in ODIM files
                    undetected = moment['what'].attrs['undetect']
                    nodata = moment['what'].attrs['nodata']
                    df_part = df_part.astype(float)
                    invalid = ((df_part == undetected) | (df_part == nodata))
                    df_part[invalid] = np.nan
                    # Calculate decoded moment values by correct scaling 
                    factor_mult = moment['what'].attrs['gain']
                    factor_add = moment['what'].attrs['offset']
                    df_part = df_part * factor_mult + factor_add
                    valid = pd.notna(df_part).astype(bool)
                    
                    # Aggregate sweepdata for one timestamp
                    # Different summary statistics
                    dB_moments_odim = [
                        'DBZH', 'DBZV', 'TH', 'TV',
                        'DBSNRH', 'DBSNRV', 'ZDR', 'UZDR',
                    ]
                    dB_moments = dB_moments_odim
                    if stat == 'mean':
                        df_part = df_part[valid].mean(axis=0)
                    elif stat == 'std':
                        if moment_name in dB_moments:
                            df_part = df_part[valid].std(axis=0, ddof=1)
                        else:
                            df_part_mean = df_part[valid].mean(axis=0)
                            df_part_std = df_part[valid].std(axis=0, ddof=1)
                            df_part = df_part_std / df_part_mean
                    elif stat == 'absdiff':
                        df_part_mean = df_part[valid].mean(axis=0)
                        df_part_diffs = df_part[valid] - df_part_mean
                        if moment_name in dB_moments:
                            df_part = np.abs(df_part_diffs).mean()
                        else:
                            df_part = np.abs(df_part_diffs / df_part_mean)
                            df_part = df_part.mean()
                    elif stat == 'median':
                        df_part = np.median(df_part[valid], axis=0)
                    else:
                        raise KeyError('Illegal summary statistic of'
                                       ' moments specified.')
                    # Reset to pd.DataFrame to add information to data
                    df_part = pd.DataFrame(df_part).T
                    df_part['date'] = timestamp
                    # Moment_name for GAMIC naming convention
                    if moment_name in odim_different:
                        index_different = odim_different.index(moment_name)
                        moment_name = gamic_different[index_different]
                    df_part['moment'] = moment_name
                    df_part = df_part.set_index(['date', 'moment'])
                        
                    # Add the new data to the DataFrame
                    df_part.columns = np.round(h, 1)
                    df_part.columns = df_part.columns.set_names('height')
                    if df is None:
                        df = df_part
                    else:
                        df = pd.concat([df, df_part])
    else:
        raise KeyError('Illegal moment file pattern/data format.')
                    
    return df, fh


def load_birdbath(birdbath_time, reader_settings):
    """
    Load DWD birdbath data required for spectral multimodal postprocessing.
    
    Args:
        birdbath_time (str):
            Timestamp of radar birdbath scan as '%Y-%m-%d %H:%M:%S'.
            
        reader_settings (dict):
            Dictionary of settings used for reading birdbath scan data and
            attaching metadata to birdbath_data output.
    
    
    Returns:
        birdbath_data (dict):
            Dictionary of 2 DataFrames: 'birdbath_spectra' and 
            'birdbath_moments', and 2 metadata keys included directly
            for easier further processing.
    """
    
    print('loading birdbath scan data for ' + birdbath_time + ' ...')
    
    # Timestamp of birdbath scan for analysis
    analysis_time = dt.datetime.strptime(birdbath_time, '%Y-%m-%d %H:%M:%S')
    
    # Input data directory
    base_path = reader_settings['data_dir']
    
    # Birdbath scan Doppler spectra file
    spectra_pattern = reader_settings['birdbath_pattern']['birdbath_spectra']
    spectra_file = analysis_time.strftime(spectra_pattern)
    spectra_file = os.path.join(base_path, spectra_file)
    
    # Birdbath scan moments file
    moment_pattern = reader_settings['birdbath_pattern']['birdbath_moments']
    moment_file = analysis_time.strftime(moment_pattern)
    moment_file = os.path.join(base_path, moment_file)
    
    # Load spectra
    df_spectra, fh_spectra = read_spectra(spectra_file)
    
    # Load reflectivity moment(s) for calibration of spectral power
    # determine moment format: GAMIC or ODIM
    if moment_pattern.startswith('Sc+'):
        format_id = 'GAMIC'
    elif moment_pattern.startswith('ras'):
        format_id = 'ODIM'
    else:
        raise KeyError('Illegal moment file pattern/data format.')
    # load data
    df_moments, fh_moments = read_moments(
        moment_file,
        moment_list=['UZh', 'Zh', 'URHOHV', 'Vh', 'UVFh', 'SNRh'],
        stat='mean',
        moment_format=format_id)
    
    # Combine data and metadata into single dictionary for further processing
    birdbath_data = dict(
        birdbath_spectra=df_spectra,
        birdbath_moments=df_moments,
        structuring_element=reader_settings['structuring_element'],
        analysis_minheight=reader_settings['analysis_minheight'])
    
    return birdbath_data
