"""
Class for loading and postprocessing DWD birdbath scan data.

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

from pybathspectra import reader, filtering, modes, plotting, saver


class BirdBathScan:
    def __init__(self):
        self.radar_data = dict()
        self.weather = dict()
        self.results = dict()
        
    def load(self, birdbath_time, loader_settings):
        """
        Load birdbath spectra and moments into dictionary.
        
        Needs birdbath timestamp and settings for reader module functions.
        """
        
        loaded = reader.load_birdbath(
            birdbath_time,
            loader_settings)
        
        self.radar_data['birdbath_moments'] = loaded['birdbath_moments']
        self.radar_data['birdbath_spectra'] = loaded['birdbath_spectra']
        self.radar_data['analysis_minheight'] = loaded['analysis_minheight']
        self.radar_data['structuring_element'] = loaded['structuring_element']
        
    def isolate(self, isolater_settings):
        """
        Isolate weather signal in radar Doppler spectra.
        
        Needs settings for filtering module functions.
        """
        
        isolated = filtering.isolate_weather(
            self.radar_data,
            isolater_settings)
        
        self.weather['isolated_weather'] = isolated['isolated_weather']
        self.weather['radar_measured'] = isolated['radar_measured']
        self.weather['thresholding'] = isolated['thresholding']
        
    def analyze(self, analysis_settings):
        """
        Multimodal analysis of weather signal.
        
        Needs settings for modes module functions.
        """
        
        analyzed = modes.multimodal_analysis(
            self.weather,
            self.radar_data,
            analysis_settings)
        
        self.results['multimodal_analysis'] = analyzed['multimodal_analysis']
        self.results['spectral_zh'] = analyzed['spectral_zh']

    def plot(self, birdbath_time, rgb_settings, plot_directory):
        """
        Visualize radar data, weather spectra, results of multimodal analysis.
        
        Needs timestamp, scaling factors for rgb plots from config file, 
        and (relative) path to directory where to save plots as .png files.
        """
        
        # Overview of radar output; moments and spectra
        plotting.plot_birdbath(
            self.radar_data,
            birdbath_time,
            plot_path=plot_directory)
        # Plot radar output in RGB and RB pseudo-colors
        plotting.plot_birdbath_rgb(
            self.radar_data,
            birdbath_time,
            scaling=rgb_settings,
            plot_path=plot_directory)
        # Plot isolated weather signal
        plotting.plot_weather(
            self.weather,
            birdbath_time,
            plot_path=plot_directory)
        # Detailed plots of (many) results of multimodal analysis
        plotting.plot_multimodal(
            self.results['multimodal_analysis'],
            birdbath_time,
            plot_path=plot_directory)
        # Plot isolated weather singal and summary of multimodal analysis
        plotting.plot_summary(
            self.results['multimodal_analysis'],
            self.weather['isolated_weather'], 
            birdbath_time,
            plot_path=plot_directory)
        
    def save(self, birdbath_time, save_directory):
        """
        Save results of multimodal analysis and spectral reflectivities.
        
        Needs birdbath timestamp and (relative) path to directory 
        where to save results in .txt files.
        """
        
        saver.save_results(
            self.results,
            birdbath_time,
            output_path=save_directory)
