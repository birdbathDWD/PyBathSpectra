"""
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

from setuptools import setup, find_packages
import pathlib


# Get the long description from the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='PyBathSpectra',
    version='0.1.0',
    description='Spectral postprocessing of DWD radar birdbath scans',
    long_description=long_description,
    # Content types: text/plain, text/x-rst, and text/markdown
    long_description_content_type='text/markdown',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
    author='Mathias Gergely',
    author_email='Mathias.Gergely@dwd.de',
    url='https://github.com/PyBathSpectra',
    packages=find_packages(include=['pybathspectra', 'pybathspectra.*']),
    license='MIT',
    keywords='radar, postprocessing, Doppler, multimodality',
    python_requires='>=3.7',  # 3.6
    install_requires=[
        'h5py>=2.9.0',  # 2.10.0
        'hdbscan>=0.8.24',  # 0.8.27
        'matplotlib>=3.1.1',  # 3.2.1
        'numpy>=1.21.0',  # 1.19.5
        'pandas>=1.2.5',  # 1.1.5
        'PyYAML>=5.4.1',  # 3.12
        'scipy>=1.3.0',  # 1.4.1
        'unidip>=0.1.1',  # 0.1.1
    ],
)
