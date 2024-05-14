=================
PySMEV
=================

PySMEV contains a set of methods to apply the Simplified Metastatistical Extreme  
Value analysis as implementd in:

Dallan, E., & Marra, F. (2022). Enhanced summer convection explains observed trends in extreme subdaily precipitation in the Eastern Italian Alps - Codes & data (Versione v1). Zenodo. https://doi.org/10.5281/zenodo.6088848


Installation
------------
Install using:



For the moment the package is not available on pypi, so you need to install it from the source code.
To do so, clone the repository and run the following command in the root folder of the repository:

.. code-block:: bash
   pip install .

Usage
-----

The package contains a class called SMEV that can be used to apply the Simplified Metastatistical Extreme Value analysis.
The class is initialized with the following parameters:

- threshold: the threshold above which the data is considered extreme
- separation: the separation between the data points in the time series
- return_period: the return period for which the analysis is performed
- durations: the durations for which the analysis is performed
- time_resolution: the time resolution of the data

The class contains the following methods:

!! TO COMPLETE !!

The following is an example of how to use the class:

.. code-block:: python

    import os
    from os.path import dirname, abspath, join
    import sys
    THIS_DIR = dirname(__file__)
    CODE_DIR = abspath(join(THIS_DIR, '/', 'src'))
    sys.path.append(CODE_DIR)
    import json
    import argparse
    import numpy as np 
    import xarray as xr 
    import pandas as pd
    from glob import glob
    from tqdm import tqdm
    import matplotlib as mpl
    import matplotlib.pyplot as plt 
    from scipy.stats import genextreme as gev

    from pysmev import *

    file_path_input="res/s0019_v3.parquet"
    # Define the file path where you want to save the dictionary
    filename_output = file_path_input.split("/")[-1].split(".")[0]

    file_path_output = f'out/{filename_output}.json'
    TYPE='numpy' # choiches numpy or panda
    S=SMEV(
        threshold=0,
        separation=24,
        return_period=get_return_period(),
        durations=[15,30,45,60,120,180,360,720,1440],
        time_resolution=5
    )

For a complete example of how to use the class, run the file `test_smev.py` in the `src` folder with the following command:

.. code-block:: bash

    python src/test_smev.py



Development
-----------
Please work on a feature branch and create a pull request to the development 
branch. If necessary to merge manually do so without fast forward:

.. code-block:: bash

    git merge --no-ff myfeature

To build a development environment run:

.. code-block:: bash

    python3 -m venv env 
    source env/bin/activate 
    pip install -e .
    pip install -r requirements.txt


Credits
-------

