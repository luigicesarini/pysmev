=================
PySMEV
=================

PySMEV contains a set of methods to apply the Simplified Metastatistical Extreme  
Value analysis as implementd in:

Dallan, E., & Marra, F. (2022). Enhanced summer convection explains observed trends in extreme subdaily precipitation in the Eastern Italian Alps - Codes & data (Versione v1). Zenodo. [https://doi.org/10.5281/zenodo.6088848](https://doi.org/10.5281/zenodo.6088848)


Installation
------------
Install using:

.. code-block:: python

   pip install __

Usage
-----
Example folder... 
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

    from smev import *

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

