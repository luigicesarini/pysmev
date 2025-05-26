=================
PySMEV
=================

PySMEV contains a set of methods to apply the Simplified Metastatistical Extreme Value analysis as presented in:

Presented in:

| Francesco Marra, Davide Zoccatelli, Moshe Armon, Efrat Morin.
| A simplified MEV formulation to model extremes emerging from multiple nonstationary underlying processes.
| Advances in Water Resources, 127, 280-290, 2019
| https://doi.org/10.1016/j.advwatres.2019.04.002
  
| Francesco Marra, Marco Borga, Efrat Morin.
| A unified framework for extreme sub-daily precipitation frequency analyses based on ordinary events. 
| Geophys. Res. Lett., 47, 18, e2020GL090209. 2020.
| https://doi.org/10.1029/2020GL090209 

| The original code of SMEV written in Matlab is available from:
| https://doi.org/10.5281/zenodo.3971557


Installation
------------
For the moment the package is not available on pypi, so you need to install it from the source code.
To do so, clone the repository and run the following command in the root folder of the repository.
  
With Conda 

.. code-block:: bash

    # create pysmev environment
    conda env create -f environment.yml
    # activate pysmev environment
    conda activate pysmev_env
    # install pytenax in editable mode
    python -m pip install -e .



Usage
-----

For a complete example of how to use the class, please see the files in the `example` folder:

| Example for 60-min data `example_smev_1h_obs.py` 
| In this case we use data coming from https://doi.org/10.5281/zenodo.7142384
| Dallan, E. (2022). How well does a convection-permitting climate model represent the reverse orographic effect of extreme hourly precipitation? - Observed precipitation data

| Example for 5-min data `example_smev_5min_obs.py` 
| In this case we use data coming from https://doi.org/10.5281/zenodo.6088847
| Dallan, E., & Marra, F. (2022). Enhanced summer convection explains observed trends in extreme subdaily precipitation in the Eastern Italian Alps - Codes & data (Versione v1). Zenodo. https://doi.org/10.5281/zenodo.6088848


Development
-----------
To build a development environment run:

With Conda 

.. code-block:: bash

    conda env create -f environment.yml
    conda activate pysmev_env
    python -m pip install -e .


Please work on a feature branch and create a pull request to the source branch.
To ensure formatting consistency, please install the pre-commit hooks by running:

.. code-block:: bash

    pre-commit install

If necessary to merge manually do so without fast forward:

.. code-block:: bash

    git merge --no-ff myfeature
	
	

Contributions
-------------

## How to Submit an Issue

We welcome your feedback and contributions! If you encounter a bug, have a feature request, or have any other issue you'd like to bring to our attention, please follow the steps below:

1. **Check for Existing Issues**: Before you submit a new issue, please check if a similar issue already exists in our [issue tracker](https://github.com/luigicesarini/pysmev/issues). If you find an existing issue that matches your concern, you can contribute to the discussion by adding your comments or reactions.

2. **Open a New Issue**: If you don't find an existing issue that matches your concern, you can open a new one by following these steps:
   - Go to the [Issues](https://github.com/luigicesarini/pysmev/issues) section of the repository.
   - Click on the **New Issue** button.
   - Select the appropriate issue template, if available.
   - Fill in the title and description with as much detail as possible. Include steps to reproduce the issue, the expected behavior, and the actual behavior. Providing screenshots or code snippets can be very helpful.
   - Submit the issue.

3. **Follow Up**: After you submit the issue, we might need more information from you. Please stay tuned for our comments and respond promptly if we request additional details.

### Issue Submission Guidelines

- **Be Clear and Descriptive**: Help us understand the issue quickly and thoroughly.
- **Provide Context**: Describe the problem, including the version of the software, operating system, and any other relevant details.
- **Include Screenshots and Logs**: If applicable, add any screenshots, logs, or stack traces that can help diagnose the problem.
- **Use a Consistent and Descriptive Title**: This helps others quickly identify issues that might be similar to theirs.
- **Be Respectful and Considerate**: Keep in mind that we are all part of a community and we aim to create a positive and collaborative environment.

Thank you for helping us improve!

[Open an Issue](https://github.com/luigicesarini/pysmev/issues/new)


Credits
-------

We wish to thank Yaniv Goldschmidt from Hebrew University yanivfry@gmail.com

PySMEV wouldn't be at this stage without pyTENAX community https://github.com/PetrVey/pyTENAX.
Many thanks to:
Ella Thomas https://github.com/ELLAtho
Jannis Hoch https://github.com/JannisHoch
