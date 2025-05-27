"""
The Example script of SMEV use for high-resolution (5-min) data in Parquet file.

In this case we use data coming from 10.5281/zenodo.6088847.
Dallan, E., & Marra, F. (2022). Enhanced summer convection explains observed trends in extreme subdaily precipitation in the Eastern Italian Alps - Codes & data (Versione v1). Zenodo. https://doi.org/10.5281/zenodo.6088848

SMEV uses numpy arrays.
"""
from importlib_resources import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import pySMEV
from pysmev import smev, plotting


S_SMEV = smev.SMEV(min_rain=0.1,
                   separation=6, #in hours, shouldn't be larger than max in durations
                   return_period=[2, 5, 10, 20, 50, 100, 200, 500],
                   durations= [10, 20, 30, 60, 120, 180, 360,],
                   time_resolution=5,  # time resolution of dataset in minutes
                   min_event_duration=30, #min duration of storms 30min
                   left_censoring=[0.95, 1],
                   )


file_path_input = files("pysmev.res").joinpath("s0001_v3.parquet")
  
data = pd.read_parquet(file_path_input)
data.index = data.index.astype('datetime64[ns]')

name_col = "value"  # name of column containing data to extract

# Push values belows S.min to 0 in prec, this is due to fact that some datasets are having drizzle issues.
data.loc[data[name_col] < S_SMEV.min_rain, name_col] = 0
# Remove incomplete years in dataset and push all nans to zero
# Years with more than 10 percent of nans or missing values are deleted
# Careful as this function also rewrites the time_resolution based on time difference of first and second time
data = S_SMEV.remove_incomplete_years(data, name_col)

# Get data from pandas to numpy array
df_arr = np.array(data[name_col])
df_dates = np.array(data.index)

# Extract datetime indexes of ordinary events
# These are time-wise indexes => returns list of np arrays with np.timeindex
idx_ordinary = S_SMEV.get_ordinary_events(data=df_arr, 
                                         dates=df_dates, 
                                         name_col=name_col, 
                                         check_gaps=False
                                         )

# Get ordinary events by removing too short events
# Returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
arr_vals, arr_dates, n_ordinary_per_year, n_oe_mean = S_SMEV.remove_short(idx_ordinary)

# Assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
dict_ordinary, dict_AMS = S_SMEV.get_ordinary_events_values(data=df_arr, 
                                                            dates=df_dates, 
                                                            arr_dates_oe=arr_dates)


# Example for return levels of 60min duration 
P = dict_ordinary["60"]["ordinary"]
blocks_id = dict_ordinary["60"]["year"]

# Estimate shape and scale parameters of weibull distribution
# We include S_SMEV.left_censoring but actually it is not needed as it auto reads it from S_SMEV class
smev_shape, smev_scale = S_SMEV.estimate_smev_parameters(P, S_SMEV.left_censoring)

# estimate return period (quantiles) with SMEV for 60min duration
smev_RL = S_SMEV.smev_return_values(S_SMEV.return_period,
                                    smev_shape, 
                                    smev_scale, 
                                    n_oe_mean
                                    )
# estimate uncertainty in RL estimation for 60min duration
smev_RL_unc = S_SMEV.SMEV_bootstrap_uncertainty(P, 
                                               blocks_id=blocks_id,
                                               niter=1000)
# validation figure SMEV vs AMs
AMS = dict_AMS["60"] # yet the annual maxima for 60min duration
plotting.SMEV_FIG_valid(AMS, 
                        S_SMEV.return_period, 
                        smev_RL, 
                        smev_RL_unc,
                        xlimits=[1, 100],
                        ylimits=[0, 100],)
plt.title("The SMEV model validation")
plt.ylabel("60-minute precipitation (mm)")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.show()


# Here example of function which returns smev_shape, smev_scale and smev_RL for all durations defined in the class
total_smev_output = S_SMEV.do_smev_all(dict_ordinary, n_oe_mean)
# Note that all RLs are in depth depth and not in intensities.
# To transfer to intensities you must multiply by 60 and divide by given duration. eg. total_smev_output["1440"]["RLs"]*60/1440
print(total_smev_output)
