#!/home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
"""
This is testing file for SMEV modified to 1h resolution of precipitation data.
There is  no option between pandas or numpy currently as in original test file.
It uses numpy version.
Input file is csv file of AA_0200 (AA_0220; x-utm:616720; y-utm:5181393; altitude: 1499)

#TODO: add results of MATLAB version to be compared with this, so later we can run this tests when changing main code.
"""
import os
# os.environ['USE_PYGEOS'] = '0'
from os.path import dirname, abspath, join
import sys
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '/', 'src'))
sys.path.append(CODE_DIR)
import numpy as np 
import xarray as xr 
import pandas as pd
import glob, os
import csv
import time
import pprint
import json

from pysmev import *


if __name__=="__main__":
    file_path_input="res/AA_0220.csv"
    # Define the file path where you want to save the dictionary
    file_path_without_extension = os.path.splitext(os.path.basename(file_path_input))[0]

    TYPE='numpy' # hard coded for numpy type, just info
    
    S=SMEV(
            threshold=0.1,
            separation=24,
            return_period=[2,5,10,20,50,100],
            durations= [x * 60 for x in [1,3,6,12,24]],
            time_resolution=60
        )
       
    #Additional setting that should be later added to S class
    min_duration = 30 #min storm duration, yes it doesn't make sense to 30min if we have hourly data but it's copy of matlab smev
    censoring = [0.85, 1] #Left censoring

    #Load data from csv file
    data=pd.read_csv(file_path_input, parse_dates=True, index_col='time')
    name_col = "precipitation" #name of column containing data to extract
    
    #push values belows 0.1 to 0 in prec
    data.loc[data[name_col] < S.threshold, name_col] = 0
    
    #get data from pandas to numpy array
    df_arr = np.array(data[name_col])
    df_dates=np.array(data.index)
    
    #extract indexes of ordinary events
    #these are time-wise indexes =>returns list of np arrays with np.timeindex
    idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col='precipitation')
    
    #get ordinary events by removing too short events
    #returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
    arr_vals,arr_dates,n_ordinary_per_year, n_oe_mean =S.remove_short(idx_ordinary,min_duration)
    
    #innitiate dictionary to store values of SMEV
    dict_param={}
    dict_rp={}
    dict_ordinary={}
        
    for d in range(len(S.durations)):   
        arr_conv=np.convolve(df_arr,np.ones(int(S.durations[d]/S.time_resolution),dtype=int),'same')
            
           
        # Create xarray dataset
        ds = xr.Dataset(
                {
                    f'tp{S.durations[d]}': (['time'], arr_conv.reshape(-1)),
                },
                coords={
                    'time':df_dates.reshape(-1)
                },
                attrs = dict(description = f"Array of {S.durations[d]} minutes precipitation data",unit = '[m]')
            )

        # Convert time index to numpy array
        time_index = ds['time'].values

        # Use numpy indexing to get the max values efficiently
        ll_vals = []
        for i in range(arr_dates.shape[0]):
            start_time_idx = np.searchsorted(time_index, arr_dates[i, 1])
               
            end_time_idx = np.searchsorted(time_index, arr_dates[i, 0])
                
            # Check if start and end times are the same
            if start_time_idx == end_time_idx:
                ll_val = ds[f'tp{S.durations[d]}'].values[start_time_idx]
            else:
                # the +1 in end_time_index is because then we search by index but we want to includde last as well,
                # without, it slices eg. end index is 10, without +1 it slices 0 to 9 instead of 0 to 10 (stops 1 before)
                ll_val = np.nanmax(ds[f'tp{S.durations[d]}'].values[start_time_idx:end_time_idx+1])
                
            ll_vals.append(ll_val)
            
        #years  of ordinary events
        ll_yrs=[arr_dates[_,1].astype('datetime64[Y]').item().year for _ in range(arr_dates.shape[0])]
            
        # Create xarray dataset 
        ds_ams = xr.Dataset(
                {
                    'vals': (['year'], ll_vals),
                },
                coords={    
                    'year':ll_yrs
                },
                attrs = dict(description = f"Array of {S.durations[d]} minutes precipitation data", unit = '[m]')
            ) * 60 / S.durations[d] #fix this as a parameters. For rainfall with different duration or for other environmetnal varaibles does not make any sense.
        
        #save OE to dictionary
        dict_ordinary.update({f"{S.durations[d]}":{'year':ll_yrs,'ordinary':[x * 60 / S.durations[d] for x in ll_vals]}})
        
        #estimate shape and  scale parameters of weibull distribution
        shape,scale=S.estimate_smev_parameters(ds_ams.vals.values, censoring)
        #calculate mean annual value of ordinary event withing dataset years  
        n_oe = n_ordinary_per_year.values.mean()
        #estimate return period (quantiles) with SMEV
        smev_RP=S.smev_return_values(S.return_period, shape, scale, n_oe)
        #save parameters and quantiles from SMEV to dictionary
        dict_param.update({f"{S.durations[d]}":[scale, shape,n_oe]})
        dict_rp.update({f"{S.durations[d]}":list(smev_RP)})
    
    #print results to console
    print('SMEV parameters')
    print(json.dumps(dict_param, indent=4))
    
    print('SMEV quantiles')
    print(json.dumps(dict_rp, indent=4))
    
    print('n oe from event separation')
    print(n_oe_mean)
    
    #save results of SMEV in dictionary to CSV file
    param_output_path = f'out/{file_path_without_extension}_phat_pysmev.csv'
    rp_output_path = f'out/{file_path_without_extension}_qnt_pysmev.csv'
            
    with open(param_output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for value in dict_param.values():
            writer.writerow(value)
                
    with open(rp_output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for value in  dict_rp.values():
            writer.writerow(value)
           