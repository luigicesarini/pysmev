#!/home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python

import os
# os.environ['USE_PYGEOS'] = '0'
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

if __name__=="__main__":
    file_path_input="res/s0019_v3.parquet"
    # Define the file path where you want to save the dictionary
    filename_output = file_path_input.split("/")[-1].split(".")[0]

    file_path_output = f'out/{filename_output}.json'
    TYPE='numpy' # choiches numpy or panda
    S=SMEV(
        threshold=0,
        separation=24,
        # return_period=[100,200],
        # durations=    [15,30]
        return_period=get_return_period(),
        durations=[15,30,45,60,120,180,360,720,1440],
        time_resolution=5
    )
    
    data=pd.read_parquet(file_path_input)
    if TYPE=="pandas":
        """"
        Get ordinary events
        """
        idx_ordinary=S.get_ordinary_events(data=data,dates=data.index,name_col='value')
        """
        Remove short events
        """
        arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary,25)

        dict_param={}
        dict_rp={}


        for d in range(len(S.durations)):

            arr_conv=np.convolve(data.value,np.ones(int(S.durations[d]/S.time_resolution),dtype=int),'same')

            # Create xarray dataset

            ds = xr.Dataset(
                {
                    f'tp{S.durations[d]}': (['time'], arr_conv.reshape(-1)),
                },
                coords={
                    'time':data.index.values.reshape(-1)
                },
                attrs = dict(description = f"Array of {S.durations[d]} minutes precipitation data",unit = '[m]')
            )

            # ds.sel(time=slice(arr_dates[-7,1],arr_dates[-7,0]))[f'tp{durations[d]}'].max(skipna=True).item()
            ll_vals=[ds.sel(time=slice(arr_dates[_,1],arr_dates[_,0]))[f'tp{S.durations[d]}'].max(skipna=True).item() for _ in range(arr_dates.shape[0])]
            ll_yrs=[int(arr_dates[_,1][0:4]) for _ in range(arr_dates.shape[0])]

            # Create xarray dataset
            ds_ams = xr.Dataset(
                {
                    'vals': (['year'], ll_vals),
                },
                coords={
                    'year':ll_yrs
                },
                attrs = dict(description = f"Array of {S.durations[d]} minutes precipitation data", unit = '[m]')
            ) * 60 / S.durations[d]

            
            d_param_year={}
            d_rp_year={}
            for YEAR in np.arange(1991,2021):
                if YEAR not in ds_ams.year.values:
                    continue
                else:    
                    shape,scale=S.estimate_smev_parameters(
                                    ds_ams.sel(year=YEAR).vals.values,
                                    [0.75, 1])
                    

                    smev_RP=S.smev_return_values(S.return_period, shape, scale, n_ordinary_per_year[n_ordinary_per_year.index==YEAR].values.item())
            
                d_param_year.update({f"{YEAR}":{'scale':scale,'shape':shape,'n':int(n_ordinary_per_year[n_ordinary_per_year.index==YEAR].values.item())}})
                d_rp_year.update({f"{YEAR}":list(smev_RP)})
        
            dict_param.update({f"{S.durations[d]}":d_param_year})
            dict_rp.update({f"{S.durations[d]}":d_rp_year})
        
        dict_final={'params':dict_param,'quantiles':dict_rp,'duration':S.durations,'RP':S.return_period}

        print(f"\n{YEAR}")
        # print(f"\nN° of ordinary events: {arr_dates.shape[0]}")
        # print(f"Threshold: {threshold:.2f}")
        # print(f"Separation between events in hours: {separation}")
        # print(f"Avg ordinary events per year: {n_ordinary:.2f}\n")

        df_rp = pd.DataFrame(dict_rp['15'])
        # df_param = pd.DataFrame(dict_param['15']).transpose()
        # df_param['n']=df_param['n'].astype(int)        
        # df_rp.index=S.return_period

        print(df_rp)
    elif TYPE=="numpy":
        """"
        Get ordinary events
        """
        df_arr=np.array(data.value)
        df_dates=np.array(data.index)
        idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates,name_col='value')

        """
        Remove short events
        """
        arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary,25)
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
            # ds.sel(time=slice(arr_dates[-7,1],arr_dates[-7,0]))[f'tp{durations[d]}'].max(skipna=True).item()
            ll_vals=[ds.sel(time=slice(arr_dates[_,1],arr_dates[_,0]))[f'tp{S.durations[d]}'].max(skipna=True).item() for _ in range(arr_dates.shape[0])]
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

            dict_ordinary.update({f"{S.durations[d]}":{'year':ll_yrs,'ordinary':ll_vals}})
            
            d_param_year={}
            d_rp_year={}
            for YEAR in np.arange(1991,2021):
                if YEAR not in ds_ams.year.values:
                    continue
                else:    
                    shape,scale=S.estimate_smev_parameters(
                                    ds_ams.sel(year=YEAR).vals.values,
                                    [0.75, 1])
                    

                    smev_RP=S.smev_return_values(S.return_period, shape, scale, n_ordinary_per_year[n_ordinary_per_year.index==YEAR].values.item())
            
                d_param_year.update({f"{YEAR}":{'scale':scale,'shape':shape,'n':int(n_ordinary_per_year[n_ordinary_per_year.index==YEAR].values.item())}})
                d_rp_year.update({f"{YEAR}":list(smev_RP)})
        
            dict_param.update({f"{S.durations[d]}":d_param_year})
            dict_rp.update({f"{S.durations[d]}":d_rp_year})
        dict_final={'params':dict_param,'quantiles':dict_rp,'duration':S.durations,'RP':S.return_period}

        print(f"\n{YEAR}")
        # print(f"\nN° of ordinary events: {arr_dates.shape[0]}")
        # print(f"Threshold: {threshold:.2f}")
        # print(f"Separation between events in hours: {separation}")
        # print(f"Avg ordinary events per year: {n_ordinary:.2f}\n")

        df_rp = pd.DataFrame(dict_rp['15'])
        # df_param = pd.DataFrame(dict_param['15']).transpose()
        # df_param['n']=df_param['n'].astype(int)        
        # df_rp.index=S.return_period

        print(df_rp)
        # print(df_param)

    # Save the dictionary to a JSON file
    with open(file_path_output, 'w') as f:
        json.dump(dict_final, f)
        
    with open(f'out/{filename_output}_ordevents.json', 'w') as f:
        json.dump(dict_ordinary, f)