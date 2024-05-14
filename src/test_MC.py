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
    
    file_path_input="out/s0019_v3.json"
    file_path_input_oridnary="out/s0019_v3_ordevents.json"
    
    with open(file_path_input, 'r') as f:
        data=json.load(f)
    
    with open(file_path_input_oridnary, 'r') as f:
        ordevents=json.load(f)

    n_ordinary=pd.DataFrame(ordevents['15']).groupby('year').count().sum().item()
    
    # print(pd.DataFrame(data['params']['15']).T)
    # print(data['quantiles'])
    # print(data['params']['15'].keys())
    n_iter=1000;  

    max_ev=np.zeros(shape=(len(data['params']['15'].keys()),len(data['duration']),n_iter)) * np.nan
    rand_ev=np.zeros(shape=(n_ordinary,len(data['duration']),n_iter)) * np.nan

    for id,duration in enumerate(data['duration']):
        i_n=0 #progressive event number
        for i_yr,yr in enumerate(list(data['params'][f'{duration}'].keys())):  #for each year
            scale=data['params'][f'{duration}'][f'{yr}']['scale']
            shape=data['params'][f'{duration}'][f'{yr}']['shape']
            n=data['params'][f'{duration}'][f'{yr}']['n']
            a=np.random.random(size=n_iter*n).reshape(n_iter,n)
            randy = wblinv(a,scale,shape) # weibull inversion for computing the events
            tmp=np.max(randy,axis=1)
            print(np.max(tmp))
            # storing the results: AM and all the events per year
            max_ev[i_yr,id,:]=tmp # AM
            rand_ev[i_n:i_n+n ,id,:]=np.transpose(randy) # random-generated events 
            i_n+=n
    

    ams=pd.DataFrame(ordevents['15']).groupby('year').max()
    print(max_ev.shape)
    print(ams)
    fig,ax=plt.subplots(1,1,figsize=(12,4))
    ax.plot(
        ams.index,
        ams.ordinary,
        alpha=0.95,
        marker='*',markersize=5,markerfacecolor='blue'
        )
    [ax.plot(ams.index,max_ev[:,0,_],alpha=0.25,marker='+',markersize=1,markerfacecolor='red') for _ in range(1000)]
    ax.set_ylabel(f"mm/hr",fontsize=18,fontweight=700)
    ax.set_yticklabels(labels=[0,5,10,15,20,25,30],fontsize=14,fontweight=400)
    # ax.set_xticklabels(labels=np.arange(1401,200),fontsize=14,fontweight=400)
    fig.suptitle(f"AMS and generated AMS",fontsize=22,fontweight=700)
    plt.savefig(f"fig/ams.png")
    plt.close()

    fig,ax=plt.subplots(1,1,figsize=(16,4))
    ax.plot(
        # pd.DataFrame(ordevents['15']).year,
        pd.DataFrame(ordevents['15']).ordinary,
        alpha=0.95,
        marker='*',markersize=3,markerfacecolor='blue'
        )
    ax.plot(
        rand_ev[:,0,100] * 4,
        alpha=0.25,
        marker='+',markersize=1,markerfacecolor='red'
    )
    ax.set_ylabel(f"mm/hr",fontsize=18,fontweight=700)
    ax.set_yticklabels(labels=[0,5,10,15,20,25,30],fontsize=14,fontweight=400)
    # ax.set_xticklabels(labels=np.arange(1401,200),fontsize=14,fontweight=400)
    fig.suptitle(f"Ordinary Events and generated ordinary events",fontsize=22,fontweight=700)
    plt.savefig(f"fig/ordinary.png")
    plt.close()