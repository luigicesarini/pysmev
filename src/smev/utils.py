import math
import numpy as np
import pandas as pd
import statsmodels.api as sm



def get_unlist(ll:list):
    ul=[]
    for sublist in ll:
        for file in sublist:
            ul.append(file)
    return ul

# def estimate_smev_parameters(ordinary_events_df, pr_field, data_portion):
#     """
    
#     Function that estimates parameters of the Weibull distribution  

#     Parameters
#     ----------
#     - ordinary_events_df (dataframe): Pandas dataframe of the ordinary events - withot zeros!!!
#     - pr_field (string): The name of the df column with precipitation values.
#     - data_portion (list): 2-elements list with the limits in probability of the data to be used for the parameters estimation
#       e.g. data_portion = [0.75, 1] uses the largest 25% values in data 
    
#     Returns
#     -------
#     - weibull_param (list): [shape,scale]

#     Examples
#     --------
#     """
    
#     sorted_df = np.sort(ordinary_events_df) #[pr_field].values
#     ECDF = (np.arange(1, 1 + len(sorted_df)) / (1 + len(sorted_df)))
#     fidx = max(1, math.floor((len(sorted_df)) * data_portion[0]))  
#     tidx = math.ceil(len(sorted_df) * data_portion[1])  
#     to_use = np.arange(fidx - 1, tidx)  
#     to_use_array = sorted_df[to_use]

#     X = (np.log(np.log(1 / (1 - ECDF[to_use]))))  
#     Y = (np.log(to_use_array))  
#     X = sm.add_constant(X)  
#     model = sm.OLS(Y, X)
#     results = model.fit()
#     param = results.params

#     slope = param[1]
#     intercept = param[0]

#     shape = 1 / slope
#     scale = np.exp(intercept)

#     weibull_param = [shape, scale]

#     return weibull_param

# def smev_return_values(return_period, shape, scale, n):
#     """
    
#     Function that calculates return values acoording to parameters of the Weibull distribution    

#     Parameters
#     ----------
#     - return_period (int): The desired return period for which intensity is calculated.
#     - shape (float): Weibull distribution shape parameter
#     - scale (float): Weibull distribution scale parameter
#     - n (float): Mean number of ordinary events per year 
    
#     Returns
#     -------
#     - intensity (float): The corresponding intensity value. 

#     Examples
#     --------
#     """

#     return_period = np.asarray(return_period)
#     quantile = (1 - (1 / return_period))
#     if shape == 0 or n == 0:
#         intensity = 0
#     else:
#         intensity = scale * ((-1) * (np.log(1 - quantile ** (1 / n)))) ** (1 / shape)

#     return intensity

# def get_ordinary_events(data,name_col, threshold, separation):
#     """
    
#     Function that extracts ordinary precipitation events out of the entire data.
    
#     Parameters
#     ----------
#     - data np.array: array containing the hourly values of precipitation.
#     - separation (int): The number of hours used to define an independet ordianry event. Defult: 24 hours.
#                     Days with precipitation amounts above this threshold are considered as ordinary events.
#     - pr_field (string): The name of the df column with precipitation values.
#     - hydro_year_field (string): The name of the df column with hydrological years values.

#     Returns
#     -------
#     - consecutive_values np.array: index of time of consecutive values defining the ordinary events.


#     Examples
#     --------
#     """
    
#     # Find values above threshold
#     above_threshold = data[data[name_col] > threshold]
#     # Find consecutive values above threshold separated by more than 24 observations
#     consecutive_values = []
#     temp = []
#     for index, row in above_threshold.iterrows():
#         if not temp:
#             temp.append(index)
#         else:
#             if index - temp[-1] > pd.Timedelta(hours=separation):
#                 if len(temp) >= 1:
#                     consecutive_values.append(temp)
#                 temp = []
#             temp.append(index)
#     if len(temp) >= 1:
#         consecutive_values.append(temp)



#     # # Find values above threshold
#     # above_threshold_indices = np.where(df['value'] > threshold)[0]

#     # # Find consecutive values above threshold separated by more than 24 observations
#     # consecutive_values = []
#     # temp = []
#     # for index in above_threshold_indices:
#     #     if not temp:
#     #         temp.append(index)
#     #     else:
#     #         if index - temp[-1] > 24:
#     #             if len(temp) > 1:
#     #                 consecutive_values.append(temp)
#     #             temp = []
#     #         temp.append(index)
#     # if len(temp) > 1:
#         # consecutive_values.append(temp)
#     return consecutive_values

def plotting_position(N):
    
    P = np.arange(1,N+1)/(N+1)
    
    return P

def get_return_period():
    """
    
    Function that return a sort array of RP.
    
    Parameters
    ----------


    Returns
    -------


    Examples
    --------
    """
    arr_1=np.exp(np.arange(np.log(1.1),np.log(250),0.1)) 
    arr_2=np.array([2, 5, 10, 20, 25, 50, 100, 200, 250, 500])
    #concatenate arr_1 and arr_2
    arr=np.concatenate([arr_1, arr_2])
    #sort array
    return list(np.sort(arr))


# def remove_short(list_ordinary:list,min_duration:int):
#     """
    
#     Function that removes ordinary events too short.
    
#     Parameters
#     ----------
#     - list_ordinary list: list of indices of ordinary events as returned by `get_ordinary_events`.
#     - min_duration (int): minimun number of minutes tto define an event.
#     - pr_field (string): The name of the df column with precipitation values.
#     - hydro_year_field (string): The name of the df column with hydrological years values.

#     Returns
#     -------
#     - consecutive_values np.array: index of time of consecutive values defining the ordinary events.


#     Examples
#     --------
#     """
#     ll_short=[True if ev[-1]-ev[0] >= pd.Timedelta(minutes=min_duration) else False for ev in list_ordinary]
#     filtered_list = [x for x, keep in zip(list_ordinary, ll_short) if keep]
#     list_year=pd.DataFrame([filtered_list[_][0].year for _ in range(len(filtered_list))],columns=['year'])

#     return ll_short

