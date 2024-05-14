import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import weibull_min

def wblinv(prob:float, shape:float, scale:float) -> np.ndarray[float]:
    return weibull_min.ppf(q=prob, c=shape, scale=scale)#.astype(np.ndarray)


def get_unlist(ll:list) -> list:
    ul=[]
    for sublist in ll:
        for file in sublist:
            ul.append(file)
    return ul


def plotting_position(N:float) -> np.ndarray[float]:
    
    P = np.arange(1,N+1)/(N+1)
    
    return P

def get_return_period() -> list[np.ndarray]:
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


if __name__=='__main__':
    # Example usage:
    scale = 7.8068
    shape = 0.8363 # scale parameter
    np.random.seed(1)
    prob  = np.random.random(size=10).reshape(5,2)  # probability

    result = wblinv(prob, shape, scale)
    print(result.shape)
    # print(f"{result=: >12.2f}")