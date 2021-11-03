'''
Functions created for the calculation of a matrix which values are 
an estimate of the error of the implied volatility grid coming from 
the MC method used to compute vanilla options.
'''

import numpy as np
import pandas as pd
from scipy import stats

def MC_err_price(S, K, S_0, n, expiries_index):
    C_err = np.zeros_like(K)        # Matrix of MC prices
    for i in range(K.shape[0]):
        for j in range(C_err.shape[1]):
            if np.isnan(C_err[i,j]):
                C_err[i,j] = np.nan
            elif(K[i,j] >= S_0):
                 C_err[i,j] = np.maximum(S[expiries_index[i]+1] - K[i,j], 0).std()
            else:
                C_err[i,j] = np.maximum(K[i,j] - S[expiries_index[i]+1], 0).std()
    C_err = C_err/np.sqrt(n)
    return C_err

def BSM_vega(vol, S_0, K, T):
    d1 = np.divide(np.log(np.divide(S_0,K)) + 0.5*vol*vol*T, vol*np.sqrt(T))
    vega = S_0*np.sqrt(T)*stats.norm.pdf(d1, loc=0, scale=1)
    return vega

def vega_error_IV(C_err, K, S_0, IV, expiries):
    IV_err = np.zeros_like(C_err)
    for i in range(C_err.shape[0]):
        for j in range(C_err.shape[1]):
            if np.isnan(C_err[i,j]):
                IV_err[i,j] = np.nan
            else:
                IV_err[i,j] = C_err[i,j]/BSM_vega(IV[i,j], S_0, K[i,j], expiries[i])
    return IV_err

def df_IV_err_creation(IV_err, strikes, expiries_nan, tenor_len):
    if IV_err.shape[0] < tenor_len:
        to_add = np.ones((tenor_len - IV_err.shape[0], IV_err.shape[1]))*np.nan
        IV_err_df = np.vstack([IV_err, to_add])
    else:
        IV_err_df = IV_err
    df_err_IV = pd.DataFrame(data = IV_err_df.T, index = strikes, columns = expiries_nan)
    df_err_IV.index.name = 'ImpVolError'
    return df_err_IV