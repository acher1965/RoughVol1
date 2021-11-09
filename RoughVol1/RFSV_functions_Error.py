'''
To calculate a matrix with estimated errors of the implied volatility from MC for vanilla options

*df_IV_err_calc  returns a dataframe with the matrix

It calls other functions defined here
'''

import numpy as np
import pandas as pd
from scipy import stats

def df_IV_err_calc(S, K, S_0, n, expiries_index, IV, expiries, strikes, expiries_nan, tenor_len):
    '''Returns the dataframe with matrix of MC IV errors'''
    C_err = MC_err_price(S, K, S_0, n, expiries_index)
    IV_err = vega_error_IV(C_err, K, S_0, IV, expiries)
    return df_IV_err_creation(IV_err, strikes, expiries_nan, tenor_len)

def MC_err_price(S, K, S_0, n, expiries_index):
    C_err = np.zeros_like(K)
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

def fit_cost_calculation(IV, volgrid, tenor_len):
    '''Function to compute uniform fitting squared error'''
    fit_cost_matr = IV.T**2 - volgrid**2
    fit_cost_exp = np.nanmean(fit_cost_matr, axis = 0)
    fit_cost_global = np.nanmean(fit_cost_matr)
    if fit_cost_exp.size < tenor_len:
        to_add = np.ones(tenor_len - fit_cost_exp.size)*np.nan
        fit_cost_arr = np.hstack([fit_cost_exp, to_add])
    fit_cost_arr = np.hstack([fit_cost_exp, fit_cost_global])
    return fit_cost_arr[:,np.newaxis]

def cost_row_df_creation(cost_row, tenor_len):
    cost_exp_names = ['CostExpiry'+str(i) for i in range(1,tenor_len+1)] + ['GlobalCost']
    col_names = ['Id', 'Underlying', 'AsOf', 'H', 'eta', 'rho' ] + cost_exp_names
    df_cost_row = pd.DataFrame(cost_row.T, columns = col_names)
    return df_cost_row

def create_and_write_df_cost(output_path, results):
    df_cost = pd.DataFrame()
    for r in results:
        df_cost = pd.concat([df_cost, r.df_cost_row])
    output_path_cost = output_path.replace(".xlsx", "_cost.xlsx")
    with pd.ExcelWriter(output_path_cost, mode="w", engine="openpyxl") as writer:
        df_cost.to_excel(writer, header = True, index = False, sheet_name = "FittingCost")