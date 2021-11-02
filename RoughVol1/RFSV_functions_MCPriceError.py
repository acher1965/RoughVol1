'''
Functions created for the calculation of a matrix which values are 
the error of prices previously computed in RFSV_main with the MC method.
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

def df_C_err_creation(C_err, strikes, expiries, tenor_len):
    if C_err.shape[0] < tenor_len:
        to_add = np.ones((tenor_len - C_err.shape[0], C_err.shape[1]))*np.nan
        C_err_df = np.vstack([C_err, to_add])
    else:
        C_err_df = C_err
    col_names = np.hstack([expiries, [np.nan] * (C_err_df.shape[0] - C_err.shape[0])])
    df_C_err = pd.DataFrame(data = C_err_df.T, index = strikes, columns = col_names)
    df_C_err.index.name = 'MCPriceError'
    return df_C_err

def writing_results(writer, df_term, df_strikes, df_IV, df_skew_smile, df_IV_approx, df_skew_smile_approx, df_C_err, sheet_name):
    startrow = 0
    df_term.to_excel(writer, header = None, index = True, na_rep = '#N/A', startcol = 3,
                    startrow = startrow, sheet_name = sheet_name)
    startrow = startrow + df_term.shape[0]
    df_strikes.to_excel(writer, header = True, index = True, na_rep = '#N/A', startcol = 3,
                        startrow = startrow, sheet_name = sheet_name)
    startrow = startrow + df_strikes.shape[0] + 1
    df_IV.to_excel(writer, header = True, index = True, na_rep = '#N/A', startcol = 3, 
                  startrow = startrow, sheet_name = sheet_name)
    startrow = startrow + df_IV.shape[0] + 1
    df_skew_smile.to_excel(writer, header = True, index = True, na_rep = '#N/A', startcol = 3, 
                  startrow = startrow, sheet_name = sheet_name)
    startrow = startrow + df_skew_smile.shape[0] + 1
    df_IV_approx.to_excel(writer, header = True, index = True, na_rep = '#N/A', startcol = 3, 
                  startrow = startrow, sheet_name = sheet_name)  
    startrow = startrow + df_IV_approx.shape[0] + 1    
    df_skew_smile_approx.to_excel(writer, header = True, index = True, na_rep = '#N/A', startcol = 3, 
                  startrow = startrow, sheet_name = sheet_name)
    startrow = startrow + df_skew_smile_approx.shape[0] + 1
    df_C_err.to_excel(writer, header = True, index = True, na_rep = '#N/A', startcol = 3,
                        startrow = startrow, sheet_name = sheet_name)

'''
Functions created for the calculation of a matrix which values are 
an estimate of the error of the implied volatility grid coming from 
the MC method used to compute vanilla options.
'''

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

# Lines to be added or replaced in the code:
# to add in RFSV_main.py and in RFSV_functions.py
# from RFSV_functions_MCErrorPrice import MC_err_price, df_C_err_creation, writing_results

# lines to add inside function calculate_request(hc, row_inputs) in RFSV_function.py
# RFSV_BF_Results = collections.namedtuple('RFSV_BF_Results',['request_id', 'df_C_err', 'df_diagn', 'df_info', 'df_int_var_std', 'df_model', 'df_sim', 'df_skew_smile', 'df_skew_smile_approx', 'df_strikes', 'df_term', 'df_IV', 'df_IV_approx'])
# C_err = MC_err_price(S, K, r.S_0, expiries_index)
# df_C_err = df_C_err_creation(C_err, hc.strikes, expiries, hc.tenor_len)
# return RFSV_BF_Results(r.request_id, df_C_err, df_diagn, df_info, df_int_var_std, df_model, df_sim, df_skew_smile, df_skew_smile_approx, df_strikes, df_term, df_IV, df_IV_approx)

# line to change in write_results_to_sheet(writer, results) in RFSV_helpers.py
# writing_results(writer, r.df_term, r.df_strikes, r.df_IV, r.df_skew_smile, r.df_IV_approx, r.df_skew_smile_approx, r.df_C_err, sheet_name)

# line to chang in writing_results(...) in RFSV_helpers.py
# def writing_results(writer, df_term, df_strikes, df_IV, df_skew_smile, df_IV_approx, df_skew_smile_approx, df_C_err, sheet_name):

# lines to add in writing_results(...) in RFSV_helpers.py
#    startrow = startrow + df_skew_smile_approx.shape[0] + 1
#    df_C_err.to_excel(writer, header = True, index = True, na_rep = '#N/A', startcol = 3,
#                        startrow = startrow, sheet_name = sheet_name)