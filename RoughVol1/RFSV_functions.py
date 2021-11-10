# -*- coding: utf-8 -*-
'''Functions to run simulations for RFSV bruteforce, rBergomi model.

Most functions written by Augusto Marcon.
AC refactored, adding:
*hard_coded_params  function encapsulating previous code
*calculate_request  function with the top level math logic

@author: amarc
'''

import time
import collections # why this not working??  from collections import namedtuple
import numpy as np
from scipy import stats, special, optimize, integrate
from RFSV_functions_Error import df_IV_err_calc, fit_cost_calculation, cost_row_df_creation

#Hard Coded Parameters
HardCodedParameters = collections.namedtuple('HardCodedParameters',['N_treshold', 'code_version', 'dK_skew', 'near_atm_strikes', 'strikes', 'tenor_len', 'dt_short', 'dt_inf', 'alpha'])
def hard_coded_params():
    ''' Hard-coded parameters'''
    N_treshold = 10
    dt_short, dt_inf, alpha = 1 / 25 / 365, 6 / 365, 0.5        # parameters for time steps
    strikes = np.arange(-2.5,2.5 + 0.5,0.5)       # in stdev terms
    near_atm_strikes = np.linspace(-0.5,0.5,11)[np.isin(np.linspace(-0.5,0.5,11), strikes, invert = True)]
    strikes = np.sort(np.hstack([strikes, near_atm_strikes]))
    
    dK_skew = 0.01       # also for smile
    tenor_len = 10       # default length of tenor
    code_version = 'RFSV_bruteforce_1_0_0'
    return HardCodedParameters(N_treshold, code_version, dK_skew, near_atm_strikes, strikes, tenor_len, dt_short, dt_inf, alpha)

#Calculation of one request row
RFSV_BF_Results = collections.namedtuple('RFSV_BF_Results',['request_id', 'full_output_flag', 'df_diagn', 'df_info', 'df_int_var_std', 'df_model', 'df_sim', 'df_skew_smile', 'df_skew_smile_approx', 'df_strikes', 'df_term', 'df_IV', 'df_IV_approx', 'df_err_IV', 'df_cost_row'])
def calculate_request(hc, row_inputs, volgrid, weigthsgrid):
    ''' Run the simulations RFSV bruteforce-rBergomi for one request'''
    
    start_time = time.time()
    
    r = row_inputs

    [expiries, forward_input, xi_input] = arrays_control(r.expiries_nan, r.forward_input_nan, r.xi_input_nan)
    [expiries_nan, forward_input_nan, xi_input_nan] = arrays_for_output(expiries, forward_input, xi_input, hc.tenor_len)
    
    t, mv = t_from_dt_e(hc.dt_short, hc.dt_inf, hc.alpha, expiries)
    T = 1
    expiries_index = np.where(np.isin(t, expiries))[0]  
    xi, xi_values = interpolating_xi(xi_input, expiries, mv)
    
    Sigma = Sigma_calculation(t,r.H,r.rho)
    L = np.linalg.cholesky(Sigma)
    
    logS, S, logv = spot_calculation(t, r.S_0, xi, r.eta, r.H, L, r.n, r.random_seed)
    
    K, C, K_skew, C_skew, warn = MC_strike_and_price(S, r.S_0 , expiries, t, hc.strikes, hc.dK_skew, xi, r.n, hc.N_treshold)
    IV, IV_skew = imp_vol_calculation(K, C, K_skew, C_skew, r.S_0, expiries)
    skew, smile = skew_smile_calculation(IV_skew, hc.dK_skew)
    sign_bound, Sigma0_d, Sigma0_dd, a_0 = Sigma_Taylor_coefficients(r.H, r.rho, r.eta, xi, expiries)
    IV_approx, skew_approx, smile_approx, IV_skew_approx = imp_vol_approx(K, K_skew, r.S_0, expiries, xi, r.H, r.rho, r.eta, hc.tenor_len)
    fit_cost_arr = fit_cost_calculation(IV, volgrid, weigthsgrid, hc.tenor_len)
    cost_row = np.vstack([r.request_id, r.underlying, r.as_of, r.H, r.eta, r.rho, fit_cost_arr])

    df_info = df_info_creation(r.request_id, r.underlying, r.as_of, r.S_0)
    df_sim = df_sim_creation(r.n, hc.N_treshold, r.random_seed, t.size)
    df_model = df_model_creation(r.H, r.eta, r.rho)
    end_time = time.time()
    total_time = end_time - start_time
    df_diagn = df_diagn_creation(hc.dK_skew, hc.dt_short, hc.dt_inf, hc.alpha, total_time, hc.code_version, Sigma0_d, Sigma0_dd, a_0)
    if (r.full_diagn_flag == True):
        df_int_var_std = df_integ_var_std_creation(logS, logv, xi_input, xi_values, t, expiries, xi, r.H, r.eta, T)
    else:
        df_int_var_std = pd.DataFrame()
    
    df_term = df_term_creation(expiries_nan, forward_input_nan, xi_input_nan)
    df_strikes = df_strikes_creation(K, hc.strikes, r.S_0, hc.tenor_len)
    df_err_IV = df_IV_err_calc(S, K, r.S_0, r.n, expiries_index, IV, expiries, hc.strikes, expiries_nan, hc.tenor_len)
    IV_df, df_IV = df_IV_creation(IV, hc.strikes, hc.tenor_len)
    df_skew_smile = df_skew_smile_creation(skew, smile, IV_skew, IV_df, hc.strikes, hc.dK_skew, expiries_nan, hc.tenor_len)
    IV_approx_df, df_IV_approx = df_IV_creation(IV_approx, hc.strikes, hc.tenor_len, 'Approx')
    df_skew_smile_approx = df_skew_smile_creation(skew_approx, smile_approx, IV_skew_approx, IV_approx_df, hc.strikes, hc.dK_skew, expiries_nan, hc.tenor_len)
    df_cost_row = cost_row_df_creation(cost_row, hc.tenor_len)
    return RFSV_BF_Results(r.request_id, r.full_output_flag, df_diagn, df_info, df_int_var_std, df_model, df_sim, df_skew_smile, df_skew_smile_approx, df_strikes, df_term, df_IV, df_IV_approx, df_err_IV, df_cost_row)

# For other parameters calculation
def parameter_with_H(H):
    gamma = 0.5 - H
    D_H = np.sqrt(2 * H) / (H + 0.5)
    return [gamma, D_H]

def t_from_dt_e(dt_short, dt_inf, alpha, e):
    dt = dt_short * np.exp(-alpha * e) + dt_inf * (1 - np.exp(-alpha * e))
    mv = np.zeros(e.size)
    
    t = np.arange(dt[0], e[0], dt[0])
    if np.isin(e[0],t, invert=True):
        t = np.hstack([t,e[0]])
    mv[0] = t.size
    for i in range(1,e.size):
        t = np.hstack([t, np.arange(e[i - 1] + dt[i], e[i], dt[i])])
        if np.isin(e[i],t, invert=True):
            t = np.hstack([t,e[i]])
        mv[i] = t.size
    mv = np.diff(mv, prepend = 0).astype(int)
    return t, mv

# Functions for computing Covariance matrix Sigma
def cov_W(j,t,gamma):
    coeff = t[j] ** (1 - 2 * gamma) * (1 - 2 * gamma) / (1 - gamma)
    return coeff * (t[j] / t[j + 1:]) ** gamma * special.hyp2f1(gamma, 1, 2 - gamma,t[j] / t[j + 1:])

def cov_W_Z(j,t,H,rho,D_H):
    return rho * D_H * (t[j] ** (H + 0.5)) * np.ones(t.size - j)

def cov_Z_W(j,t,H,rho,D_H):
    return rho * D_H * (t[j + 1:] ** (H + 0.5) - (t[j + 1:] - t[j]) ** (H + 0.5))

def cov_Z(j,t):
    return t[j] * np.ones(t[j + 1:].size)

# Vanilla Price Calculation for matrices of Spot (and K an array)
# Call for K > S_0, put for K < S_0
def MC_price(S, K, S_0):
    m = S.shape[0]
    l = K.size
    if S.ndim > 1:
        C = np.zeros((m,l))
        for i in range(l):
            if K[i] >= S_0 :
                C[:,i] = np.maximum(S - K[i], 0).mean(axis = 1)
            else: 
                C[:,i] = np.maximum(K[i] - S, 0).mean(axis = 1)
    else:
        C = np.zeros((l))
        for i in range(l):
            if K[i] >= S_0 :
                C[i] = np.maximum(S - K[i], 0).mean()
            else: 
                C[i] = np.maximum(K[i] - S, 0).mean()
    return C

# Functions for Implied Volatility calculation
# B&S call price function
def BSM_call_price(vol, S_0, K, T):
    d1 = np.divide(np.log(np.divide(S_0,K)) + 0.5 * vol * vol * T, vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    p = S_0 * stats.norm.cdf(d1, loc=0, scale=1) - K * stats.norm.cdf(d2, loc=0, scale=1)
    return p

# B&S put price function
def BSM_put_price(vol, S_0, K, T):
    d1 = np.divide(np.log(np.divide(S_0,K)) + 0.5 * vol * vol * T, vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    p = K * stats.norm.cdf(-d2, loc=0, scale=1) - S_0 * stats.norm.cdf(-d1, loc=0, scale=1)
    return p

# B&S implied vol error functions
def error_imp_vol(vol, S_0, K, T, price):
    if K >= S_0:
        return price - BSM_call_price(vol, S_0, K, T)
    else:
        return price - BSM_put_price(vol, S_0, K, T)

# B&S implied vol functions
def BSM_imp_vol(S_0, K, T, price):
    if error_imp_vol(0.0001, S_0, K, T, price) < 0 :
        return np.nan
    else:
        o = optimize.root_scalar(error_imp_vol, args= (S_0,K,T,price), bracket= [0.0001, 5.], x0= 0.2, x1= 0.4)
    return o.root

import pandas as pd
import platform
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def arrays_control(expiries_nan, forward_input_nan, xi_input_nan):
    a = np.isfinite(expiries_nan)
    b = np.isfinite(forward_input_nan)
    c = np.isfinite(xi_input_nan)
    index_intersection = np.logical_and(a,b,c)
    expiries = expiries_nan[index_intersection]
    forward_input = forward_input_nan[index_intersection]
    xi_input = xi_input_nan[index_intersection]
    return [expiries, forward_input, xi_input]

def arrays_for_output(expiries, forward_input, xi_input, tenor_len):
    if expiries.size < tenor_len:
        to_add = np.ones(tenor_len - expiries.size) * np.nan
        expiries_nan = np.hstack([expiries, to_add])
        forward_input_nan = np.hstack([forward_input, to_add])
        xi_input_nan = np.hstack([xi_input, to_add])
    else:
        expiries_nan = expiries
        forward_input_nan = forward_input
        xi_input_nan = xi_input
    return [expiries_nan, forward_input_nan, xi_input_nan]

def interpolating_xi(xi_input, expiries, mv):
    xi_values = np.zeros(expiries.size)
    xi_values = np.diff(xi_input * expiries, prepend = 0) / np.diff(expiries, prepend = 0)
    xi = np.zeros(mv.sum() + 1)
    xi[0:mv[0] + 1] = xi_values[0]
    for i in range(1,mv.size):
        xi[np.cumsum(mv)[i - 1]:np.cumsum(mv)[i]] = xi_values[i]
    xi[-1] = xi[-2]
    return xi, xi_values

def xi_function(x, t, expiries_index, xi_values):
    if (x < -0.02 or x > t[expiries_index[-1]] + 0.02):
        return 0
    if (-0.02 <= x < 0):
        mom_ang = xi_values[0] / 0.02
        return mom_ang * (x + 0.02)            
    if (0 <= x < t[expiries_index[0] - 1]):
        return xi_values[0]  
    
    for i in range(expiries_index.size - 1):
        if (t[expiries_index[i] - 1] <= x < t[expiries_index[i] + 1]):
            mom_ang = (xi_values[i + 1] - xi_values[i]) / (t[expiries_index[i] + 1] - t[expiries_index[i] - 1])
            return xi_values[i] + mom_ang * (x - t[expiries_index[i] - 1])
        if (t[expiries_index[i] + 1] <= x < t[expiries_index[i + 1] - 1]):   
            return xi_values[i + 1]
        
    if (t[expiries_index[-1] - 1] <= x <= t[expiries_index[-1]]):
        return xi_values[-1]
    if (t[expiries_index[-1]] <= x <= t[expiries_index[-1]] + 0.02):
        mom_ang = (-xi_values[-1]) / (0.02)
        return xi_values[-1] + mom_ang * (x - t[expiries_index[-1]])
    return -1

def std_integrand(y, x, H, eta, t, expiries_index, xi_values):
    gamma = 0.5 - H
    coeff = xi_function(y, t, expiries_index, xi_values) * xi_function(x, t, expiries_index, xi_values)
    coeff2 = (1 - 2 * gamma) / (1 - gamma)
    return coeff * np.exp(eta * eta * coeff2 * y ** (1 - 2 * gamma) * (y / x) ** gamma * special.hyp2f1(gamma, 1, 2 - gamma, y / x))

def Sigma_calculation(t,H,rho):
    gamma, D_H = parameter_with_H(H)
    m = t.size
    Sigma = np.zeros((2 * m,2 * m))
    for i in range(m):
        Sigma[2 * i,2 * (i + 1)::2] = cov_W(i,t,gamma)
        Sigma[2 * i,2 * i + 1::2] = cov_W_Z(i,t,H,rho,D_H)
        Sigma[2 * i + 1,2 * i + 2::2] = cov_Z_W(i,t,H,rho,D_H)
        Sigma[2 * i + 1,2 * i + 3::2] = cov_Z(i,t)
    d = np.zeros(2 * m)
    d[0::2] = t ** (2 * H)
    d[1::2] = t
    Sigma = Sigma + np.transpose(Sigma) + np.diag(d)
    return Sigma

def spot_calculation(t, S_0, xi, eta, H, L, n, random_seed):
    rng = np.random.default_rng(seed = random_seed)      # here seed can be changed
    ## Creating matrices for the algorithm
    m = t.size
    B = np.zeros((2 * m,n))
    X = np.zeros((2,2 * n))
    Ztmp = np.zeros(2 * n)
    dZ = np.zeros(2 * n)
    dt = np.diff(t, prepend = 0)
    logS = np.zeros((m + 1,2 * n))
    logS[0] = np.log(S_0)
    logv = np.zeros((m + 1,2 * n))
    logv[0] = np.log(xi[0])
    logxi = np.log(xi)
    ## Algorithm core
    for j in range(0,m):
        B[j * 2:j * 2 + 2] = rng.normal(loc = 0.0, scale = 1.0, size=(2,n))
        X[:, :n] = np.matmul(L[j * 2:j * 2 + 2],B)
        X[:, n:] = - X[:, :n]
        if (j != 0): 
            dZ = X[1] - Ztmp
        else:
            dZ = X[1]
        tmp = np.exp(0.5 * logv[j])
        logS[j + 1] = logS[j] + tmp * dZ - 0.5 * tmp * tmp * dt[j]
        logv[j + 1] = logxi[j] + eta * X[0] - 0.5 * (eta * eta) * t[j] ** (2 * H)
        Ztmp = X[1].copy()
    # Spot matrix from the log matrix
    S = np.zeros((m + 1,n))
    S = np.exp(logS)
    return logS, S, logv

def MC_strike_and_price(S, S_0 , expiries, t, strikes, dK_skew, xi, n, N_treshold):
    expiries_index = np.where(np.isin(t,expiries))[0]
    dt = np.diff(t, prepend = 0)
    cum_vol = np.zeros(t.size)
    cum_vol = np.sqrt(np.cumsum(xi[:-1] * dt))        # realized volatility
    K = np.zeros((expiries.size, strikes.size))     # matrix of strikes
    # Values to control the use of robust strikes only
    quant_c,quant_p = np.zeros(expiries.size), np.zeros(expiries.size)
    warn = False
    for i in range(expiries.size):
        quant_c[i] = np.quantile(S[expiries_index[i] + 1], 1 - N_treshold / n)
        quant_p[i] = np.quantile(S[expiries_index[i] + 1], N_treshold / n)
    for i in range(expiries.size):
        K[i] = S_0 * np.maximum(np.exp(strikes * cum_vol[expiries_index[i]]),0.01)
        if (np.any(K[i] <= quant_p[i]) or np.any(K[i] >= quant_c[i])):
            b = np.logical_or(K[i] > quant_c[i], K[i] < quant_p[i])
            K[i,b] = np.nan
            warn = True
    # Matrix of MC prices: put for K < 100, call for K >= 100
    C = np.zeros_like(K)        # Matrix of MC prices
    for i in range(expiries.size):
        tmp = np.isfinite(K[i])
        C[i,tmp] = MC_price(S[expiries_index[i] + 1], K[i,tmp], S_0)
        tmp = np.isnan(K[i])
        C[i,np.isnan(K[i])] = np.nan

    K_skew = np.zeros((expiries.size, 3))
    K_skew[:,0] = S_0 * (1 - dK_skew)
    K_skew[:,1] = S_0
    K_skew[:,2] = S_0 * (1 + dK_skew)
    for i in range(expiries.size):
        if (np.any(K_skew[i] <= quant_p[i]) or np.any(K_skew[i] >= quant_c[i])):
            b = np.logical_or(K_skew[i] > quant_c[i], K_skew[i] < quant_p[i])
            K_skew[i,b] = np.nan
    C_skew = np.zeros_like(K_skew)
    for i in range(expiries.size):
        tmp = np.isfinite(K_skew[i])
        C_skew[i,tmp] = MC_price(S[expiries_index[i] + 1], K_skew[i,tmp], S_0)
        tmp = np.isnan(K_skew[i])
        C_skew[i,np.isnan(K_skew[i])] = np.nan 
    return K, C, K_skew, C_skew, warn

def imp_vol_calculation(K, C, K_skew, C_skew, S_0, expiries):
    IV = np.zeros_like(C)
    for i in range(IV.shape[0]):
        for j in range(IV.shape[1]):
            if np.isnan(K[i,j]):
                IV[i,j] = np.nan
            else:
                IV[i,j] = BSM_imp_vol(S_0, K[i,j], expiries[i], C[i,j])
    IV_skew = np.zeros_like(C_skew)
    for i in range(IV_skew.shape[0]):
        for j in range(IV_skew.shape[1]):
                if np.isnan(K_skew[i,j]):
                    IV_skew[i,j] = np.nan
                else:
                    IV_skew[i,j] = BSM_imp_vol(S_0, K_skew[i,j], expiries[i], C_skew[i,j])
    return IV, IV_skew

def skew_smile_calculation(IV_skew, dK_skew):
    skew = np.zeros(IV_skew.shape[0])
    skew = (IV_skew[:,2] - IV_skew[:,0]) / (2 * dK_skew)
    smile = np.zeros(IV_skew.shape[0])
    smile = (IV_skew[:,2] - 2 * IV_skew[:,1] + IV_skew[:,0]) / (dK_skew ** 2)
    return skew, smile

def Sigma_Taylor_coefficients(H, rho, eta, xi, expiries):
    # Sigma coefficients from Friz and Pigato paper
    K1_1 = np.sqrt(2 * H) / ((H + 0.5) * (H + 1.5))      # <K1,1>
    K12_1 = H / ((H + 1) * (H + 0.5) * (H + 0.5))        # <(K1)^2 1,1>
    K1_Kbar1 = special.beta(H + 1.5,H + 1.5) * (2 * H) / ((H + 0.5) * (H + 0.5))    # <K1,K_bar 1>
    K21_1 = 1 / (2 * H + 1)        # <K^2 1,1>
    sign_bound = K12_1 / (6 * K1_1 * K1_1 - 2 * K12_1 - 2 * K1_Kbar1)
    sigma0 = np.sqrt(xi[0])
    sigma0_d = eta * sigma0 / 2
    sigma0_dd = eta * eta * sigma0 / 4
    sigma0_dot = - sigma0 * eta * eta / 4
    Sigma0_d = rho * sigma0_d * K1_1 / sigma0
    Sigma0_dd_term = 2 * rho * rho * (-3 * K1_1 * K1_1 + 0.5 * K12_1 + K1_Kbar1) + K12_1
    Sigma0_dd = sigma0_d * sigma0_d / (sigma0 ** 3) * Sigma0_dd_term + sigma0_dd * rho * rho * K12_1 / (sigma0 * sigma0)
    D_K_rho = K21_1 - K12_1 + rho * rho * (3 * K1_1 * K1_1 - K12_1 - 2 * K1_Kbar1)
    D_bar_K_rho = K21_1 - rho * rho * K12_1
    a_0 = sigma0_d * sigma0_d * D_K_rho + sigma0 * sigma0_dd * D_bar_K_rho + sigma0 * sigma0_dot / (H + 0.5)
    if (H == 0.5):
        a_0 = a_0 + rho * sigma0_d * sigma0 * sigma0 * K1_1
    return sign_bound, Sigma0_d, Sigma0_dd, a_0

def imp_vol_approx(K, K_skew, S_0, expiries, xi, H, rho, eta, tenor_len):
    sign_bound, Sigma0_d, Sigma0_dd, a_0 = Sigma_Taylor_coefficients(H, rho, eta, xi, expiries)
    Sigma0 = np.sqrt(xi[0])
    logK = np.log(K / S_0)
    x = logK * (expiries[:,np.newaxis] ** (H - 0.5))
    IV_approx = Sigma0 + Sigma0_d * x + 0.5 * Sigma0_dd * x * x + 0.5 * a_0 / Sigma0 * expiries[:,np.newaxis] ** (2 * H)
    skew_approx = Sigma0_d * expiries ** (H - 0.5)
    smile_approx = Sigma0_dd * expiries ** (2 * H - 1)
    logK_skew = np.log(K_skew / S_0)
    x_skew = logK_skew * (expiries[:,np.newaxis] ** (H - 0.5))
    IV_skew_approx = Sigma0 + Sigma0_d * x_skew + 0.5 * Sigma0_dd * x_skew * x_skew + 0.5 * a_0 / Sigma0 * expiries[:,np.newaxis] ** (2 * H)
    return IV_approx, skew_approx, smile_approx, IV_skew_approx

def df_info_creation(request_id, underlying, as_of, S_0):
    info_dic = {}
    info_dic['RequestId'] = [request_id, '']
    info_dic['Underlying'] = [underlying, '']
    info_dic['AsOf'] = [as_of, '']
    info_dic['Spot'] = [S_0, '']
    df_info = pd.DataFrame(data = info_dic, index = ['','']).T
    return df_info

def df_sim_creation(n, N_treshold, random_seed, m):
    sim_dic = {}
    sim_dic['Trials'] = [n, '(input)']
    sim_dic['Seed'] = [random_seed, '(input)']
    sim_dic['N_path_threshold'] = [N_treshold, '']
    sim_dic['TimeSteps'] = [m, '']
    df_sim = pd.DataFrame(data = sim_dic, index = ['', '']).T
    df_sim.index.name = 'ControlParameters'
    return df_sim

def df_model_creation(H, eta, rho):
    model_dic = {}
    model_dic['Model'] = ['roughBergomi', '']
    model_dic['H'], model_dic['Eta'], model_dic['Rho'] = [H, ''], [eta, ''], [rho, '']
    df_model = pd.DataFrame(data = model_dic, index = ['','']).T
    df_model.index.name = 'ModelParameters'
    return df_model

def df_diagn_creation(dK_skew, dt_short, dt_inf, alpha, total_time, code_version, Sigma0_d, Sigma0_dd, a_0):
    diagn_dic = {}
    diagn_dic['dK_skew'] = [dK_skew, '']
    diagn_dic['dt_short'] = [dt_short, '']
    diagn_dic['dt_inf'] = [dt_inf, '']
    diagn_dic['alpha'] = [alpha, '']
    diagn_dic['Time'] = [total_time, '(sec)']
    diagn_dic['RunningDate'] = [datetime.today().strftime('%d-%m-%Y'), '']   #  strftime('%x'), see https://strftime.org/
    diagn_dic['Code_version'] = [code_version, '']
    diagn_dic['Python_version'] = [platform.python_version(), '']
    diagn_dic['OS'] = [platform.platform(), '']
    diagn_dic['Processor'] = [platform.processor(), '']
    diagn_dic['Sigma0_1'] = [Sigma0_d, '']
    diagn_dic['Sigma0_2'] = [Sigma0_dd, '']
    diagn_dic['a_0'] = [a_0, '']
    df_diagn = pd.DataFrame(data = diagn_dic, index = ['','']).T
    df_diagn.index.name = 'SimulationDiagnostics'
    return df_diagn

def df_integ_var_std_creation(logS, logv, xi_input, xi_values, t, expiries, xi, H, eta, T):
    expiries_index = np.where(np.isin(t,expiries))[0]
    y = np.where(expiries == T)[0][0]
    dt = np.diff(t, prepend = 0)
    # std dev of integrated var using the spot process only
    integ_var_logS = np.zeros(logS.shape[1])
    for i in range(expiries_index[y] + 1):
        integ_var_logS = integ_var_logS + (logS[i + 1] - logS[i]) ** 2
    integ_var_logS_m = integ_var_logS.mean()
    integ_var_logS_std = integ_var_logS.std()
    # std dev of integrated var using the variance process only
    v = np.exp(logv[:expiries_index[y] + 1])
    integ_var_v = (v * dt[:expiries_index[y] + 1, np.newaxis]).sum(axis = 0)
    integ_var_v_m = integ_var_v.mean()
    integ_var_v_std = integ_var_v.std()
    # numerical integration of std integrated var for non flat xi
    expiries_index_T = np.where(np.isin(t, expiries[expiries <= T]))[0]
    int_subranges = np.hstack([0, t[expiries_index_T[:-1] - 1], t[expiries_index_T[:-1] + 1], t[expiries_index_T[-1]]])
    int_subranges = np.sort(int_subranges)
    tmp = 0
    for i in range(int_subranges.size - 1):
        tmp = tmp + integrate.dblquad(std_integrand, int_subranges[i], int_subranges[i + 1],
                                      lambda x :0.0, lambda x :x,
                                      args=(H, eta, t, expiries_index, xi_values))[0]
    second_mom_integ_var = 2 * tmp
    integ_var_num_m = np.dot(xi[:expiries_index[y] + 1], dt[:expiries_index[y] + 1])
    integ_var_num_std = np.sqrt(second_mom_integ_var - integ_var_num_m ** 2)
    
    arr_std = np.array([integ_var_logS_m, integ_var_logS_std, integ_var_v_m, integ_var_v_std,
                        integ_var_num_m, integ_var_num_std])
    # creating a df for the input
    std_integ_var = {}
    std_integ_var['integ_var_logS_m'] = [arr_std[0], '']
    std_integ_var['integ_var_logS_std'] = [arr_std[1], '']
    std_integ_var['integ_var_v_m'] = [arr_std[2], '']
    std_integ_var['integ_var_v_std'] = [arr_std[3], '']
    std_integ_var['integ_var_num_m'] = [arr_std[4], '']
    std_integ_var['integ_var_num_std'] = [arr_std[5], '']
    
    std_integ_var_df = pd.DataFrame(std_integ_var, index = ['','']).T
    std_integ_var_df.index.name = 'Std_Integrat_Var'
    return std_integ_var_df

def df_term_creation(expiries_nan, forward_input_nan, xi_input_nan):
    tenor_df = expiries_nan[:,np.newaxis]
    forw_df = forward_input_nan[:,np.newaxis]
    xi_df = np.sqrt(xi_input_nan[:,np.newaxis])
    col_names = ['Tenor', 'Forward', 'VarSwap']
    df_term = pd.DataFrame(data = np.concatenate([tenor_df, forw_df, xi_df], axis = 1), 
                                 index = None, columns = col_names).T
    return df_term

def df_strikes_creation(K, strikes, S_0, tenor_len):
    K_perc = K.copy() / S_0
    if K_perc.shape[0] < tenor_len:
        to_add = np.ones((tenor_len - K_perc.shape[0], K_perc.shape[1])) * np.nan
        K_df = np.vstack([K_perc, to_add])
    else:
        K_df = K_perc
    col_names = ['FwdMness'] * K_perc.shape[0] + [np.nan] * (K_df.shape[0] - K_perc.shape[0])
    df_strikes = pd.DataFrame(data = K_df.T, index = strikes, columns = col_names)
    df_strikes.index.name = 'Nstdev'
    return df_strikes

def df_IV_creation(IV, strikes, tenor_len, c=''):
    if IV.shape[0] < tenor_len:
        to_add = np.ones((tenor_len - IV.shape[0], IV.shape[1])) * np.nan
        IV_df = np.vstack([IV, to_add])
    else:
        IV_df = IV
    col_names = ['Impvol'] * IV.shape[0] + [np.nan] * (IV_df.shape[0] - IV.shape[0])
    df_IV = pd.DataFrame(data = IV_df.T, index = strikes, columns = col_names)
    df_IV.index.name = 'Nstdev'
    if (c == 'Approx'): df_IV.index.name = 'Approximation'
    return IV_df, df_IV

def df_skew_smile_creation(skew, smile, IV_skew, IV_df, strikes, dK_skew, expiries_nan, tenor_len):
    atm_index = np.where(strikes == 0)[0][0]
    atm_vol_df = IV_df[:, atm_index][:,np.newaxis]
    
    if IV_skew.shape[0] < tenor_len:
        to_add = np.ones((tenor_len - IV_skew.shape[0], IV_skew.shape[1])) * np.nan
        IV_skew_df = np.vstack([IV_skew, to_add])
        to_add = np.ones(tenor_len - skew.size) * np.nan
        skew_df = np.hstack([skew, to_add])
        smile_df = np.hstack([smile, to_add])
    else:
        IV_skew_df = IV_skew
        skew_df = skew
        smile_df = smile
    skew_df_df = skew_df[:,np.newaxis]
    smile_df_df = smile_df[:,np.newaxis]
    
    vol1_skew_df = IV_skew_df[:,0][:,np.newaxis]
    vol2_skew_df = IV_skew_df[:,2][:,np.newaxis]

    col_names = ['ATM_vol', 'ATM_skew', 'ATM_smile']
    col_names = col_names + [str((1 - dK_skew) * 100) + '%Vol', 'ATM_vol', str((1 + dK_skew) * 100) + '%Vol']
    df_skew_smile = pd.DataFrame(data = np.concatenate([atm_vol_df, skew_df_df, smile_df_df, vol1_skew_df, atm_vol_df, vol2_skew_df],
                                                       axis = 1), index = expiries_nan, columns = col_names).T
    df_skew_smile.index.name = 'Tenor'
    return df_skew_smile

def vol_surface_plot(IV, K, expiries, output_path, request_id, warn):
    # Scatter plot needs 1D arrays
    IV_plot = IV.copy()
    t_plot = expiries.copy()
    K_plot = K.copy()

    IV_s = IV_plot[np.isfinite(IV_plot)]
    K_s = K_plot[np.isfinite(K_plot)]
    t_s = np.zeros(IV_s.size)
    index_sum = 0
    for i in range(IV_plot.shape[0]):
        tmp = np.isfinite(IV_plot[i]).sum()
        t_s[index_sum : index_sum + tmp] = np.ones(tmp) * t_plot[i]
        index_sum = index_sum + tmp
    
    plotting_error = ''
    if warn == True:
        plotting_error = plotting_error + 'Warning!'
        plotting_error = plotting_error + '\nAt least one point of the Volatility surface is not sufficiently robust: '
        plotting_error = plotting_error + 'the surface may contain some errors or NaN.'
    
    save_plot_path = output_path.replace('.xlsx', '_Id' + str(request_id) + '.png') 
    fig = plt.figure(figsize=(12,8))
    ax = axes3d.Axes3D(fig)
    plt.ylabel('Strike K', fontsize = 14)
    plt.xlabel('Expiry', fontsize = 14)
    plt.title('Volatility surface', fontsize = 16)
    ax.view_init(elev = 40, azim = 30)
    ax.invert_yaxis()
    ax.scatter3D(t_s, K_s, IV_s, label = 'IV', c = 'red')
    ax.plot_trisurf(t_s, K_s, IV_s, cmap=cm.coolwarm)
    plt.savefig(save_plot_path)
    plt.close()      # remove this line if you want to see the plot
    return save_plot_path, plotting_error




