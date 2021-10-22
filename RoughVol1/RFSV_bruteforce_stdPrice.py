# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats, optimize, special
import time
from openpyxl import load_workbook, Workbook
from os import path, chdir
import platform
from datetime import datetime

# importing all functions needed
from RFSV_functions_1_0_0 import *

## INSERT INPUT FILE PATH [AND REQUESTED IDs]
chdir('C:/Users/amarc/Desktop/EQF/Python')
input_path = 'RFSV_input.xlsx'
request_id_list = [1,2,3] # [i for i in range(1,2+1)]

# Hard-coded parameters
N_treshold = 10
dt_short, dt_inf, alpha = 1/25/365, 6/365, 0.5        # parameters for time steps

strikes = np.arange(-2.5,2.5+0.5,0.5)       # in stdev terms
near_atm_strikes = np.linspace(-0.5,0.5,11)[np.isin(np.linspace(-0.5,0.5,11), strikes, invert = True)]
strikes = np.sort(np.hstack([strikes, near_atm_strikes]))

dK_skew = 0.01       # also for smile
tenor_len = 10       # default length of tenor
code_version = 'RFSV_bruteforce_1_0_0'

control_request_id_list(request_id_list)

for request_counter in range(len(request_id_list)):
    start_time = time.time()
    if request_counter == 0:
        index = index_from_input(input_path)

    id_input = request_id_list[request_counter]
    [df_input, request_id, as_of, destination_file, S_0, underlying] = df_info_parameters(input_path, id_input, index)
    output_path = datetime.today().strftime('%Y%m%d') + '_' + datetime.now().time().strftime('%H%M%S') + '_' + underlying + '.xlsx'
    [H, eta, rho] = rough_vol_parameters(df_input)
    [expiries_nan, forward_input_nan, xi_input_nan] = arrays_parameters(df_input, tenor_len)
    [expiries, forward_input, xi_input] = arrays_control(expiries_nan, forward_input_nan, xi_input_nan)
    [expiries_nan, forward_input_nan, xi_input_nan] = arrays_for_output(expiries, forward_input, xi_input, tenor_len)
    n = df_input.loc['Number of Simulation'][0]
    random_seed = df_input.loc['Seed'][0]
    df_input = pd.DataFrame()

    t, mv = t_from_dt_e(dt_short, dt_inf, alpha, expiries)
    T = expiries[-1]
    expiries_index = np.where(np.isin(t,expiries))[0]  
    xi, xi_values = interpolating_xi(xi_input, expiries, mv)

    Sigma = Sigma_calculation(t,H,rho)
    L = np.linalg.cholesky(Sigma)

    logS, S, logv = spot_calculation(t, S_0, xi, eta, H, L, n, random_seed)
    
    K, C, K_skew, C_skew, warn = MC_strike_and_price(S, S_0 , expiries, t, strikes, dK_skew, xi, n, N_treshold)
    IV, IV_skew = imp_vol_calculation(K, C, K_skew, C_skew, S_0, expiries)
    skew, smile = skew_smile_calculation(IV_skew, dK_skew)
    sign_bound, Sigma0_d, Sigma0_dd, a_0 = Sigma_Taylor_coefficients(H, rho, eta, xi, expiries)
    IV_approx, skew_approx, smile_approx, IV_skew_approx = imp_vol_approx(K, K_skew, S_0, expiries, xi, H, rho, eta, tenor_len)

    df_info = df_info_creation(request_id, underlying, as_of, S_0)
    df_sim = df_sim_creation(n, N_treshold, random_seed, t.size)
    df_model = df_model_creation(H, eta, rho)
    end_time = time.time()
    total_time = end_time - start_time
    df_diagn = df_diagn_creation(dK_skew, dt_short, dt_inf, alpha, total_time, code_version, Sigma0_d, Sigma0_dd, a_0)
    df_int_var_std = df_integ_var_std_creation(logS, logv, xi_input, xi_values, t, expiries, xi, H, eta, T)

    df_term = df_term_creation(expiries_nan, forward_input_nan, xi_input_nan)
    df_strikes = df_strikes_creation(K, strikes, S_0, tenor_len)
    IV_df, df_IV = df_IV_creation(IV, strikes, tenor_len)
    df_skew_smile = df_skew_smile_creation(skew, smile, IV_skew, IV_df, strikes, dK_skew, expiries_nan, tenor_len)
    IV_approx_df, df_IV_approx = df_IV_creation(IV_approx, strikes, tenor_len, 'Approx')
    df_skew_smile_approx = df_skew_smile_creation(skew_approx, smile_approx, IV_skew_approx, IV_approx_df, strikes, dK_skew, expiries_nan, tenor_len)
    
    if request_counter == 0:
        creating_new_file(output_path, request_id)
        book = load_workbook(output_path)      # opening the file and setting writer
        writer = pd.ExcelWriter(output_path, engine = 'openpyxl', mode = 'w') 
        writer.book = book

    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    sheet_name = 'Id' + str(request_id)
    writing_parameters_col(writer, df_info, df_model, df_sim, df_diagn, df_int_var_std, sheet_name)
    writing_results(writer, df_term, df_strikes, df_IV, df_skew_smile, df_IV_approx, df_skew_smile_approx, sheet_name)

    # save_plot_path, plotting_error = vol_surface_plot(IV, K, expiries, output_path, request_id, warn)
    
writer.save()
