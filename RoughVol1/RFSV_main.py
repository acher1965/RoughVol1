# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats, optimize, special

from openpyxl import load_workbook, Workbook
import platform

# importing all functions needed
from RFSV_functions import *
from RFSV_helpers import *

def main():
    ''' The main'''
    
    input_xlsx, row_list = read_args()
    
    hc = hard_coded_params()

    destination, df_input = get_and_check_input(input_xlsx)
    
    output_file = destination + datetime.today().strftime('%Y%m%d') + '_' + datetime.now().time().strftime('%H%M%S') + '.xlsx'   
    
    writer = create_new_workbook(output_file)
        
    for row in row_list:
        r = row-1
        #calculate    
        request_id, df_diagn, df_info, df_int_var_std, df_model, df_sim, df_skew_smile, df_skew_smile_approx, df_strikes, df_term, df_IV, df_IV_approx = calculate_request(hc,df_input, r)    
        #write one sheet into the output xlsx for this request
        write_results_to_sheet(writer, request_id, df_diagn, df_info, df_int_var_std, df_model, df_sim, df_skew_smile, df_skew_smile_approx, df_strikes, df_term, df_IV, df_IV_approx)

    writer.close()


if __name__ == '__main__':
    main()
    exit(0)
