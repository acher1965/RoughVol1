# -*- coding: utf-8 -*-
'''The main() for RFSV bruteforce, rBergomi model.

Overall logic:
read inputs from excel input file
open an excel writer for output excel file
loop over each request:
 calling the math logic
 write results to a sheet per request
finally close output excel file
'''

import time
from datetime import datetime
import pandas as pd
from RFSV_functions import hard_coded_params, calculate_request
from RFSV_helpers import read_args, get_and_check_input, get_inputs_from_row, create_new_workbook, write_results_to_sheet
from RFSV_functions_Error import create_and_write_df_cost

def main():
    '''The main for RFSV bruteforce, rBergomi model'''
    
    start_time = time.time()

    #get inputs
    input_xlsx, row_list = read_args()
        
    hc = hard_coded_params()

    destination, df_input = get_and_check_input(input_xlsx)
    
    #calculate
    results = []
    volgrid = pd.read_excel(input_xlsx, header = 0, index_col = 0, sheet_name = 'VolGrid').to_numpy(dtype = 'float64', copy = True)
    weigthsgrid = pd.read_excel(input_xlsx, header = 0, index_col = 0, sheet_name = 'WeightsGrid').to_numpy(dtype = 'float64', copy = True)
    for row in row_list:
        print('Calculating row' + str(row) + ' ...')
        r = row - 1
        row_inputs = get_inputs_from_row(df_input, r, hc.tenor_len)
        results.append(calculate_request(hc, row_inputs, volgrid, weigthsgrid))

    print('End of calculations, saving ...')

    #save to excel
    output_file = destination + datetime.now().strftime('%Y%m%d_%H%M%S') + '.xlsx'         #+ '_'.join([str(id) for id in row_list]) + '.xlsx'
    writer = create_new_workbook(output_file)
    writer_flag = False
    for r in results:
        if r.full_output_flag:
            write_results_to_sheet(writer, r)
            writer_flag = True

    if writer_flag : writer.close()
    create_and_write_df_cost(output_file, results)
    print('Saved to excel: ' + output_file)
    end_time = time.time()
    print('END. Calc time: ' + str(end_time - start_time))

if __name__ == '__main__':
    main()
    exit(0)
