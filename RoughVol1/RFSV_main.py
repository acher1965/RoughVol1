# -*- coding: utf-8 -*-
''' This module has the main() for RFSV bruteforce, rBergomi model
with the overall logic:
read inputs from excel input file
open an excel writer for output excel file
loop over each request:
 calling the math logic
 write results to a sheet per request
finally close output excel file
'''

from datetime import datetime
from RFSV_functions import hard_coded_params, calculate_request
from RFSV_helpers import read_args, get_and_check_input, create_new_workbook, write_results_to_sheet

def main():
    ''' The main for RFSV bruteforce, rBergomi model'''
    
    input_xlsx, row_list = read_args()
    
    hc = hard_coded_params()

    destination, df_input = get_and_check_input(input_xlsx)
    
    output_file = destination + datetime.today().strftime('%Y%m%d') + '_' + datetime.now().time().strftime('%H%M%S') + '.xlsx'   
    
    writer = create_new_workbook(output_file)
        
    for row in row_list:
        r = row - 1        
        results = calculate_request(hc,df_input, r)        
        write_results_to_sheet(writer, results)
        
    writer.close()

if __name__ == '__main__':
    main()
    exit(0)
