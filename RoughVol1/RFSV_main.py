# -*- coding: utf-8 -*-
"""The main() for RFSV bruteforce, rBergomi model.

Overall logic:
read inputs from excel input file
open an excel writer for output excel file
loop over each request:
 calling the math logic
 write results to a sheet per request
finally close output excel file
"""

import logging
import time
from datetime import datetime

import pandas as pd
from RFSV_functions import calculate_request, hard_coded_params
from RFSV_functions_Error import create_and_write_df_cost
from RFSV_helpers import (
    create_new_workbook,
    get_and_check_input,
    get_inputs_from_row,
    read_args,
    write_results_to_sheet,
)

# configure basic logging for CLI runs
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def main():
    """The main for RFSV bruteforce, rBergomi model"""

    start_time = time.time()

    # get inputs
    input_xlsx, row_list, log_level = read_args()
    logging.getLogger().setLevel(log_level)

    hc = hard_coded_params()

    destination, df_input = get_and_check_input(input_xlsx)

    # calculate
    results = []
    volgrid = pd.read_excel(
        input_xlsx, header=0, index_col=0, sheet_name="VolGrid"
    ).to_numpy(dtype="float64", copy=True)
    weigthsgrid = pd.read_excel(
        input_xlsx, header=0, index_col=0, sheet_name="WeightsGrid"
    ).to_numpy(dtype="float64", copy=True)
    for row in row_list:
        logging.info("Calculating row %s ...", row)
        r = row - 1
        row_inputs = get_inputs_from_row(df_input, r, hc.tenor_len)
        results.append(calculate_request(hc, row_inputs, volgrid, weigthsgrid))

    logging.info("End of calculations, saving ...")

    # save to excel
    output_file = (
        destination + datetime.now().strftime("%Y%m%d_%H%M%S") + ".xlsx"
    )  # + '_'.join([str(id) for id in row_list]) + '.xlsx'
    writer = create_new_workbook(output_file)
    writer_flag = False
    for r in results:
        if r.full_output_flag:
            write_results_to_sheet(writer, r)
            writer_flag = True

    if writer_flag:
        writer.close()
    create_and_write_df_cost(output_file, results)
    logging.info("Saved to excel: %s", output_file)
    end_time = time.time()
    logging.info("END. Calc time: %s", str(end_time - start_time))


if __name__ == "__main__":
    main()
    exit(0)
