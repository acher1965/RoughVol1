# -*- coding: utf-8 -*-
"""Non-Math helper functions for input, checks, and output

For RFSV bruteforce, rBergomi model
"""

import collections  # why this not working??  from collections import namedtuple
from dataclasses import dataclass
from os import chdir, path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def read_args() -> Tuple[str, list, int]:
    """Read the command line arguments"""
    # e.g.: --help
    # or: -f C:/Users/abech/OneDrive/EQFAugustoShare/Alberto/Running -x
    # RFSV_input_AC.xlsx 1 2 3 20
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a set of RoughVol Brute-Force requests"
    )
    parser.add_argument("-f", "--folder", default=".", help="the folder to run from")
    parser.add_argument(
        "-x",
        "--xlsx",
        default="RFSV_input.xlsx",
        help="the input excel spreadsheet , each row defines a request",
    )
    parser.add_argument(
        "row_list",
        type=int,
        nargs="+",
        help="the set of rows in XLSX to run. 0 to run the last one.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument('row_list',type=int,nargs='+', help='the set of rows in
    # XLSX to run.  0 to run the last one.')
    # group.add_argument('-a','--all',action='store_true', help='NOT IMPLEMENTED
    # YET (to run all requests in the file')
    args = parser.parse_args()

    chdir(args.folder)  # set the running folder
    input_xlsx = args.xlsx
    row_list = (
        args.row_list
    )  # note: they are actually row-index (1 stands for 3rd row), not the ids...

    check_request_list(row_list)
    log_level = getattr(__import__("logging"), args.log_level)
    return input_xlsx, row_list, log_level


def check_request_list(row_list):
    """TODO add more checks"""
    # consistency of destination across all rows requested
    if not isinstance(row_list, list):
        raise TypeError("'row_list' must be a list, even with just an element.")


def get_and_check_input(input_path) -> Tuple[str, "pd.DataFrame"]:
    """Get inputs common to this set of requests, and check them for consistency.

    Returns:
        destination (str), df_input_T (pd.DataFrame)
    """
    df_input = pd.read_excel(input_path, header=0, skiprows=1)
    index = df_input.columns
    df_input_T = df_input.T
    df_input_T.index = index
    df_input = df_input.replace("#N/A\xa0", np.nan)
    destinations = df_input_T.loc["Destination"]
    destination = destinations[0]
    for d in destinations:
        if d != destination:
            raise ValueError("destination must be the same in all rows in the row_list")

    return destination, df_input_T


@dataclass
class RFSV_RowInputs:
    request_id: Any
    as_of: Any
    n: int
    random_seed: int
    S_0: float
    underlying: Any
    H: float
    eta: float
    rho: float
    full_diagn_flag: bool
    full_output_flag: bool
    expiries_nan: NDArray[Any]
    forward_input_nan: NDArray[Any]
    xi_input_nan: NDArray[Any]


def get_inputs_from_row(
    df_input: "pd.DataFrame", row: int, tenor_len: int
) -> RFSV_RowInputs:
    """Get row specific inputs"""

    r_idx = int(row)
    request_id = df_input.loc["Row"][r_idx]
    as_of = df_input.loc["Date"][r_idx].strftime(
        "%d-%m-%Y"
    )  # changing format using strftime(), see https://strftime.org/
    S_0 = df_input.loc["Spot"][r_idx]
    underlying = df_input.loc["Underlying"][r_idx]
    n = df_input.loc["Simulations"][r_idx]
    random_seed = df_input.loc["Seed"][r_idx]
    H = df_input.loc["Roughness(Hurst)"][r_idx]
    eta = df_input.loc["VolVol(Eta)"][r_idx]
    rho = df_input.loc["Correlation(Rho)"][r_idx]
    full_diagn_flag = df_input.loc["FullDiagnostic"][r_idx]
    full_output_flag = df_input.loc["FullOutput"][r_idx]

    start_label: Any = "Tenor1"
    end_label: Any = "Tenor" + str(tenor_len)
    expiries_nan = (
        df_input.loc[start_label:end_label]
        .iloc[:, r_idx]
        .to_numpy(dtype="float64", copy=True)
        .reshape(tenor_len)
    )
    start_label = "Fwd1"
    end_label = "Fwd" + str(tenor_len)
    forward_input_nan = (
        df_input.loc[start_label:end_label]
        .iloc[:, r_idx]
        .to_numpy(dtype="float64", copy=True)
        .reshape(tenor_len)
    )
    start_label = "Impvar1"
    end_label = "Impvar" + str(tenor_len)
    xi_input_nan = (
        df_input.loc[start_label:end_label]
        .iloc[:, r_idx]
        .to_numpy(dtype="float64", copy=True)
        .reshape(tenor_len)
    ) ** 2
    return RFSV_RowInputs(
        request_id,
        as_of,
        n,
        random_seed,
        S_0,
        underlying,
        H,
        eta,
        rho,
        full_diagn_flag,
        full_output_flag,
        expiries_nan,
        forward_input_nan,
        xi_input_nan,
    )


def create_new_workbook(filename: str) -> "pd.ExcelWriter":
    """Create a new excel workbook and return it.

    Raises FileExistsError if the target file already exists.
    """
    if path.exists(filename):
        raise FileExistsError(f"{filename} already exists.")

    writer = pd.ExcelWriter(filename, engine="openpyxl", mode="w")
    return writer


def write_results_to_sheet(writer, results):
    """Adds a sheet to workbook 'writer' and writes the results of this request to it"""
    r = results
    sheet_name = "Id" + str(r.request_id)
    writing_parameters_col(
        writer,
        r.df_info,
        r.df_model,
        r.df_sim,
        r.df_diagn,
        r.df_int_var_std,
        sheet_name,
    )
    writing_results(
        writer,
        r.df_term,
        r.df_strikes,
        r.df_IV,
        r.df_skew_smile,
        r.df_IV_approx,
        r.df_skew_smile_approx,
        r.df_err_IV,
        sheet_name,
    )
    # writer.save()
    return


def writing_parameters_col(
    writer, df_info, df_model, df_sim, df_diagn, df_int_var_std, sheet_name
):
    startrow = 0
    df_info.to_excel(writer, header=None, index=True, sheet_name=sheet_name)
    startrow = startrow + df_info.shape[0] + 1
    df_model.to_excel(
        writer, header=True, index=True, startrow=startrow, sheet_name=sheet_name
    )
    startrow = startrow + df_sim.shape[0] + 2
    df_sim.to_excel(
        writer, header=True, index=True, startrow=startrow, sheet_name=sheet_name
    )
    startrow = startrow + df_sim.shape[0] + 2
    df_diagn.to_excel(
        writer, header=True, index=True, startrow=startrow, sheet_name=sheet_name
    )
    startrow = startrow + df_diagn.shape[0] + 2
    df_int_var_std.to_excel(
        writer, header=True, index=True, startrow=startrow, sheet_name=sheet_name
    )


def writing_results(
    writer,
    df_term,
    df_strikes,
    df_IV,
    df_skew_smile,
    df_IV_approx,
    df_skew_smile_approx,
    df_err_IV,
    sheet_name,
):
    startrow = 0
    df_term.to_excel(
        writer,
        header=None,
        index=True,
        na_rep="#N/A",
        startcol=3,
        startrow=startrow,
        sheet_name=sheet_name,
    )
    startrow = startrow + df_term.shape[0]
    df_strikes.to_excel(
        writer,
        header=True,
        index=True,
        na_rep="#N/A",
        startcol=3,
        startrow=startrow,
        sheet_name=sheet_name,
    )
    startrow = startrow + df_strikes.shape[0] + 1
    df_IV.to_excel(
        writer,
        header=True,
        index=True,
        na_rep="#N/A",
        startcol=3,
        startrow=startrow,
        sheet_name=sheet_name,
    )
    startrow = startrow + df_IV.shape[0] + 1
    df_skew_smile.to_excel(
        writer,
        header=True,
        index=True,
        na_rep="#N/A",
        startcol=3,
        startrow=startrow,
        sheet_name=sheet_name,
    )
    startrow = startrow + df_skew_smile.shape[0] + 1
    df_IV_approx.to_excel(
        writer,
        header=True,
        index=True,
        na_rep="#N/A",
        startcol=3,
        startrow=startrow,
        sheet_name=sheet_name,
    )
    startrow = startrow + df_IV_approx.shape[0] + 1
    df_skew_smile_approx.to_excel(
        writer,
        header=True,
        index=True,
        na_rep="#N/A",
        startcol=3,
        startrow=startrow,
        sheet_name=sheet_name,
    )
    startrow = startrow + df_skew_smile_approx.shape[0] + 1
    df_err_IV.to_excel(
        writer,
        header=True,
        index=True,
        na_rep="#N/A",
        startcol=3,
        startrow=startrow,
        sheet_name=sheet_name,
    )
