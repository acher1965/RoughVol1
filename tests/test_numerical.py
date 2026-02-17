import numpy as np
import os, sys

# Ensure project modules can be imported when tests run from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RoughVol1')))

import RFSV_functions as rf
import RFSV_functions_Error as re


def test_bsm_imp_vol_no_solution():
    # price larger than S0 for a call cannot be matched -> solver should return NaN
    S0 = 100.0
    K = 120.0  # K >= S0 triggers call branch
    T = 1.0
    price = S0 + 50.0
    iv = rf.BSM_imp_vol(S0, K, T, price)
    assert np.isnan(iv)


def test_fit_cost_calculation_shape_and_values():
    # small IV and grid where fit cost can be computed; tenor_len=2 -> expect 3 rows (2 expiries + global)
    IV = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    # fit_cost_calculation computes (IV.T - volgrid) so volgrid must match IV.T shape
    volgrid = IV.T.copy()
    weigthsgrid = np.ones_like(volgrid)
    tenor_len = 2
    fit_cost_arr = re.fit_cost_calculation(IV, volgrid, weigthsgrid, tenor_len)
    assert fit_cost_arr.shape[0] == tenor_len + 1
    # When IV equals volgrid and weights are ones, expect zeros (or near-zero)
    assert np.allclose(fit_cost_arr.flatten()[:-1], 0.0)


def test_cost_row_df_creation_columns():
    tenor_len = 3
    # create a cost_row with (6 + tenor_len + 1) rows and single column
    ncols = 6 + tenor_len + 1
    cost_row = np.arange(ncols)[:, np.newaxis]
    df = re.cost_row_df_creation(cost_row, tenor_len)
    expected_cols = ['Id', 'Underlying', 'AsOf', 'H', 'eta', 'rho'] + [f'CostExpiry{i}' for i in range(1, tenor_len+1)] + ['GlobalCost']
    assert list(df.columns) == expected_cols
