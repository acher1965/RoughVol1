"""Quick smoke runner that invokes `calculate_request` with minimal inputs.
This is intended for local quick checks only.
"""
import numpy as np
from RFSV_functions import hard_coded_params, calculate_request
from RFSV_helpers import RFSV_RowInputs


def run_smoke():
    hc = hard_coded_params()
    # minimal fake inputs
    row_inputs = RFSV_RowInputs(
        request_id='smoke',
        as_of='01-01-2000',
        n=10,
        random_seed=42,
        S_0=100.0,
        underlying='TEST',
        H=0.1,
        eta=1.0,
        rho=-0.5,
        full_diagn_flag=False,
        full_output_flag=False,
        expiries_nan=np.array([0.5,1.0]),
        forward_input_nan=np.array([100.0, 100.0]),
        xi_input_nan=np.array([0.04, 0.04])
    )

    volgrid = np.zeros((hc.tenor_len, len(hc.strikes)))
    weigthsgrid = np.ones_like(volgrid)

    res = calculate_request(hc, row_inputs, volgrid, weigthsgrid)
    print('Smoke run request_id:', res.request_id)


if __name__ == '__main__':
    run_smoke()
