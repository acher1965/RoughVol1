import os
import sys
import numpy as np

# Ensure project modules can be imported when tests run from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RoughVol1')))

import RFSV_functions as rf


def test_hard_coded_params():
    hc = rf.hard_coded_params()
    assert hasattr(hc, 'tenor_len')


def test_bsm_roundtrip():
    S0 = 100.0
    K = 110.0
    T = 1.0
    vol = 0.2
    price = rf.BSM_call_price(vol, S0, K, T)
    iv = rf.BSM_imp_vol(S0, K, T, price)
    assert not np.isnan(iv)
    assert abs(iv - vol) < 1e-3
