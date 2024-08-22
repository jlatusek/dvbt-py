from os.path import join as pjoin

import numpy as np
import pytest

from dvbt import tps
from tests.tools import PROJECT_DIR


@pytest.fixture(scope="module")
def symbols() -> np.ndarray[np.complex64]:
    data = np.load(pjoin(PROJECT_DIR, "data", "dvbt_symbols.npz"))
    symbols = data["symbols"][0:512]
    return symbols


def test_find_symbol(symbols):
    tps_bits = tps.extract_bit_data(symbols)
    frame1_idx = tps.find_frame_start(tps_bits, 0)
    frame2_idx = tps.find_frame_start(tps_bits[frame1_idx + 1 :], 1)
    frame3_idx = tps.find_frame_start(tps_bits[frame1_idx + frame2_idx + 2 :], 2)
    assert frame2_idx + 1 == 68
    assert frame3_idx + 1 == 68
