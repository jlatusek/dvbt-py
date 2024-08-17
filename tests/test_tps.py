import numpy as np


def test_extraction():
    res = Tps.extract(np.arange(6817))
    assert res.Initialisation == 0
