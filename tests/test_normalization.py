from dvbt.normalize import ScatteredPilots


def test_location_scattered_min_max():
    for i in range(4):
        sp = ScatteredPilots(i)
        res = sp.locations
        assert min(res) >= sp.kmin
        assert max(res) <= sp.kmax
