import numpy as np
from matplotlib import pyplot as plt

from .prbs import prbs


class ScatteredPilots:
    kmin = 0
    kmax = 6816

    def __init__(self, symbol_number):
        self.symbol_number = symbol_number

    def _generate_location(self, idx) -> int:
        return self.kmin + 3 * (self.symbol_number % 4) + 12 * idx

    @property
    def locations(self):
        locations = []
        i = 0
        last_location = self._generate_location(i)
        while last_location <= self.kmax:
            locations.append(last_location)
            last_location = self._generate_location(i)
            i += 1
        return np.array(locations)


def normalize_sig(sym, symbol_number, draw=False):
    # % Input:        sym          = odebrany symbol
    # %               l            = numer symbolu
    # % Output: output = znomalizowany symbol

    # %% generowanie pilotów
    # fmt: off
    pilot_index_continual = np.array(
        [
            0, 48, 54, 87, 141, 156, 192, 201, 255, 279, 282, 333, 432, 450, 483, 525, 531, 618, 636, 714, 759, 765,
            780, 804, 873, 888, 918, 939, 942, 969, 984, 1050, 1101, 1107, 1110, 1137, 1140, 1146, 1206, 1269, 1323,
            1377, 1491, 1683, 1704, 1752, 1758, 1791, 1845, 1860, 1896, 1905, 1959, 1983, 1986, 2037, 2136, 2154, 2187,
            2229, 2235, 2322, 2340, 2418, 2463, 2469, 2484, 2508, 2577, 2592, 2622, 2643, 2646, 2673, 2688, 2754, 2805,
            2811, 2814, 2841, 2844, 2850, 2910, 2973, 3027, 3081, 3195, 3387, 3408, 3456, 3462, 3495, 3549, 3564, 3600,
            3609, 3663, 3687, 3690, 3741, 3840, 3858, 3891, 3933, 3939, 4026, 4044, 4122, 4167, 4173, 4188, 4212, 4281,
            4296, 4326, 4347, 4350, 4377, 4392, 4458, 4509, 4515, 4518, 4545, 4548, 4554, 4614, 4677, 4731, 4785, 4899,
            5091, 5112, 5160, 5166, 5199, 5253, 5268, 5304, 5313, 5367, 5391, 5394, 5445, 5544, 5562, 5595, 5637, 5643,
            5730, 5748, 5826, 5871, 5877, 5892, 5916, 5985, 6000, 6030, 6051, 6054, 6081, 6096, 6162, 6213, 6219, 6222,
            6249, 6252, 6258, 6318, 6381, 6435, 6489, 6603, 6795, 6816,
        ]
    )
    # fmt: on
    sp = ScatteredPilots(symbol_number)

    all_pilot = np.append(pilot_index_continual, sp.locations)
    all_pilot = np.unique(all_pilot)

    # %%Tworzenie odpowiedzi impulsowej

    prbs_sequence = prbs(np.ones([11, 1]))
    pilots_sequence = 4 / 3 * 2 * (1 / 2 - prbs_sequence)
    ilor = sym[all_pilot] / pilots_sequence[all_pilot]
    xfit = np.arange(6817)
    interpol = np.interp(xfit, all_pilot, ilor)

    # %% Signal normalization
    data_normalized = sym / interpol

    if draw:
        plt.figure()
        plt.plot(data_normalized.real, data_normalized.imag, ".", markersize=1)
        plt.title("Signal after normalization and channel compensation")
        plt.show()
    return data_normalized
