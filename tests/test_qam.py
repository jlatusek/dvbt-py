import numpy as np
from matplotlib import pyplot as plt

from dvbt.qam import Qam


def test_print_constellation():
    # Example usage
    qam = Qam()
    qam.draw_constellation()
    assert True


def test_qam():
    qam = Qam()

    qam_signal = np.array(
        [
            complex(8, 8),
            complex(8, -8),
            complex(-8, -8),
            complex(-8, 8),
            complex(4, 4),
            complex(4, -4),
            complex(-4, -4),
            complex(-4, 4),
        ]
    )
    qam_signal /= np.abs(qam_signal).max()
    qam_signal *= np.abs(qam._max_point_position)
    plt.figure()
    plt.plot(qam_signal.real, qam_signal.imag, "o")
    plt.title("Signal original")
    plt.show()

    demodulated_signal = qam.demodulate(qam_signal)
    print(demodulated_signal)
    assert np.array_equal(demodulated_signal, np.array([0, 2, 3, 1, 60, 62, 63, 61]))


def test_generation_64():
    qam = Qam(64)
    qam.draw_constellation()


def test_generation_16():
    qam = Qam(16)
    qam.draw_constellation()
