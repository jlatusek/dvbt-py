import numpy as np
from matplotlib import pyplot as plt

from dvbt.qam import Qam


def test_print_constellation():
    # Example usage
    qam = Qam()
    qam.draw_constellation()
    assert True


def test_qam():
    # Example usage
    qam_signal = np.array([complex(1, 1), complex(-1, -1), complex(1, -1), complex(-1, 1)])
    plt.figure()
    plt.plot(qam_signal.real, qam_signal.imag, "o")
    plt.title("Signal original")
    plt.show()

    qam = Qam()
    demodulated_signal = qam.demodulate(qam_signal)
    print(demodulated_signal)
    assert demodulated_signal == np.array([0, 48, 16, 32])


def test_generation_64():
    qam = Qam(64)
    qam.draw_constellation()


def test_generation_16():
    qam = Qam(16)
    qam.draw_constellation()
