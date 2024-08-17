import matplotlib.pyplot as plt
import numpy as np
import pytest

from dvbt.qam import Qam


@pytest.mark.parametrize("m", [4, 16, 64])
def test_print_constellation(m):
    qam = Qam(m)
    qam.draw_constellation()


@pytest.mark.parametrize("noise", [0, 0.001, 0.004])
def test_qam_64(noise):
    qam = Qam()
    qam_signal = np.array(
        [
            complex(7, 7),
            complex(7, -7),
            complex(-7, -7),
            complex(-7, 7),
            complex(3, 3),
            complex(3, -3),
            complex(-3, -3),
            complex(-3, 3),
        ]
    )
    qam_signal /= np.abs(qam_signal).max()
    qam_signal *= np.abs(qam._max_point_position)
    result = np.array([0, 2, 3, 1, 60, 62, 63, 61])
    
    if noise > 0:
        qam_signal = np.repeat(qam_signal, 100)
        result = np.repeat(result, 100)

    num_symbols = len(qam_signal)
    n = (np.random.uniform(-1, 1, num_symbols) + 1j * np.random.uniform(-1, 1, num_symbols)) / np.sqrt(2)
    noise = np.sqrt(noise) * n
    qam_signal += noise

    plt.figure()
    plt.scatter(qam_signal.real, qam_signal.imag, color="red")
    plt.scatter(qam.constellation.real, qam.constellation.imag, color="blue")
    plt.show()

    demodulated_signal = qam.demodulate(qam_signal)
    assert np.array_equal(demodulated_signal, result)


def test_qam_16():
    qam = Qam(16)

    qam_signal = np.array(
        [
            complex(3, 3),
            complex(3, -3),
            complex(-3, -3),
            complex(-3, 3),
            complex(1, 1),
            complex(1, -1),
            complex(-1, -1),
            complex(-1, 1),
        ]
    )
    qam_signal /= np.abs(qam_signal).max()
    qam_signal *= np.abs(qam._max_point_position)

    plt.figure()
    plt.scatter(qam.constellation.real, qam.constellation.imag, color="blue", marker="o")
    plt.scatter(qam_signal.real, qam_signal.imag, color="red", marker=".")
    plt.show()

    demodulated_signal = qam.demodulate(qam_signal)
    assert np.array_equal(demodulated_signal, np.array([0, 2, 3, 1, 12, 14, 15, 13]))
