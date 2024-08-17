import numpy as np

from dvbt.qam import Qam


def test_print_constellation():
    # Example usage
    qam = Qam()
    qam.draw_constellation()
    assert True


def test_qam_64():
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

    demodulated_signal = qam.demodulate(qam_signal)
    assert np.array_equal(demodulated_signal, np.array([0, 2, 3, 1, 60, 62, 63, 61]))


def test_qam_16():
    qam = Qam(16)

    qam_signal = np.array(
        [
            complex(4, 4),
            complex(4, -4),
            complex(-4, -4),
            complex(-4, 4),
            complex(2, 2),
            complex(2, -2),
            complex(-2, -2),
            complex(-2, 2),
        ]
    )
    qam_signal /= np.abs(qam_signal).max()
    qam_signal *= np.abs(qam._max_point_position)

    demodulated_signal = qam.demodulate(qam_signal)
    assert np.array_equal(demodulated_signal, np.array([0, 2, 3, 1, 12, 14, 15, 13]))


def test_generation_64():
    qam = Qam(64)
    qam.draw_constellation()


def test_generation_16():
    qam = Qam(16)
    qam.draw_constellation()
