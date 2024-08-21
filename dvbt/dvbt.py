import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt
import scipy.signal as signal


def channel_filter(data: np.array, draw=False):
    cutoff = 7e6  # Pass band Frequency
    fs = 30000000  # Sampling FrequencyV
    h = signal.firwin(100, cutoff, fs=fs)
    filtered_data = signal.lfilter(h, 1.0, data)
    if draw:
        w, h = signal.freqz(h, 1.0, worN=8000)
        plt.semilogy(w / np.pi, abs(h))
        plt.title("Filter Frequency Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain")
        plt.grid(True)
        plt.show()
        # Plot phase response
        plt.plot(w / np.pi, np.unwrap(np.angle(h)))
        plt.title("Filter Phase Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase")
        plt.grid(True)
        plt.show()
    return filtered_data


def mix_frequencies(data: npt.NDArray[np.complex64], fs: int, fo: int) -> npt.NDArray[np.complex64]:
    fo = 8e6 / fs
    k = np.arange(1, len(data) + 1)
    c = np.exp(1j * (2 * np.pi * fo * k))
    data_low = data * c
    return data_low
