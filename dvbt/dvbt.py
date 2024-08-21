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


def resample(data: np.array, fs: int):
    N = len(data)
    B = 8e6
    fs_dvb = 8 / 7 * B
    data_resampled = signal.resample(data, int(N * fs_dvb / fs))
    return data_resampled


def find_symbol(data: np.array, draw=False):
    symbol_per_block = 8192
    guard_interval = 1 / 8
    N = len(data)
    guard_symbols = int(symbol_per_block * guard_interval)
    frame_len = int(symbol_per_block + guard_symbols)
    correlation = np.zeros(N, dtype=complex)
    for i in range(N):
        if i + symbol_per_block + guard_symbols > N:
            break
        first_guard = data[i : i + guard_symbols]
        second_guard = data[i + symbol_per_block : i + symbol_per_block + guard_symbols]
        corr = np.correlate(first_guard, second_guard)
        correlation[i] = corr[0]

    if draw:
        plt.figure()
        plt.plot(np.abs(correlation))
        plt.title("Correlation of guards")
        plt.show()

    # Find peaks in data source signal to find beginning of each block

    peaks = signal.find_peaks(np.abs(correlation), height=1e9, distance=symbol_per_block)
    if draw:
        plt.figure()
        plt.stem(peaks[0], peaks[1]["peak_heights"])
        plt.title("Peaks in correlation")
        plt.show()

    # Extract symbols
    symbol_end = np.arange(peaks[0][0] + frame_len, N, 8192 + 1024, dtype=int)
    data_no_guard = np.empty((len(symbol_end), 8192), dtype=complex)
    for idx, val in enumerate(symbol_end):
        data_no_guard[idx] = data[val - 8192 : val]

    data_af_fft = np.fft.fft(data_no_guard)
    data_af_fft = np.fft.fftshift(data_af_fft, axes=1)
    if draw:
        plt.figure()
        plt.plot(data_af_fft[0].real, data_af_fft[0].imag, ".")
        plt.title("Symbol po FFT")
        plt.xlim([-3e5, 3e5])
        plt.ylim([-3e5, 3e5])
        plt.show()

    # Remove zeros from the beginning and the end of the symbol
    full_symbols = data_af_fft
    symbols = full_symbols[:, 688:-687]
    one_symbol = symbols[0]

    if draw:
        plt.figure(figsize=(10, 10))
        plt.plot(one_symbol.real, one_symbol.imag, ".")
        plt.title("Symbol po usunięciu zer")
        plt.xlim([-3e5, 3e5])
        plt.ylim([-3e5, 3e5])
        plt.show()
    return symbols
