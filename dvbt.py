#!/usr/bin/env python

from os.path import join as pjoin
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.signal as signal

import dvbt.dvbt as dvbt
from dvbt.normalize import normalize_sig
from dvbt.qam import Qam
from dvbt.tps import Tps

draw = False

draw = True

# %% Load data
data = np.load(pjoin("data", "dvbt.npz"))
fc = data["fc"]
f1 = data["f1"]
f2 = data["f2"]
fs = data["fs"]
data = data["data"]

# For testing purposes limit number of processed data
data = data[0:100000]
N = len(data)
n = np.linspace(-0.5, 0.5, N)

# %% Plot fft of received data

if draw:
    plt.semilogy(n, np.abs(np.fft.fftshift(np.fft.fft(data))))
    plt.title("FFT of received data")
    plt.show()

# %% Mix frequencies to lower frequency of examined signal

data_low = dvbt.mix_frequencies(data, fs, fc)

if draw:
    plt.figure()
    plt.plot(n, 20 * np.log10(np.abs(np.fft.fft(data_low))))
    plt.title("Lowered data FFT")
    plt.show()


# %% Filter received data

data_filtered = dvbt.channel_filter(data_low, draw=False)

if draw:
    plt.figure()
    plt.semilogy(n, np.abs(np.fft.fftshift(np.fft.fft(data_filtered))))
    plt.title("Filtered data FFT")
    plt.xlim([-0.3, 0.3])
    plt.show()

# %% Resampling

B = 8e6
fs_dvb = 8 / 7 * B
data_resampled = signal.resample(data_filtered, int(N * fs_dvb / fs))
# data_resampled = signal.resample_poly(data_filtered, int(fs_dvb), int(fs))
N_resampled = len(data_resampled)
n_resampled = np.linspace(-0.5, 0.5, N_resampled)
if draw:
    plt.figure()
    plt.semilogy(n_resampled, np.abs(np.fft.fftshift(np.fft.fft(data_resampled))))
    plt.title("Resampled data FFT")
    plt.show()

# %% Find symbols

symbol_per_block = 8192
guard_interval = 1 / 8
guard_symbols = int(symbol_per_block * guard_interval)
frame_len = int(symbol_per_block + guard_symbols)
correlation = np.zeros(N_resampled, dtype=complex)
for i in range(N_resampled):
    if i + symbol_per_block + guard_symbols > N_resampled:
        break
    first_guard = data_resampled[i : i + guard_symbols]
    second_guard = data_resampled[i + symbol_per_block : i + symbol_per_block + guard_symbols]
    corr = np.correlate(first_guard, second_guard)
    correlation[i] = corr[0]

if draw:
    plt.figure()
    plt.plot(np.abs(correlation))
    plt.title("Correlation of guards")
    plt.show()

# %% Find peaks in data source signal to find beginning of each block

peaks = signal.find_peaks(np.abs(correlation), height=1e9, distance=symbol_per_block)
if draw:
    plt.figure()
    plt.stem(peaks[0], peaks[1]["peak_heights"])
    plt.title("Peaks in correlation")
    plt.show()

# %% Extract symbols
symbol_end = np.arange(peaks[0][0] + frame_len, len(data_resampled), 8192 + 1024, dtype=int)
data_no_guard = np.empty((len(symbol_end), 8192), dtype=complex)
for idx, val in enumerate(symbol_end):
    data_no_guard[idx] = data_resampled[val - 8192 : val]

data_af_fft = np.fft.fft(data_no_guard)
data_af_fft = np.fft.fftshift(data_af_fft, axes=1)
if draw:
    plt.figure()
    plt.plot(data_af_fft[0].real, data_af_fft[0].imag, ".")
    plt.title("Symbol po FFT")
    plt.xlim([-3e5, 3e5])
    plt.ylim([-3e5, 3e5])
    plt.show()


# %% Remove zeros from the beginning and the end of the symbol

full_symbols = data_af_fft
# % symbols = fftshift(data_af_fft);
begin_remove = full_symbols[:, 0:688]
end_remove = full_symbols[:, -687:]
symbols = full_symbols[:, 688:-687]
one_symbol = symbols[0]

if draw:
    plt.figure(figsize=(10, 10))
    plt.plot(one_symbol.real, one_symbol.imag, ".")
    plt.title("Symbol po usunięciu zer")
    plt.xlim([-3e5, 3e5])
    plt.ylim([-3e5, 3e5])
    plt.show()


# %% Normalizowanie sygnału
ind = 1
l = 3
normalized = normalize_sig(symbols[0], l, draw=True)

# %% QAM demodulation
qam = Qam()
demodulated = qam.demodulate(normalized)
qam.draw_constellation()
plt.figure()
plt.scatter(normalized.real, normalized.imag, marker=".", color="blue")
plt.scatter(qam.constellation.real, qam.constellation.imag, color="red", marker="o")
plt.show()

# %% TPS extraction

information = Tps.extract(demodulated)
pprint(information)
