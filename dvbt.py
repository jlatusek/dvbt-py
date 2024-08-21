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

data_resampled = dvbt.resample(data_filtered, fs)
N_resampled = len(data_resampled)
if draw:
    n_resampled = np.linspace(-0.5, 0.5, N_resampled)
    plt.figure()
    plt.semilogy(n_resampled, np.abs(np.fft.fftshift(np.fft.fft(data_resampled))))
    plt.title("Resampled data FFT")
    plt.show()

# %% Find symbols

symbols = dvbt.find_symbol(data_resampled, draw=draw)

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
