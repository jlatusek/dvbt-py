#!/usr/bin/env python

from os.path import join as pjoin

import numpy as np

from dvbt import dvbt


def main():
    data = np.load(pjoin("data", "dvbt.npz"))
    fc = data["fc"]
    f1 = data["f1"]
    f2 = data["f2"]
    fs = data["fs"]
    data = data["data"]

    data_low = dvbt.mix_frequencies(data, fs, fc)
    data_filtered = dvbt.channel_filter(data_low)
    data_resampled = dvbt.resample(data_filtered, fs)
    symbols = dvbt.find_symbol(data_resampled)

    np.savez_compressed(pjoin("data", "dvbt_symbols.npz"), symbols=symbols, fc=fc, f1=f1, f2=f2, fs=fs)


if __name__ == "__main__":
    main()
