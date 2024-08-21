from os.path import join as pjoin

import numpy as np
import scipy.io as sio


def main():
    dvbt_data = sio.loadmat(pjoin("data", "dvbt.mat"))
    fc = dvbt_data["fc"][0][0]
    f1 = dvbt_data["f1"][0][0]
    f2 = dvbt_data["f2"][0][0]
    fs = dvbt_data["fs"][0][0]
    data = np.squeeze(dvbt_data["data"])
    np.savez(pjoin("data", "dvbt_data.npz"), fc=fc, f1=f1, f2=f2, fs=fs, data=data)


if __name__ == "__main__":
    main()
