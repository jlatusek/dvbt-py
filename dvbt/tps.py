from dataclasses import dataclass

import numpy as np


class Tps:
    #  fmt: off
    tps_sig_idx = np.array([
        34, 50, 209, 346, 413, 569, 595, 688, 790, 901, 1073, 1219, 1262, 1286,
        1469, 1594, 1687, 1738, 1754, 1913, 2050, 2117, 2273, 2299, 2392, 2494, 
        2605, 2777, 2923, 2966, 2990, 3173, 3298, 3391, 3442, 3458, 3617, 3754,
        3821, 3977, 4003, 4096, 4198, 4309, 4481, 4627, 4670, 4694, 4877, 5002,
        5095, 5146, 5162, 5321, 5458, 5525, 5681, 5707, 5800, 5902, 6013, 6185,
        6331, 6374, 6398, 6581, 6706, 6799,
    ])
    # fmt: on

    class Bits:
        Initialisation = 0
        SynchronizationWord = np.arange(1, 17)
        LengthIndicator = np.arange(17, 23)
        FrameNumber = np.arange(23, 25)
        Constellation = np.arange(25, 27)
        HierarchyInformation = np.arange(27, 30)
        CodeRateHP = np.arange(30, 33)
        CodeRateLP = np.arange(33, 36)
        GuardInterval = np.arange(36, 38)
        TransmissionMode = np.arange(38, 40)
        CellID = np.arange(40, 48)
        AnexF = np.arange(48, 54)  # Supposed all to be 0
        ErrorProtection = np.arange(54, 68)

    @dataclass
    class Values:
        Initialisation: np.ndarray
        SynchronizationWord: np.ndarray
        LengthIndicator: np.ndarray
        FrameNumber: np.ndarray
        Constellation: np.ndarray
        HierarchyInformation: np.ndarray
        CodeRateHP: np.ndarray
        CodeRateLP: np.ndarray
        GuardInterval: np.ndarray
        TransmissionMode: np.ndarray
        CellID: np.ndarray
        AnexF: np.ndarray
        ErrorProtection: np.ndarray

    @classmethod
    def extract(cls, symbol: np.ndarray) -> Values:
        values = symbol[cls.tps_sig_idx]
        return Tps.Values(
            Initialisation=values[Tps.Bits.Initialisation],
            SynchronizationWord=values[Tps.Bits.SynchronizationWord],
            LengthIndicator=values[Tps.Bits.LengthIndicator],
            FrameNumber=values[Tps.Bits.FrameNumber],
            Constellation=values[Tps.Bits.Constellation],
            HierarchyInformation=values[Tps.Bits.HierarchyInformation],
            CodeRateHP=values[Tps.Bits.CodeRateHP],
            CodeRateLP=values[Tps.Bits.CodeRateLP],
            GuardInterval=values[Tps.Bits.GuardInterval],
            TransmissionMode=values[Tps.Bits.TransmissionMode],
            CellID=values[Tps.Bits.CellID],
            AnexF=values[Tps.Bits.AnexF],
            ErrorProtection=values[Tps.Bits.ErrorProtection],
        )
