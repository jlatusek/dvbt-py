from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


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
    def extract(cls, symbol: npt.NDArray[[]]) -> Values:
        values = symbol[cls.tps_sig_idx]
        return values
        # return Tps.Values(
        #     Initialisation=values[Tps.Bits.Initialisation],
        #     SynchronizationWord=values[Tps.Bits.SynchronizationWord],
        #     LengthIndicator=values[Tps.Bits.LengthIndicator],
        #     FrameNumber=values[Tps.Bits.FrameNumber],
        #     Constellation=values[Tps.Bits.Constellation],
        #     HierarchyInformation=values[Tps.Bits.HierarchyInformation],
        #     CodeRateHP=values[Tps.Bits.CodeRateHP],
        #     CodeRateLP=values[Tps.Bits.CodeRateLP],
        #     GuardInterval=values[Tps.Bits.GuardInterval],
        #     TransmissionMode=values[Tps.Bits.TransmissionMode],
        #     CellID=values[Tps.Bits.CellID],
        #     AnexF=values[Tps.Bits.AnexF],
        #     ErrorProtection=values[Tps.Bits.ErrorProtection],
        # )

    @classmethod
    def _extract_bit_data(cls, carrier_data: np.ndarray[[0, 6817], np.complex128]) -> np.ndarray[[0], np.uint8]:
        """
        Extract bit data from one of the TPS carriers
        """

        tps_ind = 34  # index of one of the TPS carriers - arbitrary choice

        tps_carrier = carrier_data[:, tps_ind]
        tps_carrier_norm = tps_carrier / np.absolute(tps_carrier)
        diffs = tps_carrier[1:] * np.conj(tps_carrier_norm[0:-1])
        bit_data = diffs < 0

        return bit_data.astype("uint8")

    @classmethod
    def _find_frame_start(cls, bit_data: np.ndarray[[0], np.uint8]):
        """
        Look for synchronisation word in bit data to find symbol number of the
        start of a frame
        """

        sync_pattern = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0])
        assert len(bit_data) > len(sync_pattern)

        matched_filter = np.empty(len(bit_data) - len(sync_pattern))

        for k in range(len(matched_filter)):
            matched_filter[k] = np.sum(bit_data[k : k + len(sync_pattern)] ^ sync_pattern)

        return np.argmax(matched_filter)
