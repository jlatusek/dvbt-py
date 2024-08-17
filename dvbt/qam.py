import numpy as np
from matplotlib import pyplot as plt


class Qam:
    def __init__(self, m: int = 64):
        points_in_row = int(np.sqrt(m))
        self._m = m
        self._bits = int(np.log2(m))
        self._constellation = np.array(
            [
                complex(x, y)
                for x in range(points_in_row - 1, -points_in_row, -2)
                for y in range(points_in_row - 1, -points_in_row, -2)
            ]
        )
        self._gray_codes = self._assign_gray_code()

    # TODO created function will hava a problem with demodulation of the signal
    # with lots of noice and with values outliers
    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        signal /= np.abs(signal).max()
        plt.figure()
        plt.scatter(self._constellation.real, self._constellation.imag, color="red")
        plt.title("Signal")
        plt.show()
        # Demodulate the signal
        demodulated = np.zeros(len(signal), dtype=int)
        for i, s in enumerate(signal):
            distances = np.abs(s - self._constellation)
            demodulated[i] = np.argmin(distances)

        return demodulated

    def draw_constellation(self):
        plt.figure()
        plt.scatter(self._constellation.real, self._constellation.imag, color="red")
        plt.title("Constellation with indexes")
        limit = np.abs(self._constellation).max()
        plt.xlim([-limit, limit])
        plt.ylim([-limit, limit])
        for i in range(len(self._constellation)):
            gray = self._gray_codes[i]
            point = self._constellation[i]
            plt.annotate("{gray:0{bits}b}".format(gray=gray, bits=self._bits), (point.real, point.imag))
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.axline((0, -limit), (0, limit), color="black", linestyle="--")
        plt.axline((-limit, 0), (limit, 0), color="black", linestyle="--")
        plt.show()

    @property
    def constellation(self) -> np.ndarray[complex]:
        return self._constellation

    def _gen_gray_codes(self, n: int) -> np.ndarray[int]:
        assert n > 0
        if n == 1:
            return np.array([0, 1], dtype=int)
        shorter_gray_codes = self._gen_gray_codes(n - 1)
        bitmask = 1 << (n - 1)
        gray_codes = shorter_gray_codes
        for gray_code in reversed(shorter_gray_codes):
            gray_codes = np.append(gray_codes, bitmask | gray_code)
        return gray_codes

    def _assign_gray_code(self) -> np.ndarray[int]:
        gray_codes = self._gen_gray_codes(int(self._bits / 2))
        assigment_codes = np.zeros(self._m, dtype=int)
        point_in_line = np.sqrt(self._m)
        for i in range(len(self._constellation)):
            real_i = int(i / point_in_line)
            imag_i = int(i % point_in_line)
            real = gray_codes[real_i]
            imag = gray_codes[imag_i]
            for j in range(self._bits):
                even = j % 2 == 0
                # 0 1 3 2
                if even:
                    assigment_codes[i] |= ((imag & (1 << int(j / 2))) >> int(j / 2)) << j
                else:
                    assigment_codes[i] |= ((real & (1 << int(j / 2))) >> int(j / 2)) << j

        return assigment_codes
