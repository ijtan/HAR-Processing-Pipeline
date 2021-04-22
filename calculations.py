from scipy.fft import fft, ifft
import numpy as np
import pandas as pd

x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
y = fft(x)
print(y)