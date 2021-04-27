from scipy.fft import fft, ifft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from reader import finalLog


def slidingWindow(data, s=2.56, overlap=0):
    ms = s*1000
    
    items = []
    currtime = data[0]['time']
    starttime = data[0]['time']
    while currtime-starttime<ms:
        nextItem = data.pop(0)
        currtime = nextItem['time']
        items.append(nextItem)
    print(len(items))




def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y


slidingWindow(finalLog)

x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])

print(y)

x = signal.medfilt(x)#median filter
noised, denoised = butter_lowpass_filter(x, 20,25, order=3) #3rd order low pass butterworth (check nyq)
lin_acc,grav = butter_lowpass_filter(x, 0.3,25, order=3)#3rd order low pass butterworth (check nyq)


# time_seconds = times.astype('datetime64[s]').astype('int64')
# lin_jerk = np.gradient(lin_acc,time_seconds) #time is assumed to be the index


mag = np.linalg.norm(x)
fastfouriered = fft(x)
