from scipy.fft import fft, ifft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm

from reader import getData


def slidingWindow(data, s=2.56,hz=50, overlap=0):
    ms = s*1000
    width=s*hz #FREQUENCY
    
    items = []
    currtime = data[0]['time']
    starttime = data[0]['time']
    finaltime = data[-1]['time']
    while currtime+ms<=finaltime:
        for startindex in range(len(data)):
            if data[startindex]['time'] >= currtime:
                break

        # for endindex in range(startindex,len(data)):
        #     if data[endindex]['time'] >= currtime+ms:
        #         break
        endindex = startindex+width
        
        items.append(data[startindex:endindex])
        currtime = currtime+ms
        print(f'{currtime-finaltime} remains\t\t\t',end='\r')

    print()    
    for item in items:
        print(len(item),end = ', ')
    return items
        
    





def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y


slidingWindow(getData())

x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])

print(y)

x = signal.medfilt(x)#median filter
noised, denoised = butter_lowpass_filter(x, 20,25, order=3) #3rd order low pass butterworth (check nyq)
lin_acc,grav = butter_lowpass_filter(x, 0.3,25, order=3)#3rd order low pass butterworth (check nyq)


# time_seconds = times.astype('datetime64[s]').astype('int64')
# lin_jerk = np.gradient(lin_acc,time_seconds) #time is assumed to be the index


mag = np.linalg.norm(x)
fastfouriered = fft(x)
