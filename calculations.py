from scipy.fft import fft, ifft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm

from reader import getData


def slidingWindow(data, s=2.56,hz=50, overlap=1.28):
    ms = s*1000 
    mso = overlap*1000
    width=s*hz #FREQUENCY
    
    items = []
    newdata = {}
    # currtime = data[0]['time']

    starttime = data[0]['time']
    finaltime = max([x['time'] for x in data])
    while starttime+ms<finaltime:
        for item in data:
            # if item['time'] < starttime:

            if item['time']<starttime:
                continue
            if item['time'] > starttime+ms:
                newstarttime = item['time']
                # print(f'break st{starttime} lni{len(items)}')
                break

            items.append(item)


        if starttime in newdata:
            print('overwriting')
        newdata[starttime] = items
        starttime = newstarttime-mso
        items = []
        # starttime=starttime+ms
        print(f'{starttime-finaltime} remains    \t\t\t', end='\r')
    # while currtime+ms<=finaltime:
    #     for startindex in range(len(data)):
    #         if data[startindex]['time'] >= currtime:
    #             break

    #     # for endindex in range(startindex,len(data)):
    #     #     if data[endindex]['time'] >= currtime+ms:
    #     #         break
    #     endindex = startindex+width
        
    #     items.append(data[startindex:endindex])
    #     currtime = currtime+ms
    #     print(f'{currtime-finaltime} remains\t\t\t',end='\r')

    print()
    sum = 0 
    count=0   
    for start,item in newdata.copy().items():
        if len(item) not in range(120,140):
            del newdata[start]
            continue
        sum += len(item)
        count+=1
        print(len(item),end = ', ')
    print('\n\nsum is',sum)
    print('count is',count)
    print('data is',len(data))
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
lin_acc,grav = butter_lowpass_filter(x, 0.3,25, order=4)#3rd order low pass butterworth (check nyq)


# time_seconds = times.astype('datetime64[s]').astype('int64')
# lin_jerk = np.gradient(lin_acc,time_seconds) #time is assumed to be the index


mag = np.linalg.norm(x)
fastfouriered = fft(x)
