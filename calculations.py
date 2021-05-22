import scipy
if scipy.__version__ == '1.1.0':
    from scipy.fftpack import fft, ifft
else:
    from scipy.fft import fft, ifft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import pickle
import os
import random
import math


from reader import getData

"""
This is the sliding window funciton, where we get data between two timestamps (any timesamp t and t+2.56 seconds), and returning the list of entries as a window.
Then we increment by 1.28seconds and repeat, until we have iterated on all data. This is really important and discussed in the report.
We also cut windows which were not of size between 120 and 140, as these would probably be outliers and there would be some error in the samping rate of data.
"""


def slidingWindow(data, s=2.56, hz=50, overlap=1.28):
    ms = s*1000
    mso = overlap*1000
    width = s*hz  # FREQUENCY

    items = []
    newdata = {}
    # currtime = data[0]['time']

    starttime = data[0]['time']
    finaltime = max([x['time'] for x in data])
    while starttime+ms < finaltime:
        for item in data:
            # if item['time'] < starttime:

            if item['time'] < starttime:
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

    print()
    sum = 0
    count = 0
    for start, item in newdata.copy().items():
        if len(item) not in range(120, 140):
            del newdata[start]
            continue
        sum += len(item)
        count += 1
        print(len(item), end=', ')
    print('\n\nsum is', sum)
    print('count is', count)
    print('data is', len(data))
    return newdata


"""
Plotting Function
This function gives us an idea of 
"""
def showGraph(window):
    # XA = oneWindow['tAcc-X']
    # YA = oneWindow['tAcc-Y']
    # ZA = oneWindow['tAcc-Z']

    # time = [1/float(50) * i for i in range(len(oneWindow))]
    # plt.plot(time, XA, label='XA',color='red')
    # plt.plot(time, YA, label='YA',color='green')
    # plt.plot(time, ZA, label='ZA', color='blue')
    time = [1/float(50) * i for i in range(len(window))]

    plt.plot(time, window['tGravityAcc-X'], label='Gravity-X', color='m')
    plt.plot(time, window['tGravityAcc-Y'], label='Gravity-Y', color='y')
    plt.plot(time, window['tGravityAcc-Z'], label='Gravity-Z', color='c')

    plt.plot(time, window['tBodyAcc-X'], label='BodyAcc-X', color='r')
    plt.plot(time, window['tBodyAcc-Y'], label='BodyAcc-Y', color='g')
    plt.plot(time, window['tBodyAcc-Z'], label='BodyAcc-Z', color='b')
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.xlabel("Time in seconds (s)")
    plt.ylabel("Acceleration (m/s$^{2}$)")
    plt.legend()
    plt.title(window['label'][0])
    plt.savefig('figs/'+window['label'][0])
    plt.show()

"""
Butterworth seperation filter helper functions
"""
def real_butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = real_butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    # z = signal.filtfilt(a,b, data)
    return y

"""
Magnitude helper function
"""
def mag_3_signals(x, y, z):  # magnitude function redefintion
    return np.array([math.sqrt((x[i]**2+y[i]**2+z[i]**2)) for i in range(len(x))])

"""
FFT helper function
"""
def applyFFT(signal):
    signal = np.asarray(signal)
    return np.abs(fft(signal))


"""
Iterates over all windows, and applies several filters and column extraction on them.
Another essential function, which expands all the columns required for the next steps
"""
def applyPreFilters(data):
    # windowed = [{'A':list(data.values())[600]}]
    for activity, windowed in data.items():
        for window in tqdm(windowed, desc=f'Pre Filtering {activity}'):

            # newcols = []

            for colKey, col in window.items():
                # colnew = {}

                if 'tAcc' not in colKey and 'tBodyGyro' not in colKey:
                    continue

                window[colKey] = signal.medfilt(window[colKey], kernel_size=3)
                window[colKey] = butter_lowpass_filter(
                    window[colKey], 20, 25, order=3)

                if 'tAcc-' in colKey:
                    gcol = 'tGravityAcc-'+colKey[-1]
                    lcol = 'tBodyAcc-'+colKey[-1]
                    window[gcol] = butter_lowpass_filter(
                        window[colKey], 0.3, 25, order=4)
                    window[lcol] = [c-l for c,
                                    l in zip(window[colKey], window[gcol])]

            window['tBodyAccJerk-X'] = np.gradient(window['tBodyAcc-X'], 0.02)
            window['tBodyAccJerk-Y'] = np.gradient(window['tBodyAcc-Y'], 0.02)
            window['tBodyAccJerk-Z'] = np.gradient(window['tBodyAcc-Z'], 0.02)
            window['tBodyGyroJerk-X'] = np.gradient(
                window['tBodyGyro-X'], 0.02)
            window['tBodyGyroJerk-Y'] = np.gradient(
                window['tBodyGyro-Y'], 0.02)
            window['tBodyGyroJerk-Z'] = np.gradient(
                window['tBodyGyro-Z'], 0.02)

            window['tBodyAccMag'] = mag_3_signals(
                window['tBodyAcc-X'],       window['tBodyAcc-Y'],        window['tBodyAcc-Z'])
            window['tGravityAccMag'] = mag_3_signals(
                window['tGravityAcc-X'],    window['tGravityAcc-Y'],     window['tGravityAcc-Z'])
            window['tBodyAccJerkMag'] = mag_3_signals(
                window['tBodyAccJerk-X'],   window['tBodyAccJerk-Y'],    window['tBodyAccJerk-Z'])
            window['tBodyGyroMag'] = mag_3_signals(
                window['tBodyGyro-X'],      window['tBodyGyro-Y'],       window['tBodyGyro-Z'])
            window['tBodyGyroJerkMag'] = mag_3_signals(
                window['tBodyGyroJerk-X'],  window['tBodyGyroJerk-Y'],   window['tBodyGyroJerk-Z'])

            window['fBodyAccMag'] = applyFFT(window['tBodyAccMag'])
            window['fBodyGyroMag'] = applyFFT(window['tBodyGyroMag'])
            window['fBodyAccJerkMag'] = applyFFT(window['tBodyAccJerkMag'])
            window['fBodyGyroJerkMag'] = applyFFT(window['tBodyGyroJerkMag'])

            window['fBodyAcc-X'] = applyFFT(window['tBodyAcc-X'])
            window['fBodyAcc-Y'] = applyFFT(window['tBodyAcc-Y'])
            window['fBodyAcc-Z'] = applyFFT(window['tBodyAcc-Z'])

            window['fBodyAccJerk-X'] = applyFFT(window['tBodyAccJerk-X'])
            window['fBodyAccJerk-Y'] = applyFFT(window['tBodyAccJerk-Y'])
            window['fBodyAccJerk-Z'] = applyFFT(window['tBodyAccJerk-Z'])

            window['fBodyGyro-X'] = applyFFT(window['tBodyGyro-X'])
            window['fBodyGyro-Y'] = applyFFT(window['tBodyGyro-Y'])
            window['fBodyGyro-Z'] = applyFFT(window['tBodyGyro-Z'])

"""
gets the raw data from the reader class and stores as panda data frames, to ready to be preprocessed
"""


def get_data():
    path = os.path.join('intermediaries', 'interm.pkl')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # return data
    else:
        data = {}
        for key, val in getData().items():
            data[key] = []
            print('Sliding over:', key)
            for stime, window in slidingWindow(val).items():
                data[key].append(window)
        with open(path, 'wb') as f:
            pickle.dump(data, f,  protocol=pickle.HIGHEST_PROTOCOL)

    pandasedData = {}
    # pandasedData = pd.DataFrame(data)
    for key, windows in tqdm(data.items(), desc='Pandafying'):
        pandasedData[key] = []
        for window in windows:
            pandasedData[key].append(pd.DataFrame(window))

    return pandasedData


"""
Checks for intermedairy files
If do not exist, regenerates all data nd saves the data

Returns data ready to be feature extracted
"""


def getPreFilteredData():
    path = os.path.join('intermediaries', 'filt_interm.pkl')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    data = get_data()
    applyPreFilters(data)
    with open(path, 'wb') as f:
        pickle.dump(data, f,  protocol=pickle.HIGHEST_PROTOCOL)
    return data


if __name__ == '__main__':

    data = getPreFilteredData()
    print(data[list(data.keys())[0]][0].head())
    print(list(data[list(data.keys())[0]][0].columns))

    for activity, windows in data.items():
        if 'walk' not in activity.lower():
            continue
        for window in windows:
            showGraph(window)
