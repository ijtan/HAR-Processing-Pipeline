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
import json
import os

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
    # return newdata
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
    return newdata
        
    

def showGraph(oneWindow,plot=False):
    # for entry in oneWindow.items():
    XA = oneWindow.XA#[w['XA'] for w in oneWindow]
    YA = oneWindow.YA#[w['YA'] for w in oneWindow]
    ZA = oneWindow.ZA  # [w['ZA'] for w in oneWindow]


    # XR = oneWindow['XR']
    # YR = oneWindow['YR']
    # ZR = oneWindow['ZR']

    time = [1/float(50) * i for i in range(len(oneWindow))]
    plt.plot(time, XA, label='XA',color='red')
    plt.plot(time, YA, label='YA',color='green')
    plt.plot(time, ZA, label='ZA', color='blue')
    
    if plot:
        plt.show()

    # visualize_signal()
        


def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    # z = signal.filtfilt(a,b, data)
    return y


# def getKOLIMZ(windowed):
#     for window in windowed:
        # for col in window:
            # yield col


def applyPreFilters(windowed):
    # windowed = [{'A':list(data.values())[600]}]
    


    for window in windowed:
        time = [1/float(50) * i for i in range(len(window))]

        # newcols = []

        for colKey,col in window.items():
            # colnew = {}

            if 'A' not in colKey and 'R' not in colKey:
                continue

            window[colKey] = signal.medfilt(window[colKey], kernel_size=3)
            window[colKey] = butter_lowpass_filter(window[colKey], 20, 25, order=3)

            if 'A' in colKey:
                # lin = {}
                # gr = {}

                gcol = colKey+'G'
                lcol = colKey+'L'
                window[gcol] = butter_lowpass_filter(window[colKey], 0.3, 25, order=4)
                window[lcol] = [c-l for c, l in zip(window[colKey], window[gcol])]

        # newcols.append(colnew)

        # window['XAL'] = [a['XAL'] for a in newcols]
        # window['YAL'] = [a['YAL'] for a in newcols]
        # window['ZAL'] = [a['ZAL'] for a in newcols]

        # window['XAG'] = [a['XAG'] for a in newcols]
        # window['YAG'] = [a['YAG'] for a in newcols]
        # window['ZAG'] = [a['ZAG'] for a in newcols]

        print(window.head())
        

        # showGraph(window)
        # plt.show()
        plt.plot(time, window['XAG'], label='xgravity',color='red')
        plt.plot(time, window['YAG'], label='ygravity', color='green')
        plt.plot(time, window['ZAG'], label='zgravity', color='blue')
        
        plt.plot(time, window['XAL'], label='xlin',color='yellow')
        plt.plot(time, window['YAL'], label='ylin', color='brown')
        plt.plot(time, window['ZAL'], label='zlin', color='pink')

        plt.title(window['lbl'][0])

        plt.show()
        continue
        # return
          # median filter

        # 3rd order low pass butterworth (check nyq)
        # totalAccel = [window[col] for key, col in windowed.items() if 'A' in key]
        

        

        

        # return

        lin_jerk = np.gradient(lin_acc, 0.02)

        

        
        # time_seconds = times.astype('datetime64[s]').astype('int64')
         # time is assumed to be the index

        mag = np.linalg.norm(x)
        fastfouriered = fft(x)


    # x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])

    


def get_data():
    if os.path.isfile('interm.json'):
        infile = open('interm.json', 'r', encoding='utf-8')
        data = json.load(infile)
        # return data
    else:
        data = slidingWindow(getData())
        output_file = open('interm.json', 'w', encoding='utf-8')
        json.dump(data, output_file, indent=4)

    pandasedData = []
    for key,window in tqdm(data.items()):
        pandasedData.append(pd.DataFrame(window))
    
    return pandasedData


if __name__ == '__main__':

    data = get_data()
    print('x')
    showGraph(data[600],plot=True)
    data = applyPreFilters(data)

    
