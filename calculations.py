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
        
    

def showGraph(oneWindow):
    # for entry in oneWindow.items():
    XA = oneWindow['XA']
    YA = oneWindow['YA']
    ZA = oneWindow['ZA']


    XR = oneWindow['XR']
    YR = oneWindow['YR']
    ZR = oneWindow['ZR']
    time = [1/float(50) * i for i in range(len(signal))]
    plt.plot(time, XA, label='XA')
    plt.plot(time, YA, label='YA')
    plt.plot(time, ZA, label='ZA')


    # visualize_signal()
        

def visualize_signal(signal, x_labels, y_labels, title, legend):
    # Inputs: signal: 1D column
    #         x_labels: the X axis info (figure)
    #         y_labels: the Y axis info (figure)
    #         title: figure's title
    #         legend : figure's legend

    # Define the figure's dimensions
    plt.figure(figsize=(20, 4))

    # convert row numbers in time durations
    time = [1/float(50) * i for i in range(len(signal))]

    # plotting the signal
    plt.plot(time, signal, label=legend)  # plot the signal and add the legend

    plt.xlabel(x_labels)  # set the label of x axis in the figure
    plt.ylabel(y_labels)  # set the label of y axis in the figure
    plt.title(title)  # set the title of the figure
    plt.legend(loc="upper left")  # set the legend in the upper left corner
    plt.show()  # show the figure

def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    z = signal.filtfilt(a,b, data)
    return y,z


# def getKOLIMZ(windowed):
#     for window in windowed:
        # for col in window:
            # yield col


def applyPreFilters(windowed):
    for window in windowed:
        for colKey,col in window.items():
            col = signal.medfilt(col, kernel_size=3)
            col = butter_lowpass_filter(col, 20, 25, order=3)
          # median filter

        # 3rd order low pass butterworth (check nyq)
        totalAccel = [col for key, col in windowed.items() if 'A' in key]
        lin_acc = butter_lowpass(totalAccel, 0.3, 25, order=4)
        gravity = totalAccel-lin_acc

        lin_jerk = np.gradient(lin_acc, 0.02)

        
        # time_seconds = times.astype('datetime64[s]').astype('int64')
         # time is assumed to be the index

        mag = np.linalg.norm(x)
        fastfouriered = fft(x)


    # x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])

    





if __name__ == '__main__':

    slid = slidingWindow(getData())

    
