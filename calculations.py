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
import pickle,os,random,math


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
        
    

def showGraph(window):
    # XA = oneWindow['tAcc-X']
    # YA = oneWindow['tAcc-Y']
    # ZA = oneWindow['tAcc-Z']

    # time = [1/float(50) * i for i in range(len(oneWindow))]
    # plt.plot(time, XA, label='XA',color='red')
    # plt.plot(time, YA, label='YA',color='green')
    # plt.plot(time, ZA, label='ZA', color='blue')
    time = [1/float(50) * i for i in range(len(window))]
    # if plot and 'driv' not in window['label'][0].lower():
                # print(window.head())
    # showGraph(window)

    plt.plot(time, window['tGravityAcc-X'],label='xgravity', color='red')
    plt.plot(time, window['tGravityAcc-Y'], label='ygravity', color='green')
    plt.plot(time, window['tGravityAcc-Z'], label='zgravity', color='blue')
    
    plt.plot(time, window['tBodyAcc-X'], label='xlin', color='yellow')
    plt.plot(time, window['tBodyAcc-Y'], label='ylin', color='brown')
    plt.plot(time, window['tBodyAcc-Z'], label='zlin', color='pink')

    plt.title(window['label'][0])

    plt.show()
        


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

def mag_3_signals(x, y, z):  # magnitude function redefintion
    return np.array([math.sqrt((x[i]**2+y[i]**2+z[i]**2)) for i in range(len(x))])

def mag_3_signals(x, y, z):  # magnitude function redefintion
    return np.array([math.sqrt((x[i]**2+y[i]**2+z[i]**2)) for i in range(len(x))])

def applyFFT(signal):
    signal = np.asarray(signal)
    return np.abs(fft(signal))

def applyPreFilters(windowed,plot=False):
    # windowed = [{'A':list(data.values())[600]}]
    
    random.shuffle(windowed)
    for window in tqdm(windowed, desc='Pre Filtering'):
        time = [1/float(50) * i for i in range(len(window))]

        # newcols = []

def applyFFT(signal):
    signal = np.asarray(signal)
    return np.abs(fft(signal))

def applyPreFilters(data):
    # windowed = [{'A':list(data.values())[600]}]
    for activity, windowed in data.items():        
        for window in tqdm(windowed, desc=f'Pre Filtering {activity}'):
            

            # newcols = []

            for colKey,col in window.items():
                # colnew = {}

                if 'tAcc' not in colKey and 'tBodyGyro' not in colKey:
                    continue

                window[colKey] = signal.medfilt(window[colKey], kernel_size=3)
                window[colKey] = butter_lowpass_filter(window[colKey], 20, 25, order=3)

                if 'tAcc-' in colKey:
                    gcol = 'tGravityAcc-'+colKey[-1]
                    lcol = 'tBodyAcc-'+colKey[-1]
                    window[gcol] = butter_lowpass_filter(window[colKey], 0.3, 25, order=4)
                    window[lcol] = [c-l for c, l in zip(window[colKey], window[gcol])]
            
            window['tBodyAccJerk-X'] = np.gradient(window['tBodyAcc-X'], 0.02)
            window['tBodyAccJerk-Y'] = np.gradient(window['tBodyAcc-Y'], 0.02)
            window['tBodyAccJerk-Z'] = np.gradient(window['tBodyAcc-Z'], 0.02)
            window['tBodyGyroJerk-X'] = np.gradient(window['tBodyGyro-X'], 0.02)
            window['tBodyGyroJerk-Y'] = np.gradient(window['tBodyGyro-Y'], 0.02)
            window['tBodyGyroJerk-Z'] = np.gradient(window['tBodyGyro-Z'], 0.02)


            window['tBodyAccMag']       = mag_3_signals(window['tBodyAcc-X'],       window['tBodyAcc-Y'],        window['tBodyAcc-Z'])
            window['tGravityAccMag']    = mag_3_signals(window['tGravityAcc-X'],    window['tGravityAcc-Y'],     window['tGravityAcc-Z'])
            window['tBodyAccJerkMag']   = mag_3_signals(window['tBodyAccJerk-X'],   window['tBodyAccJerk-Y'],    window['tBodyAccJerk-Z'])
            window['tBodyGyroMag']      = mag_3_signals(window['tBodyGyro-X'],      window['tBodyGyro-Y'],       window['tBodyGyro-Z'])
            window['tBodyGyroJerkMag']  = mag_3_signals(window['tBodyGyroJerk-X'],  window['tBodyGyroJerk-Y'],   window['tBodyGyroJerk-Z'])

            window['fBodyAccMag']       =   applyFFT(window['tBodyAccMag']      )
            window['fBodyGyroMag']      =   applyFFT(window['tBodyGyroMag']     )
            window['fBodyAccJerkMag']   =   applyFFT(window['tBodyAccJerkMag']  )
            window['fBodyGyroJerkMag']  =   applyFFT(window['tBodyGyroJerkMag'] )

            window['fBodyAcc-X']        =   applyFFT(window['tBodyAcc-X']       )
            window['fBodyAcc-Y']        =   applyFFT(window['tBodyAcc-Y']       )
            window['fBodyAcc-Z']        =   applyFFT(window['tBodyAcc-Z']       )

            window['fBodyAccJerk-X']    =   applyFFT(window['tBodyAccJerk-X']   )
            window['fBodyAccJerk-Y']    =   applyFFT(window['tBodyAccJerk-Y']   )
            window['fBodyAccJerk-Z']    =   applyFFT(window['tBodyAccJerk-Z']   )
            
            window['fBodyGyro-X']       =   applyFFT(window['tBodyGyro-X']      )
            window['fBodyGyro-Y']       =   applyFFT(window['tBodyGyro-Y']      )

            

    

def get_data():
    path = os.path.join('intermediaries','interm.pkl')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # return data
    else:
        data = {}
        for key, val in getData().items():
            data[key] = []
            print('Sliding over:',key)
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


def getPreFilteredData():
    path = os.path.join('intermediaries','filt_interm.pkl')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


    data = get_data()
    applyPreFilters(data)
    with open(path, 'wb') as f:
            pickle.dump(data,f,  protocol=pickle.HIGHEST_PROTOCOL)
    return data

if __name__ == '__main__':

    data = getPreFilteredData()
    print(data[list(data.keys())[0]][0].head())
    print(list(data[list(data.keys())[0]][0].columns))

    for activity, windows in data.items():
        # if 'driv' in activity.lower():
        #     continue
        for window in windows:
            showGraph(window)
