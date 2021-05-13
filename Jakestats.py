import statistics, math

from collections import defaultdict

import pandas as pd, numpy as np

import scipy

import scipy.stats

import calculations

def sma(mag_column):
    array=np.array(mag_column)
    return float(abs(array).sum()) #signal magnitude area of one mag column

def energy(mag_column):
    array=np.array(mag_column)
    return float((array**2).sum()) # energy of the mag signal

# arburg: auto regression coefficients using the burg method
def arCoeff(mag_column):
    
    array = np.array(mag_column)
    return list(_arburg2(array,4)[0][1:].real) # AR1, AR2, AR3, AR4 of the mag column

filtered = calculations.getPreFilteredData()
data = []
for activity, windows in filtered.items():
    data.extend(windows)

feature_vector = pd.DataFrame()

sample_frequencies = scipy.fft.fftfreq(128, d=0.02)

for window in data:
    feature_dict = defaultdict(list)
    collist = list(window.keys())[2:] #Ignores the time and label columns
    for column in collist:
        signalvals = window[column]
        meancol = column + "-mean()"
        feature_dict[meancol] = (statistics.mean(signalvals))
        mincol = column + "-min()"
        feature_dict[mincol] = (min(signalvals))
        maxcol = column + "-max()"
        feature_dict[maxcol] = (max(signalvals))
        stdcol = column + "-std()"
        feature_dict[stdcol] = statistics.stdev(signalvals)
        entcol = column + "-entropy()"
        feature_dict[entcol] = scipy.stats.entropy(abs(signalvals))
        madcol = column + "-mad()"
        feature_dict[madcol] = scipy.stats.median_abs_deviation(signalvals)
        iqrcol = column + "-iqr()"
        feature_dict[iqrcol] = scipy.stats.iqr(signalvals)
        enercol = column + "-energy()"
        feature_dict[enercol] = energy(signalvals)
        if "Mag" in column:
            smacol = column[:-3] + "-sma()" #Removes last 2 characters
            smaval = sma(signalvals) / len(signalvals)
            feature_dict[smacol] = smaval
        if column[0] == "f": #For frequency domain
            maxindscol = column + "-maxInds"
            feature_dict[maxindscol] = sample_frequencies[signalvals.argmax()+1]
            skewcol = column + "-skewness"
            feature_dict[skewcol] = scipy.stats.skew(signalvals)
            kurtosiscol = column + "-kurtosis"
            feature_dict[kurtosiscol] = scipy.stats.kurtosis(signalvals)
        #if column[0] == "t":
            #for i in range(1, 5):
                #print(i)
            #arcocol = column + "-arCoeff()"
            #feature_dict[arcocol] = arCoeff(signalvals)
    feature_dict["Label"] = window["label"].mode()[0]

    feature_vector = feature_vector.append(feature_dict, ignore_index = True)

feature_vector.to_csv('all data.csv') #Saves it to a csv file

print("")