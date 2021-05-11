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
        feature_dict[meancol].append(statistics.mean(signalvals))
        mincol = column + "-min()"
        feature_dict[mincol].append(min(signalvals))
        maxcol = column + "-max()"
        feature_dict[maxcol].append(max(signalvals))
        stdcol = column + "-std()"
        feature_dict[stdcol].append(statistics.stdev(signalvals))
        entcol = column + "-entropy()"
        feature_dict[entcol].append(scipy.stats.entropy(abs(signalvals)))
        madcol = column + "-mad()"
        feature_dict[madcol].append(scipy.stats.median_abs_deviation(signalvals))
        iqrcol = column + "-iqr()"
        feature_dict[iqrcol].append(scipy.stats.iqr(signalvals))
        enercol = column + "-energy()"
        feature_dict[enercol].append(energy(signalvals))
        if "Mag" in column:
            smacol = column[:-3] + "-sma()" #Removes last 2 characters
            smaval = sma(signalvals) / len(signalvals)
            feature_dict[smacol] = smaval
        if column[0] == "f": #For frequency domain
            maxindscol = column + "-maxInds"
            maxinds = sample_frequencies[signalvals.argmax()+1]
            feature_dict[maxindscol] = maxinds

        #arcocol = column + "-arCoeff()"
        #feature_dict[arcocol].append(arCoeff(signalvals))
        
    feature_vector = feature_vector.append(feature_dict, ignore_index = True)


print("")