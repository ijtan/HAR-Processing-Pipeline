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


for window in data:
    feature_dict = defaultdict(list)
    collist = list(window.keys())[2:] #Ignores the time and label columns
    sample_frequencies = scipy.fft.fftfreq(len(window), d=0.02) #Changed it to length of window so meanFreq can be computed
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
            meanfreqcol = column + "-meanFreq"
            othermeanfreq = np.average(signalvals, axis=0, weights=sample_frequencies) / float(signalvals.sum())
            feature_dict[meanfreqcol] = np.dot(sample_frequencies, signalvals).sum() / float(signalvals.sum()) #Very small values


        #if column[0] == "t":
        #    for i in range(1, 5):
        #        print(i)
        #    arcocol = column + "-arCoeff()"
        #    feature_dict[arcocol] = arCoeff(signalvals)
    #vector1 = window['', '', '']

    feature_dict['tBodyAcc-correlation-XY'] = float(scipy.stats.pearsonr(window['tBodyAcc-X'], window['tBodyAcc-Y'])[0])
    feature_dict['tBodyAcc-correlation-XZ'] = float(scipy.stats.pearsonr(window['tBodyAcc-X'], window['tBodyAcc-Z'])[0])
    feature_dict['tBodyAcc-correlation-YZ'] = float(scipy.stats.pearsonr(window['tBodyAcc-Y'], window['tBodyAcc-Z'])[0])

    feature_dict['tGravityAcc-correlation-XY'] = float(scipy.stats.pearsonr(window['tGravityAcc-X'], window['tGravityAcc-Y'])[0])
    feature_dict['tGravityAcc-correlation-XZ'] = float(scipy.stats.pearsonr(window['tGravityAcc-X'], window['tGravityAcc-Z'])[0])
    feature_dict['tGravityAcc-correlation-YZ'] = float(scipy.stats.pearsonr(window['tGravityAcc-Y'], window['tGravityAcc-Z'])[0])

    feature_dict['tBodyAccJerk-correlation-XY'] = float(scipy.stats.pearsonr(window['tBodyAccJerk-X'], window['tBodyAccJerk-Y'])[0])
    feature_dict['tBodyAccJerk-correlation-XZ'] = float(scipy.stats.pearsonr(window['tBodyAccJerk-X'], window['tBodyAccJerk-Z'])[0])
    feature_dict['tBodyAccJerk-correlation-YZ'] = float(scipy.stats.pearsonr(window['tBodyAccJerk-Y'], window['tBodyAccJerk-Z'])[0])

    feature_dict['tBodyGyro-correlation-XY'] = float(scipy.stats.pearsonr(window['tBodyGyro-X'], window['tBodyGyro-Y'])[0])
    feature_dict['tBodyGyro-correlation-XZ'] = float(scipy.stats.pearsonr(window['tBodyGyro-X'], window['tBodyGyro-Z'])[0])
    feature_dict['tBodyGyro-correlation-YZ'] = float(scipy.stats.pearsonr(window['tBodyGyro-Y'], window['tBodyGyro-Z'])[0])

    feature_dict['tBodyGyroJerk-correlation-XY'] = float(scipy.stats.pearsonr(window['tBodyGyroJerk-X'], window['tBodyGyroJerk-Y'])[0])
    feature_dict['tBodyGyroJerk-correlation-XZ'] = float(scipy.stats.pearsonr(window['tBodyGyroJerk-X'], window['tBodyGyroJerk-Z'])[0])
    feature_dict['tBodyGyroJerk-correlation-YZ'] = float(scipy.stats.pearsonr(window['tBodyGyroJerk-Y'], window['tBodyGyroJerk-Z'])[0])

    feature_dict["Label"] = window["label"].mode()[0] #Activity Label

    feature_vector = feature_vector.append(feature_dict, ignore_index = True)

feature_vector.to_csv('all data.csv', index = False) #Saves it to a csv file

print("")