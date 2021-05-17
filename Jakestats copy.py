import statistics, math

from collections import defaultdict

import pandas as pd, numpy as np

import scipy

import scipy.stats

import calculations

def sma(mag_column):
    array=np.array(mag_column)
    return float(abs(array).sum()) #signal magnitude area of one mag column

def energy(column):
    array=np.array(column)
    return float((array**2).sum()) # energy of the signal

# arburg: auto regression coefficients using the burg method
def arCoeff(mag_column):
    
    array = np.array(mag_column)
    return list(_arburg2(array,4)[0][1:].real) # AR1, AR2, AR3, AR4 of the mag column

def f_one_band_energy(signal, band):
    f_signal_bounded = signal[band[0]:band[1]] # select f_signal components included in the band
    energy_value = energy(f_signal_bounded) / float(len(f_signal_bounded)) # energy value of that band
    return energy_value

def bands_energy(signal):
    energy_band_1 = [(1,9),(9,17),(17,25),(25,33),(33,41),(41,49),(49,57),(57,65)] 
    energy_band_2 = [(1,17),(17,31),(31,49),(49,65)]
    energy_band_3 = [(1,25),(25,49)]

    total_axis = []
    signal = np.array(signal)
    for i in range(0,3):
        E1=[ f_one_band_energy( signal,( energy_band_1 [j][0], energy_band_1 [j][1]) ) for j in range(len(energy_band_1))] # energy bands1 values of f_signal
        E2=[ f_one_band_energy( signal,( energy_band_2 [j][0], energy_band_2 [j][1]) ) for j in range(len(energy_band_2))]# energy bands2 values of f_signal
        E3=[ f_one_band_energy( signal,( energy_band_3 [j][0], energy_band_3 [j][1]) ) for j in range(len(energy_band_3))]# energy bands3 values of f_signal
        malla = E1+E2+E3

        total_axis += malla
        print("")

def angle(vector1, vector2):
    """Calculates angle between 2 vectors"""

    vector1_mag = math.sqrt((vector1**2).sum()) # euclidian norm of V1
    vector2_mag = math.sqrt((vector2**2).sum()) # euclidian norm of V2
   
    scalar_product = np.dot(vector1, vector2) #Scalar product of vector 1 and Vector 2
    cos_angle = scalar_product/float(vector1_mag*vector2_mag) # the cosine value of the angle between V1 and V2
    
    #Using this in case some values were added automatically
    if cos_angle>1:
        cos_angle=1
    elif cos_angle<-1:
        cos_angle=-1
    
    angle_value=float(math.acos(cos_angle))

    return angle_value #Radians

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
            sample_frequencies = np.absolute(sample_frequencies)
            feature_dict[meanfreqcol] = np.ma.average(signalvals, axis=0, weights=sample_frequencies)
            if math.isinf(feature_dict[meanfreqcol]):
                print("Meanfreq is inf for window size", len(window))

        #bandssingledf = [window['fBodyAcc-X'], window['fBodyAcc-Y'], window['fBodyAcc-Z']]
        #bandsenergy = bands_energy(bandssingledf)

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

    angles = []
    V2_columns=['tGravityAcc-X','tGravityAcc-Y','tGravityAcc-Z']
    V2_Vector=np.array(window[V2_columns].mean())

    V1_columns=['tBodyAcc-X','tBodyAcc-Y','tBodyAcc-Z']
    V1_Vector=np.array(window[V1_columns].mean())
    feature_dict['angle(tBodyAccMean, gravityMean)'] = angle(V1_Vector, V2_Vector)

    V1_columns=['tBodyAccJerk-X','tBodyAccJerk-Y','tBodyAccJerk-Z']
    V1_Vector=np.array(window[V1_columns].mean())
    feature_dict['angle(tBodyAccJerkMean, gravityMean)'] = angle(V1_Vector, V2_Vector)

    V1_columns=['tBodyGyro-X','tBodyGyro-Y','tBodyGyro-Z']
    V1_Vector=np.array(window[V1_columns].mean())
    feature_dict['angle(tBodyGyroMean, gravityMean)'] = angle(V1_Vector, V2_Vector)

    V1_columns=['tBodyGyroJerk-X','tBodyGyroJerk-Y','tBodyGyroJerk-Z']
    V1_Vector=np.array(window[V1_columns].mean())
    feature_dict['angle(tBodyGyroJerkMean, gravityMean)'] = angle(V1_Vector, V2_Vector)

    V1_Vector=np.array([1,0,0]) #X-Axis
    feature_dict['angle(X-axis, gravityMean'] = angle(V1_Vector, V2_Vector)

    V1_Vector=np.array([0, 1, 0]) #Y-Axis
    feature_dict['angle(Y-axis, gravityMean'] = angle(V1_Vector, V2_Vector)

    V1_Vector=np.array([0, 0, 1]) #Z-Axis
    feature_dict['angle(Z-axis, gravityMean'] = angle(V1_Vector, V2_Vector)

    feature_dict["Label"] = window["label"].mode()[0] #Activity Label

    feature_vector = feature_vector.append(feature_dict, ignore_index = True)

feature_vector.to_csv('all data.csv', index = False) #Saves it to a csv file

print("")