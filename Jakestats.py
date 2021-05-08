import statistics

from collections import defaultdict

import pandas as pd

import calculations

data = calculations.getPreFilteredData()

print(data[0])

feature_vector = pd.DataFrame()

for window in data:
    feature_dict = defaultdict(list)
    for column, signalvals in window.items():
        if column == "time" or column == "label":
            continue
        meancol = column + "-mean()"
        feature_dict[meancol].append(statistics.mean(signalvals))
        mincol = column + "-min()"
        feature_dict[mincol].append(min(signalvals))
        maxcol = column + "-max()"
        feature_dict[maxcol].append(max(signalvals))
        stdcol = column + "-std()"
        feature_dict[stdcol] = statistics.stdev(signalvals)
    feature_vector = feature_vector.append(feature_dict, ignore_index = True)


print("")