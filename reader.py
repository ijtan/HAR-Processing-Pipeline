import os
import json
import pathlib
import numpy as np
from tqdm import tqdm

last_folder = ""
session_entries = {}

"""
File Reading function. 
In here, we iterate over all files and directories, ready any json files,
cut the first and last N (set to 2) seconds in distinct sessions
We store the entires in a dic with the gyr and acc respectively, sorted by timestamp
We extact the labels from the foldername, and add it as a column

"""
def read_all(path, entries, start_trim=2, end_trim=2, sample_rate=50):
    global last_folder
    global session_entries

    for log in pathlib.Path(path).iterdir():

        if not log.is_file():
            # print('PATH IS NOT FILE!!')
            read_all(log, entries)
            continue

        if '.json' not in str(log):
            print('not a json; skipping...')
            continue

        start_cut = (sample_rate)*start_trim
        end_cut = (sample_rate)*end_trim

        label = str(path).split('DATA_')[-1].split('_SESSION')[0]
        folder = str(path)

        with open(log, 'r', encoding="utf8") as f:
            file = json.load(f)

        for f in file:
            f.update({'lbl': label})
        for f in file:
            f.update({'folder_name': folder})

        sep = '/'
        if '\\' in str(log):
            sep = '\\'
        curr_folder = str(log).split(sep)[-2]

        if curr_folder != last_folder:

            if last_folder != "":

                session_entries['acc'] = sorted(
                    session_entries['acc'], key=lambda item: item['time'])
                session_entries['gyr'] = sorted(
                    session_entries['gyr'], key=lambda item: item['time'])

                session_entries['acc'] = session_entries['acc'][start_cut:-end_cut]
                session_entries['gyr'] = session_entries['gyr'][start_cut:-end_cut]

                entries['acc'].extend(session_entries['acc'])
                entries['gyr'].extend(session_entries['gyr'])

            session_entries['acc'] = []
            session_entries['gyr'] = []

        if 'ACC' in str(log) or '_a' in str(log):
            session_entries['acc'].extend(file)

        elif 'GYR' in str(log) or '_g' in str(log):
            session_entries['gyr'].extend(file)

        last_folder = curr_folder

"""
This function finds the amount of entries with the same index, that have a timestamp difference of more than 1 and 10 milliseconds
Ideally entries of same index of the gyr and acc lists should have the same timestamp, yet due to a limitation from the app, this was unfortunately not the case,
and these measures were put in place to try to counter act this issue
"""
def lenDiff(raw_entries):
    count = 0
    countM = 0

    for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
        # print('iter')
        if abs(acc['time']-gyr['time']) > 0:
            count += 1
        if abs(acc['time']-gyr['time']) > 10:
            countM += 1

        # if abs(acc['time']-gyr['time']) > 10:
            # print(f'out of sync file found: {acc["folder_name"]}')

    print(f'\n{count} items out of sync')
    print(f'\n{countM} items out of sync by atleast 10')
    return count


"""
This function attempts to sync the indices of the respective lists
This is done by computing a moving average over every 4 entries, and if the average deviation is higher than 17(found to be ideal number through several days of testing),
we delete an entry from the start of each list, until the average stabilizes. This is in-place to handle mismatches in lengths of sessions, as discrepancies are detected and deleted.  
"""
def sync2(logs, x=4, avg_diff=17):

    for ac in range(0, len(logs['acc'])):
        # print('avg_diff:', {sum([a['time']-g['time'] for a, g in zip(logs['acc'][ac:ac+x], logs['gyr'][ac:ac+x])])/x})
        while abs(sum([a['time']-g['time'] for a, g in zip(logs['acc'][ac:ac+x], logs['gyr'][ac:ac+x])])/x) >= avg_diff:
            lastx = [a['time']-g['time']
                     for a, g in zip(logs['acc'][ac:ac+x], logs['gyr'][ac:ac+x])]
            avg = sum(lastx)/x
            # print(f'len before: {len(logs["gyr"])}')
            if avg >= avg_diff:
                # print(f'len before: {len(logs["gyr"])}')
                del logs['gyr'][ac]
                # print(f'len after: {len(logs["gyr"])}')
            elif avg <= -avg_diff:
                del logs['acc'][ac]
            # print(f'len before: {len(logs["gyr"]}')


"""
This function reads, sorts and synchronises the data, and check for a final time for discrepancies, removing any final discrepancies which were not able to be removed after 
several passes of synchronosation.
"""
def getRaw():
    raw_entries = {'acc': [], 'gyr': []}
    read_all('INPUT', raw_entries)

    raw_entries['acc'] = sorted(
        raw_entries['acc'], key=lambda item: item['time'])
    raw_entries['gyr'] = sorted(
        raw_entries['gyr'], key=lambda item: item['time'])

    sync2(raw_entries)

    count = 0
    last = -1
    new = 1
    while new > 0 and new != last:
        print('removing mismatches!')
        for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
            if abs(acc['time']-gyr['time']) <= 10:
                acc['time'] = gyr['time']

        last = new
        new = lenDiff(raw_entries)
    count = 0
    print(f'len before: {len(raw_entries["acc"])}')
    for (ak, acc), (gk, gyr) in zip(enumerate(raw_entries['acc']), enumerate(raw_entries['gyr'])):
        # print('removing', raw_entries['acc'].index(acc))
        if abs(acc['time']-gyr['time']) > 0:
            count += 1
            del raw_entries['acc'][ak]
            del raw_entries['gyr'][gk]
            # print(f'deleting major out of sync entry found in: {acc["folder_name"]}')
    print(f'len before: {len(raw_entries["acc"])}')
    print(f'removed {count} entries')

    return raw_entries


"""
This function gets all data from the previous function, and preapres the final dictionaries, labels and formatting to be presentable, and acceptable by other classes.
So other classes can simply call this function, and all previous steps are handled accordingly through it, returning only the final dictionaries. 
"""
def getData():

    raw_entries = getRaw()
    semiFinalLog = []
    for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):

        # print(acc)
        entry = {}
        entry['time'] = acc['time']
        entry['label'] = acc['lbl']

        oldlabel = acc['lbl'].lower()
        newlabel = ''

        if 'jump' in oldlabel:
            newlabel = 'JUMPING'
        elif 'push' in oldlabel:
            newlabel = 'PUSH UP'
        elif 'swim' in oldlabel:
            newlabel = 'SWIMMING'
        elif 'up' in oldlabel:
            newlabel = 'STAIRS UP'
        elif 'down' in oldlabel:
            newlabel = 'STAIRS DN'
        elif 'walk' in oldlabel:
            newlabel = 'WALKING'
        elif 'stand' in oldlabel:
            newlabel = 'STANDING'
        elif 'driv' in oldlabel:
            newlabel = 'DRIVING'
        elif 'sitt' in oldlabel:
            newlabel = 'SITTING'
        elif 'lay' in oldlabel:
            newlabel = 'LAYING'
        else:
            continue

        # elif 'down' in oldlabel:
        #     newlabel = 'STAIRS down'

        entry['label'] = newlabel

        entry['tAcc-X'] = acc['XA']
        entry['tAcc-Y'] = acc['YA']
        entry['tAcc-Z'] = acc['ZA']

        entry['tBodyGyro-X'] = gyr['XR']
        entry['tBodyGyro-Y'] = gyr['YR']
        entry['tBodyGyro-Z'] = gyr['ZR']

        semiFinalLog.append(entry)

    finalLog = {}
    for entry in semiFinalLog:
        if entry['label'] not in finalLog:
            finalLog[entry['label']] = []
        finalLog[entry['label']].append(entry)
    return finalLog
    # finalLog = np.asarray(semiFinalLog)
    # print(semiFinalLog[0])
    # return semiFinalLog


if __name__ == '__main__':
    getData()
