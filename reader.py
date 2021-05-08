import os
import json
import pathlib
import numpy as np
from tqdm import tqdm


def read_all(path, entries):
    for log in pathlib.Path(path).iterdir():
        

        if not log.is_file():
            # print('PATH IS NOT FILE!!')
            read_all(log, entries)
            continue

        if '.json' not in str(log):
            print('not a json; skipping...')
            continue

        if 'closing' in str(log):
            continue


        label = str(path).split('DATA_')[-1].split('_SESSION')[0]
        
        first =""
        with open(log, 'r', encoding="utf8") as f:
            file = json.load(f)
            for f in file:
                f.update({'lbl': label})
            if 'ACC' in str(log) or '_a' in str(log):

                entries['acc'].extend(file)
            elif 'GYR' in str(log) or '_g' in str(log):
                entries['gyr'].extend(file)
            else:
                ValueError('other file found!')
            # print(file[0])


def lenDiff(raw_entries):
    count = 0

    for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
        # print('iter')
        if abs(acc['time']-gyr['time']) > 0:
            count += 1

    print(f'\n{count} items out of sync')
    return count


def sync2(logs, x=4, avg_diff=17):

    for ac in range(0, len(logs['acc'])):
        # print('avg_diff:', {sum([a['time']-g['time'] for a, g in zip(logs['acc'][ac:ac+x], logs['gyr'][ac:ac+x])])/x})
        while abs(sum([a['time']-g['time'] for a, g in zip(logs['acc'][ac:ac+x], logs['gyr'][ac:ac+x])])/x) >= avg_diff:
            lastx = [a['time']-g['time'] for a, g in zip(logs['acc'][ac:ac+x], logs['gyr'][ac:ac+x])]
            avg = sum(lastx)/x
            
            if avg >= avg_diff:
                del logs['gyr'][ac]
            elif avg <= -avg_diff:
                del logs['acc'][ac]



def getRaw():
    raw_entries = {'acc': [], 'gyr': []}
    read_all('INPUT', raw_entries)

    raw_entries['acc'] = sorted(raw_entries['acc'], key=lambda item: item['time'])
    raw_entries['gyr'] = sorted(raw_entries['gyr'], key=lambda item: item['time'])

    sync2(raw_entries)

    count = 0
    last = -1;
    new  = 1;
    while new > 0 and new!=last:
        print('removing mismatches!')
        for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
            if abs(acc['time']-gyr['time']) <= 10:
                acc['time'] = gyr['time']

        last = new
        new =  lenDiff(raw_entries)
    count=0
    for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
        # print('removing', raw_entries['acc'].index(acc))
        if abs(acc['time']-gyr['time']) > 0:
            count+=1
            raw_entries['acc'].remove(acc)
            raw_entries['gyr'].remove(gyr)
    print(f'removed {count} entries')

        
    return raw_entries

def getData():
    

    raw_entries = getRaw()
    semiFinalLog = []
    for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):

        # print(acc)
        entry = {}
        entry['time'] = acc['time']

        oldlabel = acc['lbl'].lower()
        newlabel =''

        if 'up' in oldlabel:
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
