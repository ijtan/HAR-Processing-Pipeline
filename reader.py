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
        # print(f'opening file: {str(log)}')
        with open(log, 'r', encoding="utf8") as f:
            file = json.load(f)
            if 'ACC' in str(log) or '_a' in str(log):
                entries['acc'].extend(file)
            elif 'GYR' in str(log) or '_g' in str(log):
                entries['gyr'].extend(file)
            else:
                ValueError('other file found!')
            # print(file[0])


def lenDiff(raw_entries):
    count = 0
    # print('len match!')
    # print(f"{len(raw_entries['acc'])}={len(raw_entries['gyr'])}")
    for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
        # print('iter')
        if abs(acc['time']-gyr['time']) > 0:
            count += 1
            # print(str(acc['time']-gyr['time'])+'\t\t' +
            #       str(raw_entries['acc'].index(acc)))
            # pass
    print(f'\n{count} items out of sync')
    return count


def sync2(logs, x=4, avg_diff=17):

    for ac in range(0, len(logs['acc'])):

        while abs(sum([a['time']-g['time'] for a, g in zip(logs['acc'][ac:ac+x], logs['gyr'][ac:ac+x])])/x) >= avg_diff:
            lastx = [a['time']-g['time']
                     for a, g in zip(logs['acc'][ac:ac+x], logs['gyr'][ac:ac+x])]
            avg = sum(lastx)/x

            if avg >= avg_diff:
                del logs['gyr'][ac]
            elif avg <= -avg_diff:
                del logs['acc'][ac]





# print(raw_entries['acc'][0])
# print(raw_entries['gyr'][0])

def getRaw():
    raw_entries = {'acc': [], 'gyr': []}
    read_all('INPUT', raw_entries)

    raw_entries['acc'] = sorted(
        raw_entries['acc'], key=lambda item: item['time'])
    raw_entries['gyr'] = sorted(
        raw_entries['gyr'], key=lambda item: item['time'])

    sync2(raw_entries)

    count = 0

    for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
        # print('iter')
        if abs(acc['time']-gyr['time']) > 4:
            count += 1
    print(f'\n{count} items out of sync by more than 4')

    while lenDiff(raw_entries) > 0:
        print('removing mismatches!')
        for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
            if abs(acc['time']-gyr['time']) <= 10:
                acc['time'] = gyr['time']
            else:
                # print('removing', raw_entries['acc'].index(acc))
                raw_entries['acc'].remove(acc)
                raw_entries['gyr'].remove(gyr)
    return raw_entries

def getData():
    

    raw_entries = getRaw()

    # print(raw_entries['acc'][0])


    # finalLog = {'XA':[],'YA':[],'ZA':[],'XR':[],'YR':[],'ZR':[],'time':[]}
    finalLog = []
    for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):

        # print(acc)
        entry = {}
        entry['time'] = acc['time']


        entry['XA'] = acc['XA']
        entry['YA'] = acc['YA']
        entry['ZA'] = acc['ZA']

        entry['XR'] = gyr['XR']
        entry['YR'] = gyr['YR']
        entry['ZR'] = gyr['ZR']

        finalLog.append(entry)
        
        
        
        # , raw_entries['acc']['YA'], raw_entries['acc']
        #                 ['ZA'], raw_entries['gyr']['XR'], raw_entries['gyr']['YR'], raw_entries['gyr']['ZR']))
    finalLog = np.asarray(finalLog)
    print(finalLog[0])
    return finalLog
    # print(raw_entries)


if __name__ == '__main__':
    getData()
