import os
import json
import pathlib


def read_all(path, entries):
    for log in pathlib.Path(path).iterdir():

        if not log.is_file():
            print('PATH IS NOT FILE!!')
            read_all(log, entries)

            continue

        with open(log, 'r') as f:
            file = json.load(f)
            if 'ACC' in str(log) or '_a' in str(log):
                entries['acc'].extend(file)
            elif 'GYR' in str(log) or '_g' in str(log):
                entries['gyr'].extend(file)
            else:
                ValueError('other file found!')
            # print(file[0])


def sync2(logs):
    # last5 = [a['time']-g['time'] for a,g in zip(logs['acc'][5:], logs['gyr'][5:])]
    # ac = 0
    # gc = 5
    # print(last5)

    for ac in range(0,len(logs['acc'])):
        last5 = [a['time']-g['time'] for a,g in zip(logs['acc'][ac:ac+5], logs['gyr'][ac:ac+5])]
        avg = sum(last5)/5
        if avg >= 15:
            raw_entries['gyr'] = raw_entries['gyr'][1:]
        elif avg <= -15:
            raw_entries['acc'] = raw_entries['acc'][1:]
        # print(last5)
        print(avg,end = ' ')

if __name__ == '__main__':
    raw_entries = {'acc': [], 'gyr': []}
    read_all('INPUT', raw_entries)

    raw_entries['acc'] = sorted(raw_entries['acc'], key=lambda item: item['time'])
    raw_entries['gyr'] = sorted(raw_entries['gyr'], key=lambda item: item['time'])

    sync2(raw_entries)

    # print(raw_entries['acc'][0]['time'])
    

        # raw_entries['acc'] = raw_entries['acc'][1:]
        # raw_entries['gyr'] = raw_entries['gyr'][1:]

        # print(raw_entries['acc'][0]['time']-raw_entries['acc'][0]['time'])
        # # print(raw_entries['acc'][0]['time']-raw_entries['acc'][0]['time'])

        # if raw_entries['acc'][0]['time']-raw_entries['gyr'][0]['time']<=-20:
        #     raw_entries['acc'] = raw_entries['acc'][1:]
        # elif raw_entries['acc'][0]['time']-raw_entries['gyr'][0]['time'] >= 20:
        #     raw_entries['gyr'] = raw_entries['gyr'][1:]


    if len(raw_entries['acc']) != len(raw_entries['gyr']):
        print('Length difference detected!')
        print(f"{len(raw_entries['acc'])}!={len(raw_entries['gyr'])}")
        if len(raw_entries['acc']) > len(raw_entries['gyr']):
            diff = len(raw_entries['acc']) - len(raw_entries['gyr'])
            raw_entries['acc'] = raw_entries['acc'][diff:]
        elif len(raw_entries['acc']) < len(raw_entries['gyr']):
            diff = len(raw_entries['gyr']) - len(raw_entries['acc'])
            raw_entries['gyr'] = raw_entries['gyr'][diff:]

    sum = 0
    total = 0
    for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
        sum += acc['time']-gyr['time']
        total+=1
    avg = sum/total
    print(f'AVG: {avg}')
    if avg >=15:
        raw_entries['gyr'] = raw_entries['gyr'][1:]
    elif avg <= -15:
        raw_entries['acc'] = raw_entries['acc'][1:]

    print('len match!')
    print(f"{len(raw_entries['acc'])}={len(raw_entries['gyr'])}")
    for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
        # print('iter')
        if abs(acc['time']-gyr['time'])>2:
            print(acc['time']-gyr['time'],end=' ')
            # pass
    # print(raw_entries)



