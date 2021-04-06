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


def sync2(logs,x=5,avg_diff = 17):

    for ac in range(0,len(logs['acc'])):       
        
        while abs(sum([a['time']-g['time'] for a, g in zip(logs['acc'][ac:ac+x], logs['gyr'][ac:ac+x])])/x)>=avg_diff:
            lastx = [a['time']-g['time']
                     for a, g in zip(logs['acc'][ac:ac+x], logs['gyr'][ac:ac+x])]
            avg = sum(lastx)/x
            if avg >= avg_diff:
                del logs['gyr'][ac]
            elif avg <= -avg_diff:
                del logs['acc'][ac]


if __name__ == '__main__':
    raw_entries = {'acc': [], 'gyr': []}
    read_all('INPUT', raw_entries)

    raw_entries['acc'] = sorted(raw_entries['acc'], key=lambda item: item['time'])
    raw_entries['gyr'] = sorted(raw_entries['gyr'], key=lambda item: item['time'])

    sync2(raw_entries)




    # sum = 0
    # total = 0
    # for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
    #     sum += acc['time']-gyr['time']
    #     total+=1
    # avg = sum/total
    # print(f'AVG: {avg}')
    # if avg >=15:
    #     raw_entries['gyr'] = raw_entries['gyr'][1:]
    # elif avg <= -15:
    #     raw_entries['acc'] = raw_entries['acc'][1:]
    count = 0
    print('len match!')
    print(f"{len(raw_entries['acc'])}={len(raw_entries['gyr'])}")
    for acc, gyr in zip(raw_entries['acc'], raw_entries['gyr']):
        # print('iter')
        if abs(acc['time']-gyr['time'])>4:
            count+=1
            print(str(acc['time']-gyr['time'])+'\t\t'+str(raw_entries['acc'].index(acc)))
            # pass
    print(f'\n{count} items out of sync by more than 4')
    # print(raw_entries)



