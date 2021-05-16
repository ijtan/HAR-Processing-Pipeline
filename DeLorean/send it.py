import os,pathlib,json
from tqdm import tqdm


def read_all(path, entries):
    global last_folder
    global session_entries

    for log in tqdm(pathlib.Path(path).iterdir(), desc="Reading"):

        if not log.is_file():
            # print('PATH IS NOT FILE!!')
            read_all(log, entries)
            continue

        if '.json' not in str(log):
            print('not a json; skipping...')
            continue

        with open(log, 'r', encoding="utf8") as f:
            file = json.load(f)

        for entry in file:
            entry['time'] += 86400000

        with open('zz'+str(log).split('.')[-2]+'-shifted.json', 'w', encoding="utf8") as f:
            json.dump(file,f)



if __name__ == '__main__':
    raw_entries = {'acc': [], 'gyr': []}
    read_all('DeLorean\\SHIFT_TIME', raw_entries)
    raw_entries['acc'] = sorted(raw_entries['acc'], key=lambda item: item['time'])
    raw_entries['gyr'] = sorted(raw_entries['gyr'], key=lambda item: item['time'])
    
    # json.dump(raw_entries)
