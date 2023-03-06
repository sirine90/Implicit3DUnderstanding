
import os
from torch.utils.data import Dataset
import json
#from configs.config import Config
from tqdm import tqdm
from configs.paths import dataroot


class FRONT3D(Dataset):
    def __init__(self, config, mode):
        '''
        initiate FRONT3D dataset for data loading
        :param config: config file
        :param mode: train/val/test mode
        '''
        self.config = config
        if mode == 'val':
            mode = 'test'
        self.mode = mode
        split_file = os.path.join(dataroot.replace('data', ''), config['data']['split'], mode + '.json')
        with open(split_file) as file:
            split = json.load(file)
        self.split = []
        skipped = 0
        for s in tqdm(split):
            s=os.path.join(dataroot.replace('data', ''), s[2:])
            if os.path.exists(s):
                self.split.append(s)
            else:
                skipped += 1
        print(f'{skipped}/{len(split)} missing samples')

    def __len__(self):
        return len(self.split)

class SUNRGBD(Dataset):
    def __init__(self, config, mode):
        '''
        initiate SUNRGBD dataset for data loading
        :param config: config file
        :param mode: train/val/test mode
        '''
        self.config = config
        self.mode = mode
        split_file = os.path.join(config['data']['split'], mode + '.json')
        with open(split_file) as file:
            self.split = json.load(file)

    def __len__(self):
        return len(self.split)


class PIX3D(Dataset):
    def __init__(self, config, mode):
        '''
        initiate PIX3D dataset for data loading
        :param config: config file
        :param mode: train/val/test mode
        '''
        self.config = config
        if mode == 'val':
            mode = 'test'
        self.mode = mode
        split_file = os.path.join(config['data']['split'], mode + '.json')
        with open(split_file) as file:
            self.split = json.load(file)

    def __len__(self):
        return len(self.split)


class PIX3DLDIF(Dataset):
    def __init__(self, config, mode):
        '''
        initiate PIX3DLDIF dataset for data loading
        :param config: config file
        :param mode: train/val/test mode
        '''
        self.config = config
        self.mode = mode
        if mode == 'val':
            mode = 'test'
        split_file = os.path.join(config['data']['split'], mode + '.json')
        with open(split_file) as file:
            split = json.load(file)
        ids = [int(os.path.basename(file).split('.')[0]) for file in split if 'flipped' not in file]
        sample_info = []
        skipped = 0
        for id in tqdm(ids):
            #metadata = metadatas[id]
            info = {}
            watertight = self.config['data'].get('watertight',
                                                 self.config['model']['mesh_reconstruction']['method'] == 'LDIF')
            ext_mgnet = 'mgn' if watertight else 'org'
            if not all([os.path.exists(path) for path in info.values()]) :
                
                skipped += 1
                continue

            info['sample_id'] = id
            sample_info.append(info)
        print(f'{skipped}/{len(ids)} missing samples')
        self.split = sample_info

    def __len__(self):
        return len(self.split)