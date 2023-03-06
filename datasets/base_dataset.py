
import numpy as np
from imageio import imread
from PIL import Image

from termcolor import colored, cprint

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from torchvision import datasets

from configs.paths import dataroot

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def CreateDataset(opt):        
    if opt.dataset_mode == 'snet':
        from datasets.dataset import FUTURENetDataset
        train_dataset = FUTURENetDataset()
        test_dataset = FUTURENetDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)
        
    elif opt.dataset_mode == 'snet_code':
        from datasets.dataset import FUTURECodeDataset
        train_dataset = FUTURECodeDataset()
        test_dataset = FUTURECodeDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)
        
    elif opt.dataset_mode == 'snet_img':
        from datasets.dataset import FUTURENetImgDataset
        train_dataset = FUTURENetImgDataset()
        test_dataset = FUTURENetImgDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)


    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    cprint("[*] Dataset has been created: %s" % (train_dataset.name()), 'blue')
    return train_dataset, test_dataset
