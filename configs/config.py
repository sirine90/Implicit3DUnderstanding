
# Basic configurations for furniture reconstruction from images.

from configs.paths import dataroot

class Config(object):
    def __init__(self, dataset):
        """
        Configuration of data paths.
        """
        self.dataset = dataset
        self.root_path = dataroot + self.dataset
        self.train_split = self.root_path + '/splits/train.json'
        self.test_split = self.root_path + '/splits/test.json'
        self.metadata_path = self.root_path + '/metadata'
        self.train_test_data_path = self.root_path + '/train_test_data'