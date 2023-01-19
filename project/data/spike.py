import logging
import numpy as np

import pytorch_lightning as pl

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional

logger = logging.getLogger(__name__) # Used to track what is happening in the program, makes it much easier to debug the code.

class SpikeDataset(Dataset):
    def __init__(self,in_path,target_path):
        super().__init__()
        self.spike_data = np.load(in_path,mmap_mode='r+')
        self.target_data = np.load(target_path, mmap_mode='r+')[:,:,None]

    def __len__(self):
        return len(self.spike_data)

    
    def __getitem__(self, idx):
        data_sample = torch.from_numpy(self.spike_data[idx]).float()
        target_sample = torch.from_numpy(self.target_data[idx]).float()
        sample = {'data': data_sample, 'target': target_sample}
        return sample

class SpikeDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.prepare_data_per_node = True
        self.in_path = config.in_path
        self.target_path = config.target_path
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_set_full = SpikeDataset(self.in_path, self.target_path)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])


        if stage == "test" or stage is None:
            self.test = SpikeDataset(self.in_path, self.target_path)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
