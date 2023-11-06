import logging
import numpy as np

import pytorch_lightning as pl

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional

logger = logging.getLogger(__name__) # Used to track what is happening in the program, makes it much easier to debug the code.

class SpikeDataset(Dataset):
    """
    Custom dataset class for handling spike data and its corresponding targets.

    Args:
        in_path (str): The path to the spike data file.
        target_path (str): The path to the target data file.

    Attributes:
        spike_data (numpy.ndarray): The spike data loaded from `in_path`.
        target_data (numpy.ndarray): The target data loaded from `target_path`.

    Methods:
        __len__(): Returns the number of data samples in the dataset.
        __getitem__(idx): Retrieves a data sample and its corresponding target by index.

    """
    def __init__(self,in_path,target_path):
        super().__init__()
        self.spike_data = np.load(in_path,mmap_mode='r+')
        self.target_data = np.load(target_path, mmap_mode='r+')[:,:,None]

    def __len__(self):
        """
        Get the number of data samples in the dataset.

        Returns:
            int: The total number of data samples in the dataset.
        """
        return len(self.spike_data)

    
    def __getitem__(self, idx):
        """
        Get a specific data sample and its corresponding target by index.

        Args:
            idx (int): Index of the data sample to retrieve.

        Returns:
            dict: A dictionary containing 'data' and 'target' as Torch tensors.
        """
        data_sample = torch.from_numpy(self.spike_data[idx]).float()
        target_sample = torch.from_numpy(self.target_data[idx]).float()
        sample = {'data': data_sample, 'target': target_sample}
        return sample

class SpikeDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for handling spike data and dataloaders for training, validation, and testing.

    Args:
        config: Configuration file containing relevant data paths, batch size, and the number of workers.

    Attributes:
        prepare_data_per_node (bool): Whether to prepare data per node.
        in_path (str): The path to the spike data file.
        target_path (str): The path to the target data file.
        batch_size (int): The batch size for data loading.
        num_workers (int): The number of workers for data loading.

    Methods:
        setup(stage: Optional[str] = None): Sets up the dataset for training, validation, or testing.
        train_dataloader(): Returns the dataloader for training.
        val_dataloader(): Returns the dataloader for validation.
        test_dataloader(): Returns the dataloader for testing.

    """
    def __init__(self, config):
        super().__init__()
        self.prepare_data_per_node = True
        self.in_path = config.in_path
        self.target_path = config.target_path
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

    def setup(self, stage: Optional[str] = None):
        """
        Sets up the dataset for training, validation, or testing.

        Args:
            stage (str, optional): The stage for which to set up the dataset. Options are "fit" (training),
            "test" (testing), or None (default, for both).

        """
        if stage == "fit" or stage is None:
            train_set_full = SpikeDataset(self.in_path, self.target_path)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])


        if stage == "test" or stage is None:
            self.test = SpikeDataset(self.in_path, self.target_path)

    def train_dataloader(self):
        """
        Returns the dataloader for training.

        Returns:
            DataLoader: The dataloader for the training dataset.

        """
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Returns the dataloader for validation.

        Returns:
            DataLoader: The dataloader for the validation dataset.

        """

        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Returns the dataloader for testing.

        Returns:
            DataLoader: The dataloader for the testing dataset.

        """
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
