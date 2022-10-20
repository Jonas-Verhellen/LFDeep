import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

logger = logging.getLogger(__name__)

class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "", batch_size: int = 32, num_workers: int = 8, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        logger.info("Initialize CIFAR DataModule")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, download=True, transform=ToTensor())
            split_size = [int(0.8 * len(cifar_full)), int(0.2 * len(cifar_full))]
            self.mnist_train, self.mnist_val = random_split(cifar_full, split_size)
            self.dims = tuple(self.mnist_train[0][0].shape)
        elif stage == 'test' or stage is None:
            self.mnist_test = CIFAR10(self.data_dir, train=False, download=True, transform=ToTensor())

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    # Double workers for val and test loaders since there is no backward pass and GPU computation is faster
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers*2)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers*2)
