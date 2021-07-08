from math import floor
import multiprocessing
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

DEFAULT_NUM_WORKERS = multiprocessing.cpu_count()


class CIFAR10DataModule(pl.LightningDataModule):
    """Dataloaders and setup for the CIFAR10 dataset.

    Arguments:
    batch_size: int. Size of batches in training, validation, and test
    train_size: int or float. If int, number of examples in training set,
                If float, fraction of examples in training set.
    debug:  bool. If True, cut dataset size by a factor of 10.
    """
    data_root = Path(".") / "data"
    seed = 117
    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

    def __init__(self, batch_size, train_size=0.8):
        super().__init__()

        self.train_size, self.batch_size = train_size, batch_size

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def prepare_data(self):
        """Download the dataset."""
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                     download=True, transform=self.transform)

        torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                     download=True, transform=self.transform)

    def setup(self, stage=None):
        """Set up training and test data and perform our train/val split."""
        if stage in (None, "fit"):
            self.cifar10_fit = torchvision.datasets.CIFAR10(
                self.data_dir, train=True, download=False, transform=self.transform)
            total_size, *self.dims = self.cifar10_fit.data.shape
            split_sizes = self.get_split_sizes(self.train_size, total_size)

            split_generator = torch.Generator().manual_seed(self.seed)
            self.training_data, self.validation_data = torch.utils.data.random_split(
                self.cifar10_fit, split_sizes, split_generator)

        if stage in (None, "test"):
            self.test = torchvision.datasets.CIFAR10(self.data_dir, train=False,
                                                     transform=self.transform)

    def train_dataloader(self):
        trainloader = DataLoader(self.training_data, batch_size=self.batch_size,
                                 shuffle=True, num_workers=DEFAULT_NUM_WORKERS)
        return trainloader

    def val_dataloader(self):
        valloader = DataLoader(self.validation_data, batch_size=2 * self.batch_size,
                               shuffle=False, num_workers=DEFAULT_NUM_WORKERS)
        return valloader

    def test_dataloader(self):
        testloader = DataLoader(self.test_data, batch_size=2 * self.batch_size,
                                shuffle=False, num_workers=DEFAULT_NUM_WORKERS)
        return testloader

    @staticmethod
    def get_split_sizes(train_size, total_size):
        if isinstance(train_size, float):
            train_size = floor(total_size * train_size)

        val_size = total_size - train_size

        return train_size, val_size
