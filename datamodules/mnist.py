"""Lightning DataModules and associated utilities for MNIST-style datasets.

Based on code from torchvision MNIST datasets.

All datasets are stored in memory as tensors and then converted to PIL
before applying preprocessing.

MNIST-style datasets that are not the handwritten digits dataset can inherit
from the base class and only need to over-write the mirrors, resources, classes,
and folder attributes/properties.

Unlike most MNIST implementations but like most PNG images, uses 255 to represent white
background and 0 to represent black foreground/strokes.
"""
from math import floor
import multiprocessing
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision

DEFAULT_WORKERS = multiprocessing.cpu_count()  # cpu_count is a good default worker count


class AbstractMNISTDataModule(pl.LightningDataModule):
    """Abstract DataModule for MNIST-style datasets.

    Must be made concrete by defining a .setup method which sets attributes
    self.training_data and self.validation_data and self.test_data.
    """
    data_root = Path(".") / "data"
    seed = 117

    def __init__(self, batch_size=64, validation_size=10000, num_workers=DEFAULT_WORKERS,
                 transform=None, target_transform=None):
        super().__init__()
        self._dataset = None

        self.batch_size = batch_size
        self.validation_size = validation_size
        self.num_workers = DEFAULT_WORKERS

        self.transform = transform
        self.target_transform = target_transform

    def prepare_data(self):
        # download the data from the internet
        self.dataset(self.data_root, download=True)
        self.dataset(self.data_root, train=False, download=True)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.mnist_fit = self.dataset(
                self.data_root, train=True, download=False,
                transform=self.transform, target_transform=self.target_transform)

            total_size, *self.dims = self.mnist_fit.data.shape
            split_sizes = [total_size - self.validation_size, self.validation_size]
            split_generator = torch.Generator().manual_seed(self.seed)
            self.training_data, self.validation_data = torch.utils.data.random_split(
                self.mnist_fit, split_sizes, split_generator)

        if stage in (None, "test"):
            self.mnist_test = self.dataset(
                self.data_root, train=False, download=False,
                transform=None, target_transform=None)
            self.test_data = self.mnist_test

    def train_dataloader(self):
        """The DataLoaders returned by a DataModule produce data for a model.

        This DataLoader is used during training.
        """
        return DataLoader(self.training_data, batch_size=self.batch_size,
                          num_workers=DEFAULT_WORKERS)

    def val_dataloader(self):
        """The DataLoaders returned by a DataModule produce data for a model.

        This DataLoader is used during validation, at the end of each epoch.
        """
        return DataLoader(self.validation_data, batch_size=2 * self.batch_size,
                          num_workers=DEFAULT_WORKERS, shuffle=False)

    def test_dataloader(self):
        """The DataLoaders returned by a DataModule produce data for a model.

        This DataLoader is used during testing, at the end of training.
        """
        return DataLoader(self.test_data, batch_size=2 * self.batch_size,
                          num_workers=DEFAULT_WORKERS, shuffle=False)

    @property
    def dataset(self):
        if self._dataset is None:
            raise NotImplementedError("must provide an MNIST-style dataset class")
        return self._dataset

    @dataset.setter
    def dataset(self, cls):
        if not issubclass(cls, torchvision.datasets.MNIST):
            raise ValueError(f"dataset must be subclass of torchvision.datasets.MNIST, was {cls}")
        self._dataset = cls

    @staticmethod
    def get_split_sizes(train_size, total_size):
        if isinstance(train_size, float):
            train_size = floor(total_size * train_size)

        val_size = total_size - train_size

        return train_size, val_size


class MNISTDataModule(AbstractMNISTDataModule):
    """DataModule for the MNIST handwritten digit classification task."""

    def __init__(self, batch_size=64, validation_size=10000, transform=None,
                 target_transform=None):
        super().__init__(batch_size=batch_size, validation_size=validation_size,
                         transform=transform, target_transform=target_transform)
        self.dataset = ClassificationMNIST
        self.classes = self.dataset.classes


class AutoEncoderMNISTDataModule(AbstractMNISTDataModule):
    """DataModule for an MNIST handwritten digit auto-encoding task."""

    def __init__(self, batch_size=64, validation_size=10000, transform=None,
                 target_transform=None):
        super().__init__(batch_size=batch_size, validation_size=validation_size,
                         transform=transform, target_transform=target_transform)
        self.dataset = AutoEncoderMNIST
        self.classes = self.dataset.classes


class FashionMNISTDataModule(AbstractMNISTDataModule):
    """DataModule for the MNIST handwritten digit classification task."""

    def __init__(self, batch_size=64, validation_size=10000, transform=None,
                 target_transform=None):
        super().__init__(batch_size=batch_size, validation_size=validation_size,
                         transform=transform, target_transform=target_transform)
        self.dataset = ClassificationFashionMNIST
        self.classes = self.dataset.classes


class AutoEncoderFashionMNISTDataModule(AbstractMNISTDataModule):
    """DataModule for an MNIST handwritten digit auto-encoding task."""

    def __init__(self, batch_size=64, validation_size=10000, transform=None,
                 target_transform=None):
        super().__init__(batch_size=batch_size, validation_size=validation_size,
                         transform=transform, target_transform=target_transform)
        self.dataset = AutoEncoderFashionMNIST
        self.classes = self.dataset.classes


class ClassificationMNIST(torchvision.datasets.MNIST):
    """Dataset for the MNIST handwritten digit recognition task.

    Modified from torchvision MNIST Dataset code.
    """
    # remove slow mirror from default list
    mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
               if not mirror.startswith("http://yann.lecun.com")]

    classes = [str(ii) for ii in range(10)]

    default_transform = torchvision.transforms.ToTensor()
    default_target_transform = torchvision.transforms.Compose([])

    def __getitem__(self, index):
        """Gets the image and label at index.

        Args:
          index (int): Index

        Returns:
          tuple: (image, target) where target is the index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        img = reverse_palette(img)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.default_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = self.default_target_transform(target)

        return img, target

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


class AutoEncoderMNIST(torchvision.datasets.MNIST):
    """Dataset for the MNIST handwritten digit reconstruction task.

    Modified from torchvision MNIST Dataset code.
    """
    # remove slow mirror from default list
    mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
               if not mirror.startswith("http://yann.lecun.com")]

    classes = [str(ii) for ii in range(10)]

    default_transform = torchvision.transforms.ToTensor()
    default_target_transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        """Gets the image at index.

        Args:
          index (int): Index

        Returns:
          tuple: (image, image)
        """
        _img = self.data[index]
        _img = reverse_palette(_img)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(_img.numpy(), mode="L")
        target = Image.fromarray(_img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.default_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = self.default_target_transform(target)

        return img, target

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


class ClassificationFashionMNIST(ClassificationMNIST):
    """Dataset for the MNIST fashion item recognition task."""

    mirrors = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310")
    ]

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
               "Shirt", "Sneaker", "Bag", "Ankle boot"]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "processed")


class AutoEncoderFashionMNIST(AutoEncoderMNIST):
    """Dataset for the MNIST fashion item reconstruction task."""

    mirrors = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310")
    ]

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
               "Shirt", "Sneaker", "Bag", "Ankle boot"]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "processed")


def reverse_palette(img):
    return np.abs(255 - img)
