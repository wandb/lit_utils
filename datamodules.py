"""Lightning DataModules and associated utilities."""
import multiprocessing
import os

from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision

DEFAULT_WORKERS = multiprocessing.cpu_count()  # cpu_count is a good default worker count


# drop slow mirror from list of MNIST mirrors
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]


class AbstractMNISTDataModule(pl.LightningDataModule):
    """Abstract DataModule for the MNIST handwritten digit dataset.

    Must be made concrete by defining a .setup method which sets attrributes
    self.training_data and self.validation_data.

    Only downloads the training set, but performs a validation split in the
    setup step.
    """

    def __init__(self, batch_size=64, validation_size=10_000, num_workers=DEFAULT_WORKERS):
        super().__init__()
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.num_workers = DEFAULT_WORKERS

    def prepare_data(self):
        # download the data from the internet
        torchvision.datasets.MNIST(".", download=True)

    def setup(self, stage=None):
        raise NotImplementedError

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


class MNISTDataModule(AbstractMNISTDataModule):
    """DataModule for the MNIST handwritten digit classification task.

    Converts images to float and normalizes to [0, 1] in setup.
    """

    def setup(self, stage=None):
        mnist = torchvision.datasets.MNIST(".", train=True, download=False)

        self.digits, self.targets = mnist.data.float(), mnist.targets
        self.digits = torch.divide(self.digits, 255.)

        self.training_data = TensorDataset(self.digits[:-self.validation_size, None, :, :],
                                           self.targets[:-self.validation_size])
        self.validation_data = TensorDataset(self.digits[-self.validation_size:, None, :, :],
                                             self.targets[-self.validation_size:])


class AutoEncoderMNIST(torchvision.datasets.MNIST):
    """Dataset for the MNIST handwritten digit reconstruction task.

    Modified from torchvision MNIST Dataset code.
    """

    def __getitem__(self, index):
        """Gets the image at index.

        Args:
          index (int): Index

        Returns:
          tuple: (image, image)
        """
        _img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(_img.numpy(), mode="L")
        target = Image.fromarray(_img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


class AutoEncoderMNISTDataModule(AbstractMNISTDataModule):
    """DataModule for an MNIST handwritten digit auto-encoding task."""

    def __init__(self, batch_size=64, validation_size=10_000, transforms=None):
        super().__init__(batch_size=batch_size, validation_size=validation_size)

        if transforms is None:
            transforms = []
        if isinstance(transforms, torch.nn.Module):
            transforms = [transforms]

        self.transforms = [torchvision.transforms.ToTensor()] + transforms
        self.full_size = 60_000

    def setup(self, stage=None):
        composed_transform = torchvision.transforms.Compose(self.transforms)

        if stage == "fit" or stage is None:
            mnist_full = AutoEncoderMNIST(
                ".", train=True, download=False,
                transform=composed_transform,
                target_transform=torchvision.transforms.ToTensor())
            split_sizes = [self.full_size - self.validation_size, self.validation_size]
            self.training_data, self.validation_data = torch.utils.data.random_split(
                mnist_full, split_sizes)
        else:
            raise NotImplementedError
