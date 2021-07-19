"""Lightning DataModules and associated utilities for FER2013 dataset."""
import multiprocessing
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

DEFAULT_NUM_WORKERS = multiprocessing.cpu_count()


class FERDataModule(pl.LightningDataModule):
    """DataModule for downloading and preparing the FER2013 dataset."""

    tar_url = "https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar"
    data_root = Path(".") / "data"
    local_path = data_root / "fer2013"
    csv_file = local_path / "fer2013.csv"
    width, height = 48, 48
    classes = ["anger", "disgust", "fear", "happiness",
               "sadness", "surprise", "neurality"]

    def __init__(self, batch_size=64, validation_size=0.2, num_workers=DEFAULT_NUM_WORKERS):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = 2 * self.batch_size
        self.num_workers = num_workers
        self.validation_size = validation_size

    def setup(self):
        # download the data from the internet
        os.makedirs(self.local_path, exist_ok=True)
        self.download_data()

    def prepare_data(self):
        # read data from a .csv file
        faces, emotions = self.read_data()

        # normalize it
        faces = torch.divide(faces, 255.)

        # split it into training and validation
        num_validation = int(len(faces) * self.validation_size)

        self.training_data = torch.utils.data.TensorDataset(
          faces[:-num_validation], emotions[:-num_validation])
        self.validation_data = torch.utils.data.TensorDataset(
          faces[-num_validation:], emotions[-num_validation:])

        # record metadata
        self.num_total, self.num_classes = emotions.shape[0], len(self.classes)
        self.num_train = self.num_total - num_validation
        self.num_validation = num_validation

    def train_dataloader(self):
        """The DataLoaders returned by a DataModule produce data for a model.
        This DataLoader is used during training.
        """
        return DataLoader(self.training_data, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """The DataLoaders returned by a DataModule produce data for a model.
        This DataLoader is used during validation, at the end of each epoch.
        """
        return DataLoader(self.validation_data, batch_size=self.val_batch_size,
                          num_workers=self.num_workers)

    def download_data(self):
        if not os.path.exists(self.csv_file):
            print("Downloading the face emotion dataset...")
            subprocess.check_output(f"curl -SL {self.tar_url} | tar xz", shell=True)
            subprocess.check_output(f"mv fer2013 -t {self.data_root}", shell=True)
            print("...done")

    def read_data(self):
        """Read the data from a .csv into torch Tensors."""
        data = pd.read_csv(self.csv_file)
        pixels = data["pixels"].tolist()
        faces = []
        for pixel_sequence in pixels:
            face = np.asarray(pixel_sequence.split(
                " "), dtype=np.uint8).reshape(1, self.width, self.height)
            faces.append(face.astype("float32"))

        faces = np.asarray(faces)
        emotions = data["emotion"].to_numpy()

        return torch.tensor(faces), torch.tensor(emotions)
