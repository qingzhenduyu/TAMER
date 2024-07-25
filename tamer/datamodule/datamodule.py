import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader

from tamer.datamodule.dataset import HMEDataset

from .vocab import vocab

Data = List[Tuple[str, np.ndarray, List[str]]]

# load data


def data_iterator(
    data: Data,
    batch_size: int,
    max_size: int,
    is_train: bool,
    maxlen: int = 200,
):
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    data.sort(key=lambda x: x[1].shape[0] * x[1].shape[1])

    i = 0
    for fname, fea, lab in data:
        size = fea.shape[0] * fea.shape[1]
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if is_train and len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif is_train and size > max_size:
            print(
                f"image: {fname} size: {fea.shape[0]} x {fea.shape[1]} =  bigger than {max_size}, ignore"
            )
        else:
            if batch_image_size > max_size or i == batch_size:  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                label_batch = []
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))


def extract_data(folder: str, dir_name: str) -> Data:
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    with open(os.path.join(folder, dir_name, "images.pkl"), "rb") as f:
        images = pickle.load(f)
    with open(os.path.join(folder, dir_name, "caption.txt"), "r") as f:
        captions = f.readlines()
    data = []
    for line in captions:
        tmp = line.strip().split()
        img_name = tmp[0]
        formula = tmp[1:]
        img = images[img_name]
        data.append((img_name, img, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

    # return fnames, x, x_mask, seqs_y
    return Batch(fnames, x, x_mask, seqs_y)


def build_dataset(archive, folder: str, batch_size: int, max_size: int, is_train: bool):
    data = extract_data(archive, folder)
    return data_iterator(data, batch_size, max_size, is_train)


class HMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        folder: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data/crohme",
        test_folder: str = "2014",
        max_size: int = 32e4,
        scale_to_limit: bool = True,
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        num_workers: int = 5,
        scale_aug: bool = False,
    ) -> None:
        super().__init__()
        assert isinstance(test_folder, str)
        self.folder = folder
        self.test_folder = test_folder
        self.max_size = max_size
        self.scale_to_limit = scale_to_limit
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug

        vocab.init(os.path.join(folder, "dictionary.txt"))

        print(f"Load data from: {self.folder}")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = HMEDataset(
                build_dataset(
                    self.folder, "train", self.train_batch_size, self.max_size, True
                ),
                True,
                self.scale_aug,
                self.scale_to_limit,
            )
            self.val_dataset = HMEDataset(
                build_dataset(
                    self.folder,
                    self.test_folder,
                    self.eval_batch_size,
                    self.max_size,
                    False,
                ),
                False,
                self.scale_aug,
                self.scale_to_limit,
            )
        if stage == "test" or stage is None:
            self.test_dataset = HMEDataset(
                build_dataset(
                    self.folder,
                    self.test_folder,
                    self.eval_batch_size,
                    self.max_size,
                    False,
                ),
                False,
                self.scale_aug,
                self.scale_to_limit,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
