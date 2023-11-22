import pytorch_lightning as pl
import random
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional

from data.imagenet import logger


class HEDataModule(pl.LightningDataModule):
    def __init__(self,
                 num_workers: int,
                 batch_size: int,
                 data_dir: str,
                 train_transforms,
                 num_images: int,
                 val_transforms = None,
                 size_val_set: int = 10):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.size_val_set = size_val_set
        self.num_images = num_images
        self.file_list = self.make_dataset(data_dir)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.im_train = None
        self.im_val = None
        random.shuffle(self.file_list)
        logger.info(f"Found {len(self.file_list)} many images")

    def make_dataset(self, data_dir):
        files = os.listdir(data_dir)

        # Filter the list to include only PNG files
        instances = [os.path.join(data_dir, file) for file in files if file.lower().endswith('.png')]

        return instances

    def __len__(self):
        return len(self.file_list)

    def setup(self, stage: Optional[str] = None):
        # Split test set in val an test
        if stage == 'fit' or stage is None:
            self.im_train = UnlabelledHe(self.file_list, self.train_transforms)
            assert len(self.im_train) == self.num_images
            print(f"Train size {len(self.im_train)}")
        else:
            raise NotImplementedError("There is no dedicated val/test set.")
        logger.info(f"Data Module setup at stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.im_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          drop_last=True, pin_memory=True)


class UnlabelledHe(Dataset):

    def __init__(self, file_list, transforms):
        self.file_names = file_list
        self.transform = transforms

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = self.file_names[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

