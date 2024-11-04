import os
import cv2
import logging
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from mushroom_dataset import MushroomDataset
from collections import Counter
from config import Config  # Adjust the import path based on your project structure

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger if not already added
if not logger.hasHandlers():
    logger.addHandler(handler)

class MushroomDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, num_workers=4, train_transforms=None, val_transforms=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        logger.info(f"DataModule initialized with batch_size: {batch_size}, workers: {self.num_workers}")

    def setup(self, stage=None):
        logger.info(f"Setting up DataModule for stage: {stage}")

        image_paths = []
        labels = []
        self.class_to_idx = {}

        for idx, genus in enumerate(sorted(os.listdir(self.data_path))):
            genus_path = os.path.join(self.data_path, genus)
            if os.path.isdir(genus_path):
                self.class_to_idx[genus] = idx
                logger.info(f"Processing genus {genus} (class {idx})")
                for img_name in os.listdir(genus_path):
                    img_path = os.path.join(genus_path, img_name)
                    # Verify that the image can be loaded
                    if cv2.imread(img_path) is not None:
                        image_paths.append(img_path)
                        labels.append(idx)
                    else:
                        logger.warning(f"Could not read image {img_path}")

        # Ensure data and labels are of the same length
        assert len(image_paths) == len(labels), "Mismatch between image paths and labels"

        # Compute class weights for handling class imbalance
        label_counts = Counter(labels)
        total_samples = len(labels)
        self.class_weights = []
        for i in range(Config.NUM_CLASSES):
            count = label_counts.get(i, 0)
            weight = total_samples / (Config.NUM_CLASSES * count) if count > 0 else 0.0
            self.class_weights.append(weight)
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )

        self.train_dataset = MushroomDataset(train_paths, train_labels, transform=self.train_transforms)
        self.val_dataset = MushroomDataset(val_paths, val_labels, transform=self.val_transforms)

        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Val dataset size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )
