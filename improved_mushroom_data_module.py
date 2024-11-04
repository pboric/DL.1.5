# improved_mushroom_data_module.py

import os
import cv2
import logging
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from collections import Counter
import platform

from improved_mushroom_dataset import MushroomDataset
from config import Config

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(handler)

class MushroomDataModule(pl.LightningDataModule):
    """
    Enhanced DataModule with improved handling of workers, data loading,
    and error checking.
    """
    def __init__(self, data_path, batch_size=32, num_workers=None,
                 train_transforms=None, val_transforms=None):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        
        # Smart num_workers setting
        if num_workers is None:
            if platform.system() == 'Windows':
                num_workers = min(int(os.cpu_count() * 0.75), 8)
            else:
                num_workers = min(int(os.cpu_count() * 0.75), 15)
        
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        
        self.train_dataset = None
        self.val_dataset = None
        self.class_weights = None
        self.class_to_idx = None
        
        logger.info(f"DataModule initialized with batch_size: {batch_size}, workers: {self.num_workers}")

    def setup(self, stage=None):
        logger.info(f"Setting up DataModule for stage: {stage}")

        image_paths = []
        labels = []
        self.class_to_idx = {}

        # Collect and validate data
        for idx, genus in enumerate(sorted(os.listdir(self.data_path))):
            genus_path = self.data_path / genus
            if genus_path.is_dir():
                self.class_to_idx[genus] = idx
                logger.info(f"Processing genus {genus} (class {idx})")
                
                for img_path in genus_path.glob("*.jpg"):
                    # Verify image is readable
                    try:
                        if cv2.imread(str(img_path)) is not None:
                            image_paths.append(str(img_path))
                            labels.append(idx)
                        else:
                            logger.warning(f"Could not read image {img_path}")
                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {str(e)}")

        # Ensure data and labels match
        assert len(image_paths) == len(labels), "Mismatch between image paths and labels"
        logger.info(f"Found {len(image_paths)} valid images across {len(self.class_to_idx)} classes")

        # Compute class weights
        label_counts = Counter(labels)
        total_samples = len(labels)
        self.class_weights = torch.tensor([
            total_samples / (len(self.class_to_idx) * label_counts.get(i, 1))
            for i in range(len(self.class_to_idx))
        ], dtype=torch.float)

        # Stratified split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Create datasets
        self.train_dataset = MushroomDataset(train_paths, train_labels, transform=self.train_transforms)
        self.val_dataset = MushroomDataset(val_paths, val_labels, transform=self.val_transforms)

        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Val dataset size: {len(self.val_dataset)}")
        logger.info("Class weights computed and datasets created successfully")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None
        )