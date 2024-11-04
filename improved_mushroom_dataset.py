# improved_mushroom_dataset.py

import cv2
import numpy as np
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class MushroomDataset(Dataset):
    """
    Enhanced dataset class for mushroom images with better error handling
    and logging.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Validate inputs
        if len(self.image_paths) != len(self.labels):
            raise ValueError("Number of images and labels must match")
        
        # Verify images are readable at initialization
        self._validate_images()
        
        logger.info(f"Dataset initialized with {len(self.image_paths)} images")

    def _validate_images(self):
        """Verify all images are readable at initialization"""
        invalid_images = []
        for img_path in self.image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    invalid_images.append(img_path)
            except Exception as e:
                invalid_images.append(img_path)
                logger.error(f"Error reading image {img_path}: {str(e)}")
        
        if invalid_images:
            raise ValueError(f"Found {len(invalid_images)} invalid images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = self.labels[idx]

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            return image, label
            
        except Exception as e:
            logger.error(f"Error processing image at index {idx}: {str(e)}")
            raise