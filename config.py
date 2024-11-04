import os 
import torch

class Config:
    # Data
    DATA_PATH = r"C:\Users\kresi\OneDrive\Desktop\Turing college\DL\Mushroom\data"
    IMG_SIZE = 224  # Adjusted for ResNet34
    BATCH_SIZE = 32  # Adjusted batch size

    # Model
    NUM_CLASSES = 9
    LEARNING_RATE = 1e-4  # Adjusted learning rate

    # Training
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5

    # Device will be set in the main block to avoid issues with multiprocessing
    DEVICE = None
