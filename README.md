# 🍄Mushroom Classification Project - Version 2

## Project Overview

This project focuses on building an improved machine learning model for mushroom classification. The enhancements are based on:

1. **Exploratory Data Analysis (EDA)**
   - Understanding the dataset's structure and characteristics.
   - Analyzing class distribution, image dimensions, and sample visuals.
   - Extracting and analyzing color and texture features.
   - Performing statistical analyses to inform feature selection.

2. **Insights from Previous Model Evaluation**
   - Addressing the low performance on the **Suillus** class.
   - Resolving the **Lactarius-Russula** confusion.
   - Improving the recall for **Cortinarius**.

3. **Architecture Improvements**
   - Introducing intermediate layers in the model architecture.
   - Adding special pathways for problematic classes.

4. **Multiple Optimizer Comparison**
   - Experimenting with different optimizers: **AdamW**, **SGD with Nesterov momentum**, and **RAdam**.

**Data Source**:

- The dataset used in this project is sourced from Kaggle: [Mushrooms Classification: Common Genus Images](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images)

**Previous Model Results**:

- **Overall Accuracy**: 88%
- **Areas for Improvement**:
  - **Suillus** class had an F1-score of 0.73.
  - Notable confusion between **Lactarius** and **Russula** classes.
  - **Cortinarius** class had a recall of 77%.

---

## Table of Contents

- [Project Overview](#project-overview)
- [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
- [2. Imports and Setup](#2-imports-and-setup)
- [3. Data Transforms](#3-data-transforms)
- [4. Enhanced Model Architecture](#4-enhanced-model-architecture)
- [5. Training and Evaluation Functions](#5-training-and-evaluation-functions)
- [6. Main Execution](#6-main-execution)
- [7. Results Analysis](#7-results-analysis)
- [8. Conclusion and Recommendations](#8-conclusion-and-recommendations)
- [9. How to Run](#9-how-to-run)
- [10. Dependencies](#10-dependencies)
- [11. Acknowledgments](#11-acknowledgments)

---

## 1. Exploratory Data Analysis (EDA)

The EDA aims to understand the dataset's structure, identify potential issues, and inform feature selection for the model.

### 1.1 Dataset Structure and Distribution

- **Class Distribution**: Significant class imbalance was observed.
  - **Most Represented Classes**: *Lactarius* and *Boletus*.
  - **Least Represented Classes**: *Agaricus* and *Suillus*.
  - **Action**: Consider data augmentation for underrepresented classes.

- **Image Dimensions**: Varied image sizes centered around common dimensions.
  - **Action**: Standardize image dimensions during preprocessing.

### 1.2 Visual Inspection

- **Sample Images**: Displayed samples from each genus for visual understanding.
  - **Observation**: Each genus exhibits distinct color and structural patterns.
  - **Action**: Utilize these patterns for feature extraction.

### 1.3 Feature Extraction and Analysis

#### 1.3.1 Color Features

- **Mean Color Analysis**: Calculated average BGR values for each image.
  - **Findings**:
    - **Blue and Green Channels**: Showed significant differences across genera.
    - **Red Channel**: Less discriminative.
  - **Action**: Focus on informative color channels during feature selection.

#### 1.3.2 Texture Features

- **Local Binary Patterns (LBP)**: Extracted texture descriptors from grayscale images.
  - **Findings**:
    - Variability in texture patterns across genera.
    - Overlaps suggest additional features may be beneficial.
  - **Action**: Consider combining LBP with other texture descriptors.

### 1.4 Statistical Analyses

- **ANOVA Tests**: Conducted to assess mean differences in color channels.
  - **Results**:
    - **Blue and Green Channels**: Significant differences (p < 0.001).
    - **Red Channel**: No significant difference (p = 0.597).
  - **Action**: Emphasize blue and green channels in modeling.

- **Chi-Square Tests**: Evaluated the independence between genera and color distributions.
  - **Result**: Color patterns are non-randomly distributed among genera (p < 0.05).
  - **Action**: Reinforce the inclusion of color features.

- **MANOVA**: Assessed multivariate differences in combined color and texture features.
  - **Result**: Significant differences across genera (p < 0.001).
  - **Action**: Use combined features for improved classification.

### 1.5 Outlier Detection

- **Isolation Forest**: Detected potential outliers in the dataset.
  - **Findings**: 45 outliers identified.
  - **Action**: Review these samples for data quality issues.

### 1.6 Key EDA Takeaways

- **Dataset Challenges**:
  - Class imbalance.
  - Feature overlap.

- **Dataset Strengths**:
  - Distinctive color and texture features.
  - Sufficient discriminative information for modeling.

- **Recommendations**:
  - Implement data augmentation for underrepresented classes.
  - Focus on blue and green color channels.
  - Combine color and texture features in the model.

---

## 2. Imports and Setup

This section covers the initial setup and imports required for the project.

### 2.1 Basic Dependencies

- **Core Libraries**: `os`, `time`, `multiprocessing`, `pathlib`.
- **Data Manipulation and Visualization**: `numpy`, `matplotlib`, `seaborn`.

### 2.2 Deep Learning Framework

- **PyTorch**: For model development.
- **PyTorch Lightning**: To simplify training loops.
- **Torchmetrics**: For performance metrics.
- **Pre-trained Models**: Using `ResNet34` from `torchvision.models`.

### 2.3 Data Augmentation

- **Albumentations**: For advanced image transformations and augmentations.

### 2.4 System Setup

- **Windows Compatibility**: Ensuring multiprocessing works on Windows systems.
- **CUDA and Tensor Core Configuration**: Leveraging GPU acceleration if available.
- **Checkpoint Directory Management**: Setting up directories for saving model checkpoints.

```python
# Imports and basic setup
import os
import time
import multiprocessing
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics
from torchvision import models
from torchvision.models import ResNet34_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import confusion_matrix, classification_report

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom modules
from config import Config
from improved_mushroom_dataset import MushroomDataset
from improved_mushroom_data_module import MushroomDataModule

# Multiprocessing setup for Windows
multiprocessing.set_start_method('spawn', force=True)

# CUDA and Tensor Core setup
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')
    logger.info("Enabled Tensor Cores with medium precision")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Checkpoint directory setup
checkpoint_dir = Path('checkpoints')
if checkpoint_dir.exists():
    import shutil
    shutil.rmtree(checkpoint_dir)
checkpoint_dir.mkdir(exist_ok=True)
```

---

## 3. Data Transforms

This section defines the image transformation pipelines for training and validation datasets using Albumentations.

### 3.1 Training Transforms

- **Resizing**: Standardizing image dimensions.
- **Data Augmentation**:
  - Horizontal and vertical flips.
  - Random rotations up to 30 degrees.
  - Brightness and contrast adjustments.
  - Color jittering and hue saturation changes.
  - Random resized cropping.
- **Normalization**: Using ImageNet mean and standard deviation.

### 3.2 Validation Transforms

- **Resizing**: Standardizing image dimensions.
- **Normalization**: Consistent normalization without augmentations.

```python
train_transforms = A.Compose([
    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ColorJitter(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.RandomResizedCrop(Config.IMG_SIZE, Config.IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

---

## 4. Enhanced Model Architecture

This section introduces the enhanced neural network model built using PyTorch Lightning.

### 4.1 Model Components

- **ResNet34 Backbone**: Pre-trained on ImageNet for feature extraction.
- **Custom Intermediate Layers**:
  - Additional fully connected layers for better feature learning.
- **Special Pathways for Problem Classes**:
  - Separate pathways for **Suillus** and **Lactarius-Russula** classes to handle their specific features.
- **Final Classifier**: Combines features from all pathways to make the final prediction.

### 4.2 Key Methods

- **Forward Pass**: Defines how data flows through the model.
- **Training Step**: Includes loss calculation and metric updates during training.
- **Validation Step**: Evaluates the model on the validation set.
- **Optimizer Configuration**: Supports multiple optimizers with learning rate schedulers.

### 4.3 Special Features

- **Class Weighting**: Addresses class imbalance by weighting the loss function.
- **Multiple Optimizer Support**: Easily switch between optimizers for experimentation.
- **Custom Metrics Tracking**: Tracks accuracy and F1-scores, especially for the **Suillus** class.

```python
class EnhancedMushroomClassifier(pl.LightningModule):
    def __init__(self, num_classes=Config.NUM_CLASSES, learning_rate=Config.LEARNING_RATE,
                 class_weights=None, optimizer_name='adamw', dropout_rate=0.5):
        super().__init__()
        
        # Model initialization code...
```

*(Full code available in the repository)*

---

## 5. Training and Evaluation Functions

Utility functions to train the model, evaluate performance, and visualize results.

### 5.1 Model Evaluation

- **Confusion Matrix Generation**: Understand misclassifications.
- **Classification Report Creation**: Detailed metrics per class.
- **Performance Metrics Calculation**: Accuracy, precision, recall, and F1-score.

### 5.2 Results Visualization

- **Accuracy Comparison Plots**: Compare overall accuracy across different optimizers.
- **F1-Score Visualization**: Focus on the **Suillus** class performance.
- **Confusion Matrix Heatmaps**: Visual representation of model predictions.

```python
def evaluate_model(model, data_module):
    """Evaluate model performance on validation set and measure inference time"""
    # Evaluation code...

def plot_results(results):
    """Plot accuracy, Suillus F1-score, and average inference time comparisons"""
    # Plotting code...

def plot_confusion_matrices(results):
    """Plot confusion matrices for each optimizer"""
    # Plotting code...
```

---

## 6. Main Execution

This section orchestrates the entire training and evaluation process.

### 6.1 Setup Phase

- **Random Seed Setting**: Ensures reproducibility.
- **Device Configuration**: Utilizes GPU if available.
- **DataModule Initialization**: Prepares data loaders and applies transforms.

### 6.2 Training Loop

- **Optimizer Iteration**: Trains models using different optimizers.
- **Checkpoint Management**: Saves the best model based on validation loss.
- **Error Handling**: Captures and logs any exceptions during training.

### 6.3 Results Collection

- **Performance Metrics Gathering**: Collects metrics after each training session.
- **Results Visualization**: Plots graphs and confusion matrices.
- **Detailed Reporting**: Outputs class-wise performance metrics.

```python
if __name__ == "__main__":
    # Main execution code...
```

---

## 7. Results Analysis

The enhancements led to noticeable improvements in model performance, especially with the **AdamW** optimizer.

### 7.1 Key Findings

1. **Best Performing Optimizer: AdamW**
   - **Overall Accuracy**: 88.61%
   - **Suillus F1-Score**: 0.75
   - **Strong Performance**: Across all classes, notably:
     - **Boletus**: F1-score of 0.9459
     - **Lactarius**: F1-score of 0.9041
     - **Amanita**: F1-score of 0.8963

2. **Poor Performance of Alternative Optimizers**
   - **SGD with Nesterov Momentum**:
     - **Accuracy**: 29.71%
     - **Suillus F1-Score**: 0.1308
     - **Struggled**: Across most classes.
   - **RAdam**:
     - **Accuracy**: 29.41%
     - **Slightly Better Suillus F1-Score**: 0.1667
     - **Best Performance**: On **Hygrocybe** class (F1-score of 0.4598).

3. **Class-wise Analysis (AdamW)**
   - **Strongest Classifications**:
     - **Boletus**: F1-score of 0.9459, Precision: 0.9571, Recall: 0.9349
     - **Lactarius**: F1-score of 0.9041, Precision: 0.9205, Recall: 0.8882
     - **Hygrocybe**: F1-score of 0.9048
   - **Areas for Improvement**:
     - **Suillus**: Despite improvements, still the weakest (F1-score of 0.7500)
     - **Entoloma**: Lower precision (0.7857) leading to an F1-score of 0.8408

4. **Improvements Over Previous Model**
   - **Overall Accuracy Increase**: From 88% to 88.61%
   - **Suillus F1-Score**: Improved from 0.73 to 0.75
   - **Reduced Lactarius-Russula Confusion**: Both classes now have F1-scores above 0.86

---

## 8. Conclusion and Recommendations

### 8.1 Optimizer Selection

- **AdamW** is the preferred optimizer for this task.
- The significant performance gap indicates that AdamW handles the complex architecture and data nuances more effectively.

### 8.2 Further Improvements

- **Focused Augmentation for Suillus Class**: Enhance data diversity to improve model generalization for this class.
- **Architecture Modifications**: Explore network adjustments to improve precision for the **Entoloma** class.
- **Ensemble Methods**: Consider combining multiple models trained with AdamW to boost overall performance.

### 8.3 EDA's Role in Model Improvement

- The EDA provided critical insights that shaped the model's feature selection and preprocessing steps.
- Emphasizing color channels with significant differences (blue and green) and combining them with texture features improved the model's discriminative power.
- Identifying and addressing class imbalance and outliers contributed to better model generalization.

---

## 9. How to Run

### 9.1 Prerequisites

- **Python 3.7 or higher**
- **PyTorch compatible GPU** (optional but recommended)

### 9.2 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/pboric/DL.1.5.git
   cd DL.1.5
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### 9.3 Dataset Preparation

1. **Download the Dataset**

   - Download the dataset from Kaggle: [Mushrooms Classification: Common Genus Images](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images)

2. **Extract the Dataset**

   - Extract the dataset into a directory specified in `Config.DATA_PATH`.

3. **Data Directory Structure**

   - Ensure the dataset directory is structured as follows:

     ```
     data/
       Agaricus/
         image1.jpg
         image2.jpg
         ...
       Amanita/
         image1.jpg
         image2.jpg
         ...
       ...
     ```

### 9.4 Running the Project

1. **Execute the Training Script**

   ```bash
   python main.py
   ```

2. **View Results**

   - Training progress and results will be displayed in the console.
   - Checkpoints and logs will be saved in the `checkpoints` directory.

3. **Reproducing EDA**

   - To reproduce the EDA, run the `eda.py` script (if provided):

     ```bash
     python eda.py
     ```

   - Alternatively, review the Jupyter notebook `EDA.ipynb` for detailed analysis.

---

## 10. Dependencies

- **Python Libraries**:
  - `torch`
  - `pytorch-lightning`
  - `torchmetrics`
  - `torchvision`
  - `albumentations`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `statsmodels`
  - `imblearn`
  - `opencv-python`
  - `Pillow`

- **Others**:
  - Ensure that you have the `config.py`, `improved_mushroom_dataset.py`, and `improved_mushroom_data_module.py` files in your project directory.

---

## 11. Acknowledgments

- **Kaggle Dataset**: [Mushrooms Classification: Common Genus Images](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images) for providing the dataset used in this project.
- **PyTorch and PyTorch Lightning**: For providing powerful frameworks for deep learning.
- **Albumentations**: For efficient and flexible data augmentation.
- **The Mushroom Dataset Contributors**: For making the dataset available for research purposes.

---

**Note**: This project is for educational purposes and is part of an effort to improve machine learning models for image classification tasks.

---

# Contact Information

For any questions or suggestions, please contact:

- **Petar Krešimir Borić**
- **Email**: kresimirpet@gmail.com
- **GitHub**: [pboric](https://github.com/pboric)

---

Thank you for your interest in the Enhanced Mushroom Classification Project!
