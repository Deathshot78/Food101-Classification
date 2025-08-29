from torchvision import datasets
from pathlib import Path
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms as T
import numpy as np
import torchvision
from torchvision.datasets import Food101
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, Any
import random


def get_model_components(
    model_name: str, 
    return_classifier: bool = False, 
    augmentation_level: str = "default"
) -> Dict[str, Any]:
    """
    Retrieves pre-trained model components from torchvision.

    This function fetches the appropriate weights and transforms for a given
    model. It supports different levels of training data augmentation.

    Args:
        model_name (str): The name of the model to get components for.
            Supported models include "EfficientNet_V2_S" and "EfficientNet_B2".
        return_classifier (bool, optional): If True, the model's classifier
            head is also returned. Defaults to False.
        augmentation_level (str, optional): The level of data augmentation to use
            for the training set. Can be "default" or "strong". 
            Defaults to "default".

    Returns:
        Dict[str, Any]: A dictionary containing the requested components.
            Always includes 'train_transforms' and 'val_transforms'.
            Includes 'classifier' if return_classifier is True.
            
    Raises:
        ValueError: If model_name or augmentation_level is not supported.
    """
    model_registry = {
        "EfficientNet_V2_S": (
            torchvision.models.efficientnet_v2_s,
            torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        ),
        "EfficientNet_B2": (
            torchvision.models.efficientnet_b2,
            torchvision.models.EfficientNet_B2_Weights.DEFAULT
        )
    }

    if model_name not in model_registry:
        raise ValueError(f"Model '{model_name}' is not supported. "
                         f"Supported models are: {list(model_registry.keys())}")

    # 1. Look up the model and weights classes
    model_class, weights_class = model_registry[model_name]
    weights = weights_class
    val_transforms = weights.transforms()

    # 2. Create the training transforms based on the desired level
    if augmentation_level == "default":
        train_transforms = T.Compose([
            T.TrivialAugmentWide(),
            val_transforms  # val_transforms includes ToTensor and Normalize
        ])
    elif augmentation_level == "strong":
        # Note: We don't need to add ToTensor() or Normalize() here because
        # they are already included inside the 'val_transforms' pipeline.
        train_transforms = T.Compose([
            T.RandomResizedCrop(size=val_transforms.crop_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandAugment(num_ops=2, magnitude=9),
            # RandomErasing should be applied to a tensor, so we apply it after
            # val_transforms, which handles the PIL -> Tensor conversion.
            val_transforms, 
            T.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
        ])
    else:
        raise ValueError(f"Augmentation level '{augmentation_level}' is not supported. "
                         f"Choose from 'default' or 'strong'.")
    
    # 3. Prepare the dictionary to be returned
    components = {
        "train_transforms": train_transforms,
        "val_transforms": val_transforms
    }

    # 4. Optionally, instantiate the model to get the classifier
    if return_classifier:
        model = model_class(weights=weights)
        components["classifier"] = model.classifier

    return components
        
class CustomFood101(Dataset):
    """A PyTorch Dataset for Food101 with conditional downloading and subset support.

    This class wraps the torchvision Food101 dataset. It only downloads the data
    if the specified directory doesn't already exist. It can also create a
    reproducible, shuffled subset of the data for faster experimentation.

    Args:
        split (str): The dataset split, either "train" or "test".
        transform (callable, optional): A function/transform to apply to the images.
        data_dir (str, optional): The directory to store the data. Defaults to "data".
        subset_fraction (float, optional): The fraction of the dataset to use.
            Defaults to 1.0 (using the full dataset).
    """

    def __init__(self, split, transform=None, data_dir="data", subset_fraction: float = 0.1):
        # Check if the dataset already exists before setting the download flag.
        dataset_path = os.path.join(data_dir, "food-101")
        should_download = not os.path.isdir(dataset_path)

        # 1. Load the full dataset metadata with the conditional flag
        self.full_dataset = Food101(root=data_dir, split=split, transform=transform, download=should_download)
        self.classes = self.full_dataset.classes

        # 2. Create a reproducible subset of indices
        if subset_fraction < 1.0:
            num_samples = int(len(self.full_dataset) * subset_fraction)
            all_indices = list(range(len(self.full_dataset)))
            # Shuffle with a fixed seed for reproducibility
            random.Random(42).shuffle(all_indices)
            self.indices = all_indices[:num_samples]
        else:
            self.indices = list(range(len(self.full_dataset)))

    def __len__(self):
        """Returns the total number of samples in the subset."""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Fetches the sample for the given subset index and applies the transform.
        """
        # Map the subset index to the actual index in the full dataset
        original_idx = self.indices[idx]
        image, label = self.full_dataset[original_idx]
        return image, label

class Food101DataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for the Food101 dataset.

    This module encapsulates all data-related logic, including downloading,
    processing, and creating DataLoaders for the training, validation, and
    test sets. It uses the CustomFood101 dataset internally and allows for
    controlling the fraction of data used in the training and validation splits.

    Args:
        data_dir (str, optional): Root directory for the data. Defaults to "data".
        batch_size (int, optional): The batch size for DataLoaders. Defaults to 32.
        num_workers (int, optional): Number of workers for data loading. Defaults to 2.
        train_transforms (callable, optional): Transformations for the training set.
        val_transforms (callable, optional): Transformations for the validation/test set.
        subset_fraction (float, optional): The fraction of data to use for training
            and validation. Defaults to 1.0.
    """
    def __init__(self, data_dir="data", batch_size=32, num_workers=2,
                 train_transforms=None, val_transforms=None, subset_fraction: float = 0.5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.subset_fraction = subset_fraction

        self.classes = []

    def prepare_data(self):
        """Downloads data if needed."""
        CustomFood101(split='train', data_dir=self.data_dir)
        CustomFood101(split='test', data_dir=self.data_dir)

    def setup(self, stage=None):
        """Assigns datasets, passing the subset_fraction."""
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomFood101(split='train', transform=self.train_transforms,
                                               data_dir=self.data_dir, subset_fraction=self.subset_fraction)
            self.val_dataset = CustomFood101(split='test', transform=self.val_transforms,
                                             data_dir=self.data_dir, subset_fraction=self.subset_fraction)
            self.classes = self.train_dataset.classes

        if stage == 'test' or stage is None:
            self.test_dataset = CustomFood101(split='test', transform=self.val_transforms,
                                              data_dir=self.data_dir, subset_fraction=1.0) # Use full test set
            if not self.classes:
                self.classes = self.test_dataset.classes

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    

if __name__ == "__main__":
    # Define configuration for the script
    DATA_DIR = "data"
    MODEL_NAME = "EfficientNet_V2_S"
    BATCH_SIZE = 32

    print(f"Running data preparation script for model: {MODEL_NAME}")

    # 1. Get model-specific transforms
    components = get_model_components(MODEL_NAME)
    train_transforms = components["train_transforms"]
    val_transforms = components["val_transforms"]
    
    # 2. Instantiate the DataModule
    datamodule = Food101DataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        subset_fraction=0.1  # Use a small subset for quick verification
    )

    # 3. Trigger download and setup
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    
    # 4. (Optional) Verification Step
    print("\n--- Verifying Dataloader ---")
    # Get one batch from the training dataloader
    train_dl = datamodule.train_dataloader()
    images, labels = next(iter(train_dl))
    
    print(f"Number of classes: {len(datamodule.classes)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print("--- Verification Complete ---")    