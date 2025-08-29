from prepare_data import Food101DataModule, CustomFood101, get_model_components
from models import EffNetV2_S , EffNetb2
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping ,ModelCheckpoint
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List

DATA_DIR = "data"
MODEL_NAME = "EfficientNet_V2_S"
BATCH_SIZE = 32
SUBSET_FRACTION = 0.2 # Useing a smaller subset for quick testing
CHECKPOINT_PATH = "checkpoints/best-model-epoch=22-val_acc=0.8541.ckpt"  # Path to your trained model checkpoint

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], figsize: tuple = (25, 25)):
    """
    Creates and saves a multi-class confusion matrix plot.

    This function normalizes the confusion matrix to show prediction
    percentages for each class, visualizes it as a heatmap, and saves
    the resulting figure to a file.

    Args:
        cm (np.ndarray): The confusion matrix from torchmetrics or scikit-learn.
        class_names (List[str]): A list of class names for the labels.
        figsize (tuple, optional): The size of the figure. Defaults to (25, 25).
    """
    # 1. Normalize the confusion matrix to show percentages
    # Add a small epsilon to prevent division by zero
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    # 2. Create a DataFrame for a beautiful plot with labels
    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)

    # 3. Create the plot
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=False, cmap='Blues') # Annotations off for 101 classes

    # 4. Format the plot
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=8)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()

    # 5. Save the figure and show the plot
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Confusion matrix plot saved to confusion_matrix.png")
    plt.show()

def run_training_session(
    model_name: str = "EfficientNet_V2_S",
    batch_size: int = 32,
    data_dir: str = 'data',
    subset_fraction: float = 1.0,
    checkpoint_path: str = "checkpoints/",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    freeze_features: bool = True,
    early_stopping_patience: int = 5,
    max_epochs: int = 100,
    accelerator: str = 'auto',
    resume_from_checkpoint: Optional[str] = None
) -> Trainer:
    """
    Sets up and runs a complete training session for a specified model.

    This function handles the entire pipeline: data preparation, model
    instantiation, logger and callback setup, and trainer execution.

    Args:
        model_name (str): The name of the model architecture to train.
        batch_size (int): The number of samples per batch.
        data_dir (str): The root directory for the dataset.
        subset_fraction (float): The fraction of the dataset to use for training.
        checkpoint_path (str): Directory to save model checkpoints.
        lr (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        freeze_features (bool): Flag to control the fine-tuning strategy
            (e.g., for two-stage training).
        early_stopping_patience (int): Number of epochs with no improvement
            after which training will be stopped.
        max_epochs (int): The maximum number of epochs to train for.
        accelerator (str): The hardware accelerator to use ('auto', 'cpu', 'gpu').
        resume_from_checkpoint (Optional[str]): Path to a checkpoint file to
            resume training from. Defaults to None.

    Returns:
        Trainer: The PyTorch Lightning Trainer object after fitting is complete.
    """
    # A registry to map model names to their actual classes
    model_class_registry = {
        "EfficientNet_V2_S": EffNetV2_S,
        "EfficientNet_B2": EffNetb2,
    }
    if model_name not in model_class_registry:
        raise ValueError(f"Model '{model_name}' is not a recognized class.")

    # Get model-specific transforms
    components = get_model_components(model_name)
    train_transforms = components["train_transforms"]
    val_transforms = components["val_transforms"]

    # Set up the DataModule
    food_datamodule = Food101DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        subset_fraction=subset_fraction
    )
    food_datamodule.prepare_data()
    food_datamodule.setup()

    # Instantiate the model dynamically
    model_class = model_class_registry[model_name]
    model = model_class(
        num_classes=len(food_datamodule.classes),
        class_names=food_datamodule.classes,
        lr=lr,
        weight_decay=weight_decay,
        freeze_features=freeze_features
    )

    # Set up logger and callbacks
    logger = CSVLogger(save_dir="logs/", name=model_name)
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min"
    )
    best_model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="best-model-{epoch:02d}-{val_acc:.4f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max"
    )
    
    callbacks = [early_stop_callback, best_model_checkpoint]

    # Instantiate the Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=callbacks,
        logger=logger,
    )

    # Start training
    trainer.fit(
        model,
        datamodule=food_datamodule,
        ckpt_path=resume_from_checkpoint 
    )
    
    return trainer

# ===================================================================
# Main Execution Block
# ===================================================================
if __name__ == "__main__":
    
    # --- 1. DEFINE YOUR TRAINING CONFIGURATION HERE ---
    config = {
        "model_name": "EfficientNet_V2_S",
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 50,
        "subset_fraction": 1.0,  # Use 1.0 for the full dataset
        "freeze_features": True,
        "early_stopping_patience": 10
    }

    # --- 2. PRINT CONFIGURATION AND START TRAINING ---
    print("--- Starting Training Session ---")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("---------------------------------")
    
    run_training_session(
        model_name=config["model_name"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        max_epochs=config["epochs"],
        subset_fraction=config["subset_fraction"],
        freeze_features=config["freeze_features"],
        early_stopping_patience=config["early_stopping_patience"]
    )
    
    print("\n--- Training Session Complete ---")

    print("\n--- Starting Evaluation on Test Set ---")

    print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")

    # Step 1: Set up the DataModule for the test set
    components = get_model_components(MODEL_NAME)
    val_transforms = components["val_transforms"]
    
    datamodule = Food101DataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_transforms=val_transforms
    )
    # This prepares the test dataloader specifically
    datamodule.setup(stage='test')

    # Step 2: Load the trained model from the checkpoint file
    model = EffNetV2_S.load_from_checkpoint(CHECKPOINT_PATH)
    model.class_names = datamodule.classes
    model.eval() # Set the model to evaluation mode

    # Step 3: Create a Trainer and run the test
    trainer = pl.Trainer(accelerator='auto')
    
    # This call will run the test_step and automatically trigger the 
    # on_test_end hook in your model, which generates the plot.
    trainer.test(model, datamodule=datamodule)
    
    print("\nEvaluation complete. The confusion matrix plot has been saved.")