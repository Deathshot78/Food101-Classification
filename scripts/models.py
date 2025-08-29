import torch
import torchvision
import pytorch_lightning as pl
from torch import nn
from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class EffNetV2_S(pl.LightningModule):
    """A PyTorch Lightning Module for fine-tuning EfficientNetV2-S.

    This module encapsulates the EfficientNetV2-S model and provides a flexible
    fine-tuning strategy. It can be configured for Stage 1 (training only the
    classifier and later feature blocks) or Stage 2 (training the entire model).

    Args:
        lr (float, optional): The learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-4.
        num_classes (int, optional): The number of output classes. Defaults to 101.
        class_names (list, optional): A list of class names for logging. Defaults to None.
        freeze_features (bool, optional): If True, freezes the backbone and unfreezes
            only the later blocks (Stage 1). If False, all features are trainable
            (Stage 2). Defaults to True.
        unfreeze_from_block (int, optional): Which feature block to start unfreezing
            from. Used only if freeze_features is True. Defaults to -3 (last 3 blocks).
    """
    
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_classes: int = 101,
        class_names: list = None,
        freeze_features: bool = True,         # True = Stage 1, False = Stage 2
        unfreeze_from_block: int = -3          # Only used if freeze_features=True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.class_names = class_names if class_names else [str(i) for i in range(num_classes)]

        # Load pretrained weights
        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_v2_s(weights=weights)

        # ---- Freezing strategy ----
        if freeze_features:
            # Freeze all first
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze from a specific block (default: last 3 blocks)
            for param in self.model.features[unfreeze_from_block:].parameters():
                param.requires_grad = True
        else:
            # Stage 2: unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True

        # Classifier head
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=self.hparams.num_classes, bias=True)
        )

        # Loss & metrics
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.val_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.val_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_accuracy(logits, y)
        self.train_f1(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_accuracy(logits, y)
        self.val_f1(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        self.val_conf_matrix.update(logits, y)

    def on_validation_epoch_end(self):
        cm = self.val_conf_matrix.compute()
        per_class_acc = cm.diag() / (cm.sum(dim=1) + 1e-6)
        print("\n--- Per-Class Validation Accuracy ---")
        for i, acc in enumerate(per_class_acc):
            self.log(f'val_acc/{self.class_names[i]}', acc.item(), on_epoch=True)
            print(f"{self.class_names[i]:<20}: {acc.item():.4f}")
        print("------------------------------------")
        self.val_conf_matrix.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_conf_matrix.update(logits, y)

    def on_test_end(self):
        cm = self.test_conf_matrix.compute()
        print("\nGenerating final confusion matrix plot...")
        self.test_conf_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
    
class EffNetb2(pl.LightningModule):
    """A PyTorch Lightning Module for fine-tuning EfficientNet-B2.

    This module encapsulates the EfficientNet-B2 model and provides a flexible
    fine-tuning strategy. It can be configured for Stage 1 (training only the
    classifier and later feature blocks) or Stage 2 (training the entire model).

    Args:
        lr (float, optional): The learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-4.
        num_classes (int, optional): The number of output classes. Defaults to 101.
        class_names (list, optional): A list of class names for logging. Defaults to None.
        freeze_features (bool, optional): If True, freezes the backbone and unfreezes
            only the later blocks (Stage 1). If False, all features are trainable
            (Stage 2). Defaults to True.
        unfreeze_from_block (int, optional): Which feature block to start unfreezing
            from. Used only if freeze_features is True. Defaults to -3 (last 3 blocks).
    """

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_classes: int = 101,
        class_names: list = None,
        freeze_features: bool = True,
        unfreeze_from_block: int = -3
    ):
        super().__init__()
        self.save_hyperparameters()
        self.class_names = class_names if class_names is not None else [str(i) for i in range(num_classes)]

        # Model setup
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b2(weights=weights)
        
        # --- : Flexible Freezing Strategy ---
        if self.hparams.freeze_features:
            # Stage 1: Freeze all first
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze from a specific block (default: last 3 blocks)
            for param in self.model.features[self.hparams.unfreeze_from_block:].parameters():
                param.requires_grad = True
        else:
            # Stage 2: unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True

        # Classifier head
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1408, out_features=self.hparams.num_classes)
        )

        # Metrics
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.val_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.val_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_accuracy(logits, y)
        self.train_f1(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_accuracy(logits, y)
        self.val_f1(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        self.val_conf_matrix.update(logits, y)

    def on_validation_epoch_end(self):
        cm = self.val_conf_matrix.compute()

        # Add a small epsilon (1e-6) to the denominator for numerical stability.
        per_class_acc = cm.diag() / (cm.sum(dim=1) + 1e-6)

        print("\n--- Per-Class Validation Accuracy ---")
        for i, acc in enumerate(per_class_acc):
            class_name = self.class_names[i]
            self.log(f'val_acc/{class_name}', acc.item(), on_epoch=True)
            print(f"{class_name:<20}: {acc.item():.4f}")
        print("------------------------------------")

        self.val_conf_matrix.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_conf_matrix.update(logits, y)

    def on_test_end(self):
        cm = self.test_conf_matrix.compute()
        print("\nGenerating final confusion matrix plot...")
        # Assuming plot_confusion_matrix is defined elsewhere
        # plot_confusion_matrix(cm.cpu().numpy(), class_names=self.class_names)
        self.test_conf_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
