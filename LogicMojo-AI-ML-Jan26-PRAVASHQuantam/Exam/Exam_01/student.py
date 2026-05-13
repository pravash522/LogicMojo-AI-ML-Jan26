from __future__ import annotations

from typing import Iterable

import pandas as pd
import torch
from torch import nn


def build_customer_product_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build customer-level product interaction features.

    Required input columns:
    customer_id, product_id, event_type, price, quantity, is_returned
    """
    raise NotImplementedError


def top_products_by_revenue(events: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Return top products by non-returned purchase revenue.

    Output columns:
    product_id, non_returned_units, gross_revenue, unique_buyers, return_rate
    """
    raise NotImplementedError


def make_balanced_sampler_weights(labels: torch.Tensor) -> torch.Tensor:
    """Return one sampling weight per label using inverse class frequency.

    Use this formula for class c:
        total_samples / (num_classes * count_of_class_c)
    """
    counts = torch.bincount(labels)
    num_classes = len(counts)
    total_samples = len(labels)
    balance_sampler_weights = total_samples / (num_classes * counts.float())
    return balance_sampler_weights[labels]



def conv2d_output_shape(
    input_hw: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    """Return output height and width for a 2D convolution."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    input_h, input_w = input_hw
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    output_h = ((input_h + 2 * padding_h - dilation_h * (kernel_h - 1)- 1)// stride_h) + 1
    output_w = ((input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w) + 1
    return (output_h, output_w)


class TinyCnn(nn.Module):
    """Small CNN for 28x28 grayscale image classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
) -> dict[str, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    model.to(device)
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
    
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    
    return {"loss": avg_loss, "accuracy": avg_accuracy}

def confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return a num_classes x num_classes confusion matrix.

    Rows are true labels. Columns are predicted labels.
    """
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in zip(y_true, y_pred):
        cm[x.item(), y.item()] += 1
    return cm


def macro_f1_from_confusion(cm: torch.Tensor) -> float:
    """Compute macro F1 from a confusion matrix.

    Rows are true labels. Columns are predicted labels.
    """
    num_classes = cm.shape[0]
    f1_scores = []
    
    for class_idx in range(num_classes):
        tp = cm[class_idx, class_idx].item()
        fp = cm[:, class_idx].sum().item() - tp
        fn = cm[class_idx, :].sum().item() - tp

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        f1_scores.append(f1)
    
    macro_f1 = sum(f1_scores) / num_classes
    
    return macro_f1
