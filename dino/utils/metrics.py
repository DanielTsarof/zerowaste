import numpy as np

def calculate_IoU(y_true, y_pred, num_classes):
    """Calculate Intersection over Union (IoU) for each class and mean IoU."""
    ious = []
    for cls in range(num_classes):
        true_mask = (y_true == cls).astype(np.uint8)
        pred_mask = (y_pred == cls).astype(np.uint8)

        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()

        if union == 0:
            ious.append(1.0)  # Perfect IoU for empty class
        else:
            ious.append(intersection / union)

    return ious, np.mean(ious)

def calculate_precision(y_true, y_pred, num_classes):
    """Calculate Precision for each class."""
    precisions = []
    for cls in range(num_classes):
        true_mask = (y_true == cls).astype(np.uint8)
        pred_mask = (y_pred == cls).astype(np.uint8)

        tp = np.logical_and(true_mask, pred_mask).sum()
        fp = pred_mask.sum() - tp

        if tp + fp == 0:
            precisions.append(1.0)  # Perfect precision for empty class
        else:
            precisions.append(tp / (tp + fp))

    return precisions, np.mean(precisions)

def calculate_pixel_accuracy(y_true, y_pred):
    """Calculate Pixel Accuracy."""
    correct = (y_true == y_pred).sum()
    total = y_true.size
    return correct / total
