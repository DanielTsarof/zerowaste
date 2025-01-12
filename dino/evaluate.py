import os
import torch
import numpy as np
from tqdm.auto import tqdm
from models.segmentation_model import SimpleSegmentationModel
from datasets.segmentation_dataset import MySegmentationDataset
from utils.io import load_config
from utils.metrics import calculate_IoU, calculate_precision, calculate_pixel_accuracy

def evaluate(config_path, checkpoint_path):
    # Load configuration
    config = load_config(config_path)

    # Paths and parameters
    val_data_dir = config["data"]["val_data_dir"]
    val_seg_dir = config["data"]["val_seg_dir"]
    num_classes = config["model"]["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    val_dataset = MySegmentationDataset(
        image_paths=sorted(os.listdir(val_data_dir)),
        mask_paths=sorted(os.listdir(val_seg_dir)),
        transform=None  # Add your transform logic
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load model
    model = SimpleSegmentationModel(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Metrics initialization
    iou_scores = []
    precision_scores = []
    pixel_accuracies = []

    # Evaluation loop
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Compute metrics
            y_true = masks.cpu().numpy()
            y_pred = preds.cpu().numpy()

            _, mean_iou = calculate_IoU(y_true, y_pred, num_classes=num_classes)
            _, mean_precision = calculate_precision(y_true, y_pred, num_classes=num_classes)
            pixel_acc = calculate_pixel_accuracy(y_true, y_pred)

            iou_scores.append(mean_iou)
            precision_scores.append(mean_precision)
            pixel_accuracies.append(pixel_acc)

    # Calculate average metrics
    avg_mIoU = np.mean(iou_scores)
    avg_precision = np.mean(precision_scores)
    avg_pixel_accuracy = np.mean(pixel_accuracies)

    print(f"mIoU: {avg_mIoU:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Pixel Accuracy: {avg_pixel_accuracy:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a segmentation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint)
