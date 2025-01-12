import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from models.segmentation_model import SimpleSegmentationModel
from datasets.segmentation_dataset import MySegmentationDataset
from utils.io import load_config, save_checkpoint
from utils.metrics import calculate_IoU, calculate_precision, calculate_pixel_accuracy
from utils.train_utils import create_weighted_sampler


def main(config_path):
    # Load Configuration
    config = load_config(config_path)

    # Paths and Parameters
    train_data_dir = config["data"]["train_data_dir"]
    train_seg_dir = config["data"]["train_seg_dir"]
    val_data_dir = config["data"]["val_data_dir"]
    val_seg_dir = config["data"]["val_seg_dir"]
    weights_csv = config["data"]["weights_csv"]
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["optimizer"]["lr"]
    num_classes = config["model"]["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    train_dataset = MySegmentationDataset(
        image_paths=sorted(os.listdir(train_data_dir)),
        mask_paths=sorted(os.listdir(train_seg_dir)),
        transform=None  # Add your transform logic
    )
    val_dataset = MySegmentationDataset(
        image_paths=sorted(os.listdir(val_data_dir)),
        mask_paths=sorted(os.listdir(val_seg_dir)),
        transform=None  # Add your transform logic
    )

    sampler = create_weighted_sampler(weights_csv)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = SimpleSegmentationModel(num_classes=num_classes).to(device)

    # Load pre-trained weights if available
    if config["model"].get("pretrained_weights"):
        checkpoint = torch.load(config["model"]["pretrained_weights"], map_location=device)
        model.load_state_dict(checkpoint["model"])

    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"])

    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1} Training Loss: {running_loss / len(train_loader):.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        iou_scores = []
        precision_scores = []
        pixel_accuracies = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

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

        avg_val_loss = val_loss / len(val_loader)
        avg_mIoU = np.mean(iou_scores)
        avg_precision = np.mean(precision_scores)
        avg_pixel_accuracy = np.mean(pixel_accuracies)

        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch + 1} mIoU: {avg_mIoU:.4f}")
        print(f"Epoch {epoch + 1} Precision: {avg_precision:.4f}")
        print(f"Epoch {epoch + 1} Pixel Accuracy: {avg_pixel_accuracy:.4f}")

        # Save Checkpoint
        save_checkpoint(
            {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            os.path.join(output_dir, f"model_epoch_{epoch + 1}.pth")
        )
        scheduler.step()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    main(args.config)
