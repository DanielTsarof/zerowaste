import torch.nn as nn
import torch.hub

class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Backbone (DINOv2 in this case)
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        # Simple segmentation head
        self.conv1 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)  # Extract features
        x = self.conv1(features)
        x = self.relu(x)
        x = self.conv2(x)
        return x
