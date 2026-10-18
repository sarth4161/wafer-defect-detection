"""CNN built from scratch (no pretrained weights)."""
import torch.nn as nn


def conv_bn_relu(in_ch, out_ch):
    return [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()]          # NOT inplace -> needed for Grad-CAM hooks


class WaferCNN(nn.Module):
    """
    Input : (B, 2, 64, 64)   channel 0 = die present, channel 1 = die defective
    Output: (B, num_classes)

    64x64 -> 32x32 -> 16x16 -> 8x8 -> global average pool -> linear
    The last layer of `features` is the layer Grad-CAM looks at.
    """

    def __init__(self, num_classes=9, in_channels=2):
        super().__init__()
        self.features = nn.Sequential(
            *conv_bn_relu(in_channels, 32), *conv_bn_relu(32, 32), nn.MaxPool2d(2),   # 32x32
            *conv_bn_relu(32, 64), *conv_bn_relu(64, 64), nn.MaxPool2d(2),            # 16x16
            *conv_bn_relu(64, 128), *conv_bn_relu(128, 128), nn.MaxPool2d(2),         # 8x8
            *conv_bn_relu(128, 128),                                                  # 8x8 (Grad-CAM layer)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
