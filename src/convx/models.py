import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvX(nn.Module):
    """
    ConvX / MultiScaleResNetFCDD Architecture.
    Equalizes spatial features to Stage 2 (28x28 resolution / stride 8).
    """
    def __init__(self, num_classes=20):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.stage1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1)  # 256 ch, 56x56
        self.stage2 = base.layer2  # 512 ch, 28x28
        self.stage3 = base.layer3  # 1024 ch, 14x14
        self.stage4 = base.layer4  # 2048 ch, 7x7

        # Post-Fusion Compression Head (3840 channels -> 512)
        self.compress = nn.Conv2d(3840, 512, kernel_size=1, bias=False)
        self.bn_compress = nn.BatchNorm2d(512)

        self.bottleneck = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.convx_head = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        # Spatial resolution equalization targeting Stage 2 (28x28)
        target_size = s2.shape[2:]

        s1_down = F.interpolate(s1, size=target_size, mode='bilinear', align_corners=False)
        s3_up   = F.interpolate(s3, size=target_size, mode='bilinear', align_corners=False)
        s4_up   = F.interpolate(s4, size=target_size, mode='bilinear', align_corners=False)

        # Feature concatenation (256 + 512 + 1024 + 2048 = 3840 channels)
        fused = torch.cat([s1_down, s2, s3_up, s4_up], dim=1)
        fused_compressed = F.relu(self.bn_compress(self.compress(fused)))
        out = self.convx_head(F.relu(self.bottleneck(fused_compressed)))

        # FCDD Heatmaps (28x28) & Spatial Mean Anomaly Scores
        heatmaps = torch.sqrt(torch.pow(out, 2) + 1) - 1
        scores = heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1).mean(dim=2)

        return scores, heatmaps


class StandardBaseline(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, trace=False):
        features = self.backbone(x)
        pooled = self.avgpool(features)
        out = self.fc(torch.flatten(pooled, 1))
        return out, features