import torch.nn as nn
import torch
from torchvision import models

def get_resnet(name, pretrained=True, feature_dim=128):
    if name == "ResNet18":
        resnet = models.resnet18(pretrained=pretrained)
        resnet.fc = nn.Linear(resnet.fc.in_features,feature_dim)
    elif name == "ResNet34":
        resnet = models.resnet34(pretrained=pretrained)
        resnet.fc = nn.Linear(resnet.fc.in_features, feature_dim)
    elif name == "ResNet50":
        resnet = models.resnet50(pretrained=pretrained)
        resnet.fc = nn.Linear(resnet.fc.in_features, feature_dim)
    else:
        raise ValueError(f"Invalid model name: {name}")
    return resnet
