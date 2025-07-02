from torchvision.models import mobilenet_v3_small
import torch.nn as nn

def get_encoder():
    encoder = mobilenet_v3_small(pretrained=False)
    encoder.classifier = nn.Identity()
    return nn.Sequential(encoder, nn.Flatten())
