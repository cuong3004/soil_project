# from torchvision.models import mobilenet_v3_small
import torch.nn as nn
import torch
import os


def get_encoder():
    os.chdir("/content/MobileViTv3-PyTorch/MobileViTv3-v1")
    model_path = '/content/Models/MobileViTv3-v1/results_classification/mobilevitv3_XS_e300_7671/checkpoint_ema_best.pt'
    model = torch.load('/content/MobileViTv3-PyTorch/MobileViTv3-v1/model_structure.pt', weights_only=False)
    model_weights = torch.load(model_path, map_location="cpu")
    model.load_state_dict(model_weights)
    os.chdir("/content/soil_project/soil")
    model.classifier = nn.Identity()
    
    model = nn.Sequential(
        model,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1)  # Giữ batch dim
    )
    
    model(torch.ones((1,3,256,256), dtype=torch.float32))

    return model
