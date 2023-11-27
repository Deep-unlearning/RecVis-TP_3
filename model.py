import torch
import torch.nn as nn
import torch.nn.functional as F
# Use pretrained model from torchvision
from torchvision import models
import timm

nclasses = 250

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Load the pre-trained EfficientNet_V2_S model
        self.model = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=True)

    def forward(self, x):
        x = self.model(x)
        
        return x
