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
        self.model = timm.create_model('eva02_small_patch14_336.mim_in22k_ft_in1k', pretrained=True)
      
        self.model.head = nn.Linear(384, nclasses)

    def forward(self, x):
        x = self.model(x)
        
        return x
