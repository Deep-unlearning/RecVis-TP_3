import torch
import torch.nn as nn
import torch.nn.functional as F
# Use pretrained model from torchvision
from torchvision import models

nclasses = 250

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Use pretrained EfficientNet model
        self.efficientnet = models.swin_v2_t(weights='DEFAULT')
        
        # Replace the last layer with a new, untrained one
        num_ftrs = self.efficientnet.head.in_features
        self.efficientnet.head = nn.Linear(num_ftrs, nclasses)



    def forward(self, x):
        # Pass x through the EfficientNet model
        x = self.efficientnet(x)

        return x