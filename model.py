import torch
import torch.nn as nn
import torch.nn.functional as F
# Use pretrained model from torchvision
from torchvision import models

nclasses = 250

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Load the pre-trained EfficientNet_V2_S model
        self.model = models.efficientnet_b4(weights='DEFAULT')
        # Modify the classifier for the number of classes in your dataset
        # Define the layer up to which you want to freeze
        freeze_until = 4  # For example, freeze up to the end of block 6 in 'features'
        
        # Freeze layers in 'features' up to the specified block
        for name, parameter in self.model.named_parameters():
            if name.split('.')[1].isdigit() and int(name.split('.')[1]) < freeze_until:
                parameter.requires_grad = False
                
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, nclasses)

    def forward(self, x):
        # Forward pass through the network
        x = self.model(x)
        return x
