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
        self.model = models.efficientnet_v2_s(weights='DEFAULT')
        # Modify the classifier for the number of classes in your dataset
        # Define the layer up to which you want to freeze
        n_freeze = 2

        for i, block in enumerate(self.model.features):
            if i < n_freeze:
                for param in block.parameters():
                    param.requires_grad = False
                
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, nclasses)

   

    def forward(self, x):
        x = self.model(x)
        
        return x
