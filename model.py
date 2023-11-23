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
        self.efficientnet = models.efficientnet_v2_m(weights='DEFAULT')

        # Get the number of input features of the original classifier
        num_features = self.efficientnet.classifier[1].in_features

        # Add dropout before the final fully connected layer
        self.dropout = nn.Dropout(0.2)  # You can adjust the dropout rate as needed

        # Replace the classifier with a new one (dropout + linear layer)
        self.efficientnet.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(num_features, nclasses)
        )

    def forward(self, x):
        # Pass x through the EfficientNet model
        x = self.efficientnet(x)
        return x