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
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, nclasses)

    def forward(self, x):
        # Forward pass through the network
        x = self.model(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.softmax(x, dim=1)

        return x