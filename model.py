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

        # Changing the Classifier
        self.model.classifier = nn.Sequential(nn.Linear(1408,512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.4),
                                  nn.Linear(512,128),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.4),
                                  nn.Linear(128,nclasses))

        # Making the Classifier layer Trainable                           
        for param in self.model.classifier.parameters():
          param.requires_grad = True

    def forward(self, x):
        # Forward pass through the network
        return self.model(x)