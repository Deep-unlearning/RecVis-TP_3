import torch
import torch.nn as nn
import torch.nn.functional as F
# Use pretrained model from torchvision
from torchvision import models

nclasses = 250

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Use pretrained ImageNet model
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        Freeze layers up to layer2
        for name, param in self.resnet50.named_parameters():
            if "layer1" in name:
                param.requires_grad = False
            else:
                break  # Exit the loop after layer2


        # Replace last layer
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, nclasses)


    def forward(self, x):
        x = self.resnet50(x)
        # add dropout layer
        x = F.dropout(x, p=0.2, training=self.training)
        # add softmax activation layer
        x = F.log_softmax(x, dim=1)
        return x
