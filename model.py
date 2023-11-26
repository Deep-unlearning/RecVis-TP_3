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
        self.model = models.models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        # Modify the classifier for the number of classes in your dataset
        # Define the layer up to which you want to freeze
                # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last 'unfreeze_last_n' layers in the encoder
        num_encoder_layers = len(self.model.encoder.layers)
        layers_to_unfreeze = range(num_encoder_layers - self.unfreeze_last_n, num_encoder_layers)
        for layer_idx in layers_to_unfreeze:
            for param in self.model.encoder.layers[layer_idx].parameters():
                param.requires_grad = True

        # Unfreeze the classification head
        for param in self.model.heads.parameters():
            param.requires_grad = True
            
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, nclasses)

    def forward(self, x):
        x = self.model(x)
        
        return x
