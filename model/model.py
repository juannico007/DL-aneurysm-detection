from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class AneurysmDetectionModel(nn.Module):
    """A simple 3D CNN model for binary aneurysm classification."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 256), batch_size : int = 64):
        """
        Initialize the model.
        
        Parameters
        ----------
            input_shape: Shape of input volumes (width, height, depth)
        """
        super().__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.layer_activations = []

        self.conv1 = nn.Conv3d(1, 8, 3, padding="same")
        self.pool1 = nn.MaxPool3d(2)
        self.bn1 = nn.GroupNorm(4, 8)

        self.conv2 = nn.Conv3d(8, 16, 3, padding="same")
        self.pool2 = nn.MaxPool3d(2)
        self.bn2 = nn.GroupNorm(8, 16)

        self.conv3 = nn.Conv3d(16, 32, 3, padding="same")
        self.pool3 = nn.MaxPool3d(2)
        self.bn3 = nn.GroupNorm(8, 32)

        self.conv4 = nn.Conv3d(32, 64, 3, padding="same")
        self.pool4 = nn.MaxPool3d(2)
        self.bn4 = nn.GroupNorm(16, 64)

        self.conv5 = nn.Conv3d(64, 128, 3, padding="same")
        self.pool5 = nn.MaxPool3d(2)
        self.bn5 = nn.GroupNorm(32, 128)

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        
        #for m in self.modules():
        #    if isinstance(m, (nn.Conv3d, nn.Linear)):
        #        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        #        if m.bias is not None:
        #            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Returns
        ----------
            Pythorch model
        """
        self.layer_activations = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        self.layer_activations.append(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        self.layer_activations.append(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        self.layer_activations.append(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        self.layer_activations.append(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x)
        self.layer_activations.append(x)
        x = self.pool5(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.fc1(x))
        self.layer_activations.append(x)
        x = self.dropout(x)
        x = self.fc2(x)
        self.layer_activations.append(x)
        # x = torch.sigmoid(self.fc2(x))
        
        
        return x