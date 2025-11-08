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

        self.conv1 = nn.Conv3d(1, 8, 3, padding="same")
        self.pool1 = nn.MaxPool3d(2)
        self.bn1 = nn.BatchNorm3d(8)

        self.conv2 = nn.Conv3d(8, 16, 3, padding="same")
        self.pool2 = nn.MaxPool3d(2)
        self.bn2 = nn.BatchNorm3d(16)

        self.conv3 = nn.Conv3d(16, 32, 3, padding="same")
        self.pool3 = nn.MaxPool3d(2)
        self.bn3 = nn.BatchNorm3d(32)

        self.conv4 = nn.Conv3d(32, 64, 3, padding="same")
        self.pool4 = nn.MaxPool3d(2)
        self.bn4 = nn.BatchNorm3d(64)
        
        self.conv5 = nn.Conv3d(64, 128, 3, padding="same")
        self.pool5 = nn.MaxPool3d(2)
        self.bn5 = nn.BatchNorm3d(128)

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Returns
        ----------
            Pythorch model
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = self.pool1(x)
        x = self.bn1(x)
        
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = self.pool2(x)
        x = self.bn2(x)
        
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        x = self.pool3(x)
        x = self.bn3(x)
        
        x = F.leaky_relu(self.conv4(x), negative_slope=0.01)
        x = self.pool4(x)
        x = self.bn4(x)
        
        x = F.leaky_relu(self.conv5(x), negative_slope=0.01)
        x = self.pool5(x)
        x = self.bn5(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = torch.sigmoid(self.fc2(x))
        
        return x
    
