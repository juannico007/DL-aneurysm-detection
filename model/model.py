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
        self.bn1 = nn.InstanceNorm3d(8)

        self.conv2 = nn.Conv3d(8, 16, 3, padding="same")
        self.pool2 = nn.MaxPool3d(2)
        self.bn2 = nn.InstanceNorm3d(16)

        self.conv3 = nn.Conv3d(16, 32, 3, padding="same")
        self.pool3 = nn.MaxPool3d(2)
        self.bn3 = nn.InstanceNorm3d(32)

        self.conv4 = nn.Conv3d(32, 64, 3, padding="same")
        self.pool4 = nn.MaxPool3d(2)
        self.bn4 = nn.InstanceNorm3d(64)
        
        self.conv5 = nn.Conv3d(64, 128, 3, padding="same")
        self.pool5 = nn.MaxPool3d(2)
        self.bn5 = nn.InstanceNorm3d(128)

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
    
class double_conv3D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv3D_block, self).__init__()
        
        self.conv3D = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1 if out_channels <= 32 else 0.2 if out_channels <= 128 else 0.3),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.conv3D(x)
        return x

class up_conv3D_block(nn.Module):
    def __init__(self, in_channels, out_channels, scale_tuple):
        super(up_conv3D_block, self).__init__()
        
        self.up_conv3D = nn.Sequential(
            nn.Upsample(scale_factor=scale_tuple, mode='trilinear'),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up_conv3D(x)
        return x

class AneurysmDetection3DUNetModel(nn.Module):
    """3DUNet model for aneurysm detection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        #Encoder
        self.conv1 = double_conv3D_block(in_channels=in_channels, out_channels=16)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        
        self.conv2 = double_conv3D_block(in_channels=16, out_channels=32)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        
        self.conv3 = double_conv3D_block(in_channels=32, out_channels=64)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        
        self.conv4 = double_conv3D_block(in_channels=64, out_channels=128)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        
        #Bottleneck
        self.conv5 = double_conv3D_block(in_channels=128, out_channels=256)
        
        #Decoder
        self.upconv6 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6 = double_conv3D_block(in_channels=256, out_channels=128)
        
        self.upconv7 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7 = double_conv3D_block(in_channels=128, out_channels=64)
        
        self.upconv8 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8 = double_conv3D_block(in_channels=64, out_channels=32)
        
        self.upconv9 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9 = double_conv3D_block(in_channels=32, out_channels=16)
        
        self.out_conv = nn.Conv3d(in_channels=16, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
 
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
    
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
    
        c5 = self.conv5(p4)
    
        u6 = self.upconv6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(u6)
    
        u7 = self.upconv7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)
    
        u8 = self.upconv8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)
    
        u9 = self.upconv9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)
    
        outputs = self.out_conv(c9)
    
        return outputs