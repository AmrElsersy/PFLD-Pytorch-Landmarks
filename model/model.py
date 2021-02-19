"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: PFLD & Auxiliary Models for landmarks detection & head pose estimation
"""
import torch
import torch.nn as nn

import sys
sys.path.insert(1, 'model')

from DepthSepConv import DepthSepConvBlock
from BottleneckResidual import BottleneckResidualBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def ConvRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.ReLU(inplace=True)
    )

class PFLD(nn.Module):
    def __init__(self, device_cpu = False):
        super(PFLD, self).__init__()
        self.device = torch.device('cpu') if device_cpu else device

        self.conv = ConvBlock(in_channels=3, out_channels=64, stride=2, padding=1)
        self.depth_wise_conv = DepthSepConvBlock(in_channels=64, out_channels=64).to(device)

        # 1 Bottlenck Non-Resiudal(as stride =2) used for reducing tensor dim size
        self.bottleneck_1_first = BottleneckResidualBlock(64, 64, expand_factor=2, stride=2).to(device)
        # 4 Bottleneck Residual Blocks with the same in/out channel size
        self.bottleneck_1 = nn.ModuleList([BottleneckResidualBlock(64, 64, expand_factor=2, stride=1).to(device) for i in range(3)])
        self.bottleneck_1_last = BottleneckResidualBlock(64, 64, expand_factor=2, stride=1).to(device)

        # 1 Bottleneck to expand channel size
        self.bottleneck_2 = BottleneckResidualBlock(64, 128, expand_factor=2, stride=2).to(device)        
        
        # 6 Bottleneck Resiudal Blocks with the same in/out channel size
        self.bottleneck_3 = nn.ModuleList([BottleneckResidualBlock(128,128, expand_factor=4, stride=1).to(device) for i in range(6)])
        self.bottleneck_3[0].use_residual_component = False

        # last Bottleneck to reduce channel size
        self.bottleneck_4 = BottleneckResidualBlock(128, 16, expand_factor=2, stride=1).to(device) #16x 14x14

        # last layers S1 & S2 & S3 used together as input to the head as a multi scale features
        self.conv1 = ConvBlock(in_channels=16, out_channels=32, stride=2, padding=1) # 16x 14x14 -> 32x 7x7
        self.conv2 = ConvRelu(in_channels=32, out_channels=128, kernel_size=7) # 32x 7x7 -> 128x 1x1 

        # avg pooling is used to flatten the output of the last conv layers
        self.avg_pool1 = nn.AvgPool2d(kernel_size=14)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=7)
        # input = 16(flatten of bottleneck4) + 32(flatten of conv1) + 128(flatten of conv2)
        self.fc = nn.Linear(176, 196)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.depth_wise_conv(x)
        
        # ======== bottleneck 1 ========
        x = self.bottleneck_1_first(x) 
        for block1 in self.bottleneck_1:
            x = block1(x)

        # Auxiliary Network takes that branch as its input
        features_auxiliary = self.bottleneck_1_last(x)
        
        # ======== bottleneck 2 ========
        x = self.bottleneck_2(features_auxiliary)

        # bottleneck 3
        for block3 in self.bottleneck_3:
            x = block3(x)

        # ======== bottleneck 4 ========
        x = self.bottleneck_4(x)

        # ======== S1 & S2 & S3 ========
        s1 = self.avg_pool1(x)
        s1 = s1.view(s1.shape[0], -1)

        x = self.conv1(x)
        s2 = self.avg_pool2(x)
        s2 = s2.view(s2.shape[0], -1)

        s3 = self.conv2(x)
        s3 = s3.view(s3.shape[0], -1)

        # 176 = 16 + 32 + 128 of s1 + s2 + s3 concatination
        multi_scale_features = torch.cat([s1,s2,s3], dim=1)
        landmarks = self.fc(multi_scale_features)
        return features_auxiliary, landmarks


class AuxiliaryNet(nn.Module):
    """
        Head Pose Estimation Net
        Euler Angles Regression
    """
    def __init__(self):
        super(AuxiliaryNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64,  out_channels= 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=32,  kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32,  out_channels=128,  kernel_size=7, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        """
            Computes Euler angles
            Parameters:
                x: shape = 64 channel x  28x28            
            Returns:
                tensor(3,1) euler angles
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool(x)
        # Flatten
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    auxiliary = AuxiliaryNet().to(device)
    pfld = PFLD().to(device)

    x = torch.randn((10, 3,112,112)).to(device)
    print("x shape:",x.shape)
    features, landmarks = pfld(x)
    print("features:",features.shape)
    print("landmarks:",landmarks.shape)

    euler_angles = auxiliary(features)
    print("euler_angles", euler_angles.shape)

