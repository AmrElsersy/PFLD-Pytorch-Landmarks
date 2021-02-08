import torch
import torch.nn as nn

class AuxiliaryNet(nn.Module):
    """
        Head Pose Estimation Net
        Euler angles regression
    """
    def __init__(self):
        self.conv1 = nn.Conv2d(in_channels=64,  out_channels= 128, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=32,  kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=32,  out_channels=128,  kernel_size=7, stride=1)

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = nn.Flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)

        return x

