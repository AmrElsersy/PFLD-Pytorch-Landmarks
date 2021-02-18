"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Inverted Resuedial Linear Block used in PFLD backbone
"""
import torch
import torch.nn as nn

class BottleneckResidualBlock(nn.Module):
    """
        Inverted Resuedial Linear Block from MobileNetv2 paper
        Uses Depth wise Conv & Residual & Expant-Squeeze 
    """
    def __init__(self, in_channels, out_channels, expand_factor=2, stride = 1, padding=1, force_residual=False):
        super(BottleneckResidualBlock, self).__init__()

        # residual component is not used in case of stride = 1 & in_n = out_n  
        assert stride in [1,2]
        self.use_residual_component = True if stride == 1 and in_channels == out_channels else False

        # Expantion 1x1 Conv to increase num of channels
        expand_channels = in_channels * expand_factor
        self.expantion_pointwise_conv = nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_channels)
        # ============================= we modified it from ReLU6 to normal ReLU ===================
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        # Depth wise 3x3 Conv 
        self.depth_wise_conv = nn.Conv2d(expand_channels, expand_channels, kernel_size=3, stride=stride, 
                                        groups=expand_channels, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_channels)

        # Squeeze (Projection) 1x1 Conv to reduce n_channels to match the initial number of channels
        self.squeeze_pointwise_conv = nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    """ Notes:
        - if expand factor = 1, we will skip the expantion layer, but in PFLD it is never = 1
        - Bottleneck Residual solves the problem of low quality feature extraction in low resolution (small dim tensors),
          as it expands the dims to a higher resolution then apply depth conv then squeeze it again.
        - it also solves the problem of very deep networks using residual.
        - it also have small size of parameters as it seperate the depth wise conv from point wise
        - it is called inverted residual as it follow the approach of narrow-wide-narrow instead of wide-narrow-wide
        - it is called linear because it has linear activation(identity) at the last layer(squeeze_pointwise)
    """
    def forward_without_res(self, x):
        # expantion 1x1 conv
        x = self.expantion_pointwise_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        # depth wise 3x3 conv
        x = self.depth_wise_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # squeeze 1x1 conv
        x = self.squeeze_pointwise_conv(x)
        x = self.bn3(x)
        return x

    def forward(self, x):
        if self.use_residual_component:
            return x + self.forward_without_res(x)
        else:
            return self.forward_without_res(x)


if __name__ == "__main__":
    bottleneck = BottleneckResidualBlock(4, 4, expand_factor=2, stride=1, padding=1)
    x = torch.randn((2,4,1280,720)) # batch, channels, W, H
    print("input_shape:", x.shape)
    y = bottleneck(x)
    print("output_shape:", y.shape)
