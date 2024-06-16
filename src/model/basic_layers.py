import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten, sigmoid
import torch.optim as optim

"""
Residual Block

Arguments:
        input_channels:     number of channels of the input.
        output_channels:    number of channels of the output of the block.
"""
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock, self).__init__()
        self.batch1 = nn.BatchNorm2d(num_features=input_channels)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, padding='same')
        self.batch2 = nn.BatchNorm2d(num_features=output_channels)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding='same')
        
        self.conv_res = None
        if input_channels != output_channels:
            self.conv_res = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1)
        
    def forward(self, x):
        residual = x
        x = self.batch1(x)
        x = F.leaky_relu(x)
        x = self.conv1(x)
        x = self.batch2(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        if self.conv_res is not None:
            residual = self.conv_res(residual)
        x = x + residual
        return x

"""
Residual Block

Arguments:
        input_channels:     number of channels of the input.
        output_channels:    number of channels of the output of the block.
"""
class AttentionModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(AttentionModule, self).__init__()
        self.res1 = ResidualBlock(input_channels = input_channels, output_channels = output_channels)
        
        # SOFT MASK branch
        self.max1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.res2 = ResidualBlock(input_channels = output_channels, output_channels=output_channels)

        self.skip = ResidualBlock(input_channels = output_channels, output_channels=output_channels)

        self.max2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.res3 = ResidualBlock(input_channels = output_channels, output_channels=output_channels)

        self.res4 = ResidualBlock(input_channels = output_channels, output_channels=output_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.res5 = ResidualBlock(input_channels=output_channels, output_channels=output_channels)

        self.conv1 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=1)

        # TRUNK branch
        self.res1_trunk = ResidualBlock(input_channels = output_channels, output_channels=output_channels)
        self.res2_trunk = ResidualBlock(input_channels = output_channels, output_channels=output_channels)

        self.res_final = ResidualBlock(input_channels = output_channels, output_channels=output_channels)

    def forward(self, x):

        x = self.res1(x)

        # TRUNK branch
        trunk_x = self.res1_trunk(x)
        trunk_x = self.res2_trunk(trunk_x)

        # SOFT MASK branch
        ## Downsampling
        x = self.max1(x)
        x = self.res2(x)
        ## Here skip has shape [B, 32, 56, 56]
        skip = self.skip(x)

        x = self.max2(x)
        x = self.res3(x)
        ## Here the x has shape [B, 32, 28, 28]
        
        ## Upsampling
        x = self.res4(x)
        x = self.up1(x)

        x = x + skip

        x = self.res5(x)
        x = self.up2(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = sigmoid(x)

        # Union of the two branches
        x = x * trunk_x
        x = x + trunk_x
        x = self.res_final(x)

        return x



