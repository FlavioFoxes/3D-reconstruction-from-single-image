import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock, self).__init__()
        self.batch1 = nn.BatchNorm2d(num_features=input_channels)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, padding='same')
        self.batch2 = nn.BatchNorm2d(num_features=output_channels)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding='same')
        
        

    def forward(self, x):
        residual = x
        x = self.batch1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.conv2(x)
        if residual.shape[1] != x.shape[1]:
            conv_res = nn.Conv2d(in_channels=residual.shape[1], out_channels=x.shape[1], kernel_size=1)
            residual = conv_res(residual)
        x = x + residual
        return x

# TODO: prova a runnare il codice man mano, stampando le dimensioni che escono dai 
#       layer in modo da indovinarli qui

class DownsamplingBlock(nn.Module):
    def __init__(self):
        super(self).__init__()


class DownsamplingBlock(nn.Module):
    def __init__(self, input_channels, shap):
        super(self).__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding='same')
        self.res = ResidualBlock()
class UpsamplingBlock(nn.Module):
    def __init__(self):
        super(self).__init__()


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
        self.up1 = nn.Upsample(size=(56, 56))

        self.res5 = ResidualBlock(input_channels=output_channels, output_channels=output_channels)
        self.up2 = nn.Upsample(size=(112,112))

        self.conv = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=1)

        # TRUNK branch
        self.res1_trunk = ResidualBlock(input_channels = output_channels, output_channels=output_channels)
        self.res2_trunk = ResidualBlock(input_channels = output_channels, output_channels=output_channels)

        self.res_final = ResidualBlock(input_channels = output_channels, output_channels=output_channels)

    def forward(self, x):
        x = self.res1(x)

        # TRUNK branch
        trunk_x = self.res1_trunk(x)
        trunk_x = self.res2_trunk(x)

        # SOFT MASK branch
        x = self.max1(x)
        x = self.res2(x)
        ## Here skip has shape [1, 32, 56, 56]
        skip = self.skip(x)
        x = self.max2(x)
        x = self.res3(x)
        ## Here the x has shape [1, 32, 28, 28]
        x = self.res4(x)
        x = self.up1(x)
        x = x + skip
        x = self.res5(x)
        x = self.up2(x)
        x = self.conv(x)
        x = self.conv(x)
        x = F.sigmoid(x)

        # Union of the two branches
        x = torch.matmul(x, trunk_x)
        x = x + trunk_x
        x = self.res_final(x)

        return x


        
        

