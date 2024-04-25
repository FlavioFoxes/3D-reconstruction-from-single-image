import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_layers import ResidualBlock, AttentionModule

class Network(nn.Module):
    def __init__(self, input_channels):
        super(Network, self).__init__()
        # Calculate padding dynamically for 'same' padding
        self.padding1 = nn.ZeroPad2d((2, 2, 2, 2))  # Padding for conv1 to maintain spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=2)
        # conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')

        # ATTENTION MODULE (must be put in basic_layers)
        # Now it's here to check the shapes
        # self.res1 = ResidualBlock(input_channels = 64, output_channels = 64)
        
        # self.max1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # self.res2 = ResidualBlock(input_channels = 64, output_channels=64)

        # self.skip = ResidualBlock(input_channels = 64, output_channels=64)

        # self.max2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # self.res3 = ResidualBlock(input_channels = 64, output_channels=64)

        # self.res4 = ResidualBlock(input_channels= 64, output_channels=64)
        # self.up1 = nn.Upsample(size=(56, 56))

        # self.res5 = ResidualBlock(input_channels=64, output_channels=64)
        # self.up2 = nn.Upsample(size=(112,112))

        # self.conv_a1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.attention1 = AttentionModule(input_channels= 64, output_channels=64)


    def forward(self, x):
        x = self.padding1(x)  # Apply padding for conv1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # # Here x has shape [1, 64, 112, 112]
        
        x = self.attention1(x)
        # # Here x has shape [1, 64, 112, 112]
        
        print("Final shape of x:    ", x.shape)
        return x



model = Network(4)
input_tensor = torch.randn(1, 4, 224, 224)  # Example input tensor with batch size 16 and image size 224x224
output_tensor = model(input_tensor)
