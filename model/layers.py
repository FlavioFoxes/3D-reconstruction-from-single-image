import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_layers import ResidualBlock, AttentionModule

class Network(nn.Module):
    def __init__(self, input_channels):
        super(Network, self).__init__()
        # ENCODER

        # Calculate padding dynamically for 'same' padding
        self.padding1 = nn.ZeroPad2d((2, 2, 2, 2))  # Padding for conv1 to maintain spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')

        ##################
        self.attention1 = AttentionModule(input_channels= 64, output_channels=64)

        self.res1 = ResidualBlock(input_channels=64, output_channels=128)
        self.max1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))

        ##################
        self.attention2 = AttentionModule(input_channels=128, output_channels=128)

        self.res2 = ResidualBlock(input_channels=128, output_channels=256)
        self.max2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))

        ##################
        self.attention3 = AttentionModule(input_channels=256, output_channels=256)

        self.res3 = ResidualBlock(input_channels=256, output_channels=512)
        self.max3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))

        ##################
        self.res4 = ResidualBlock(input_channels=512, output_channels=512)
        self.res5 = ResidualBlock(input_channels=512, output_channels=512)
        self.res6 = ResidualBlock(input_channels=512, output_channels=512)

    def forward(self, x):
        # ENCODER
        x = self.padding1(x)  # Apply padding for conv1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # # Here x has shape [1, 64, 112, 112]
        
        x = self.attention1(x)
        # # Here x has shape [1, 64, 112, 112]

        x = self.res1(x)
        x = self.max1(x)
        x1 = x
        # # Here x has shape [1, 128, 55, 55]

        x = self.attention2(x)
        # # Here x has shape [1, 128, 55, 55]

        x = self.res2(x)
        x = self.max2(x)
        x2 = x
        # # Here x has shape [1, 256, 27, 27]

        x = self.res3(x)
        x = self.max3(x)
        x3 = x
        # # Here x has shape [1, 512, 13, 13]

        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        # # Here x has shape [1, 512, 13, 13]

        # DECODER


        print("Final shape of x:    ", x.shape)
        return x



model = Network(4)
input_tensor = torch.randn(1, 4, 224, 224)  # Example input tensor with batch size 16 and image size 224x224
output_tensor = model(input_tensor)
