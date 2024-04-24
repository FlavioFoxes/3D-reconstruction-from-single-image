import torch
import torch.nn as nn
from basic_layers import ResidualBlock

class Network(nn.Module):
    def __init__(self, input_channels):
        super(Network, self).__init__()
        # Calculate padding dynamically for 'same' padding
        self.padding1 = nn.ZeroPad2d((2, 2, 2, 2))  # Padding for conv1 to maintain spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=2)
        # conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')

        # ATTENTION MODULE (must be put in basic_layers)
        # Now it's here to check the shapes
        self.res1 = ResidualBlock(input_channels = 32, output_channels = 32)
        
        self.max1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.res2 = ResidualBlock(input_channels = 32, output_channels=32)

        self.skip = ResidualBlock(input_channels=32, output_channels=32)

        self.max2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.res3 = ResidualBlock(input_channels = 32, output_channels=32)

    def forward(self, x):
        x = self.padding1(x)  # Apply padding for conv1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res1(x)

        x = self.max1(x)
        x = self.res2(x)

        # Here skip has shape [1, 32, 56, 56]
        skip = self.skip(x)
        
        x = self.max2(x)
        x = self.res3(x)
        # Here the x has shape [1, 32, 28, 28]
        
        print("Final shape of x:    ", x.shape)
        return x



model = Network(4)
input_tensor = torch.randn(1, 4, 224, 224)  # Example input tensor with batch size 16 and image size 224x224
output_tensor = model(input_tensor)
