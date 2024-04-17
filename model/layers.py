import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_channels):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)

        # Calculate padding dynamically for 'same' padding
        self.padding1 = nn.ZeroPad2d((2, 2, 2, 2))  # Padding for conv1 to maintain spatial dimensions
        self.padding2 = nn.ZeroPad2d((1, 1, 1, 1))  # Padding for conv2 to maintain spatial dimensions

    def forward(self, x):
        x = self.padding1(x)  # Apply padding for conv1
        x = self.conv1(x)
        x = self.padding2(x)  # Apply padding for conv2
        x = self.conv2(x)
        print("Shape after conv2:", x.shape)

        return x
