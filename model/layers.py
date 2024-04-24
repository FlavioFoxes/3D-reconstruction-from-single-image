import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_channels):
        super(Network, self).__init__()
        # Calculate padding dynamically for 'same' padding
        self.padding1 = nn.ZeroPad2d((2, 2, 2, 2))  # Padding for conv1 to maintain spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=2)
        # conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')


    def forward(self, x):
        x = self.padding1(x)  # Apply padding for conv1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        print("Final shape of x:    ", x.shape)
        return x



model = Network(4)
input_tensor = torch.randn(1, 4, 224, 224)  # Example input tensor with batch size 16 and image size 224x224
output_tensor = model(input_tensor)
