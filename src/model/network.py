import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_layers import ResidualBlock, AttentionModule

class Network(nn.Module):
    def __init__(self, input_channels):
        super(Network, self).__init__()
        # ENCODER

        # Calculate padding dynamically for 'same' padding
        # # INPUT = [1, 4, 224, 224]
        self.conv0 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=5)
        # # Here x has shape [1, 8, 232, 232]
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=5)
        # # Here x has shape [1, 16, 240, 240]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=5)
        # # Here x has shape [1, 32, 248, 248]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=5)
        # # Here x has shape [1, 64, 256, 256]
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        # # Here x has shape [1, 64, 128, 128]
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

   

        ##################
        self.attention1 = AttentionModule(input_channels= 64, output_channels=64)

        self.res1 = ResidualBlock(input_channels=64, output_channels=128)
        self.max1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        ##################
        self.attention2 = AttentionModule(input_channels=128, output_channels=128)

        self.res2 = ResidualBlock(input_channels=128, output_channels=256)
        self.max2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        ##################
        self.attention3 = AttentionModule(input_channels=256, output_channels=256)

        self.res3 = ResidualBlock(input_channels=256, output_channels=512)
        self.max3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        ##################
        self.res4 = ResidualBlock(input_channels=512, output_channels=512)
        self.res5 = ResidualBlock(input_channels=512, output_channels=512)
        self.res6 = ResidualBlock(input_channels=512, output_channels=512)

        # DECODER
        self.conv_d1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2)
        self.deconv_d1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=6, stride=2)
        self.conv_x3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv_d2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.deconv_d2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv_x2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_d3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv_d3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv_x1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv_d4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_final = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # ENCODER
        # x = self.padding1(x)  # Apply padding for conv1
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # # Here x has shape [1, 64, 128, 128]
        
        x = self.attention1(x)
        x = self.res1(x)
        x = self.max1(x)
        x1 = x
        # # # Here x has shape [1, 128, 64, 64]

        x = self.attention2(x)
        x = self.res2(x)
        x = self.max2(x)
        x2 = x
        # # # Here x has shape [1, 256, 32, 32]

        x = self.res3(x)
        x = self.max3(x)
        x3 = x
        # # # Here x has shape [1, 512, 16, 16]

        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        # # # Here x has shape [1, 512, 16, 16]
        print("Final ENCODER shape:    ", x.shape)
        

        #################################
        # DECODER

        x = self.conv_d1(x)
        x = F.relu(x)
        x = self.deconv_d1(x)
        x3 = self.conv_x3(x3)
        x = F.relu(x + x3)
        # # # Here x has shape [1, 256, 16, 16]


        x = self.conv_d2(x)
        x = F.relu(x)
        x = self.deconv_d2(x)
        x2 = self.conv_x2(x2)
        x = F.relu(x + x2)
        # # # Here x has shape [1, 128, 32, 32]

        x = self.conv_d3(x)
        x = F.relu(x)
        x = self.deconv_d3(x)
        x1 = self.conv_x1(x1)
        x = F.relu(x + x1)
        # # # Here x has shape [1, 64, 64, 64]

        x = self.conv_d4(x)
        x = F.relu(x)
        x = self.conv_final(x)

        x = x.view(x.size(dim=0), 1024, 3)
        print("Final x shape:    ", x.shape)
        
        return x



# model = Network(4)
# input_tensor = torch.randn(1, 4, 224, 224)  # Example input tensor with batch size 16 and image size 224x224
# output_tensor = model(input_tensor)
