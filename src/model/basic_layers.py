import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten, sigmoid
import torch.optim as optim


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
        # self.up1 = nn.Upsample(size=(56, 56))   # this is defined inside the forward function to guarantee 
                                                  # the right size for the upsampling

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.res5 = ResidualBlock(input_channels=output_channels, output_channels=output_channels)
        # self.up2 = nn.Upsample(size=(112,112))  # this is defined inside the forward function to guarantee 
                                                  # the right size for the upsampling

        self.conv1 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=1)

        # TRUNK branch
        self.res1_trunk = ResidualBlock(input_channels = output_channels, output_channels=output_channels)
        self.res2_trunk = ResidualBlock(input_channels = output_channels, output_channels=output_channels)

        self.res_final = ResidualBlock(input_channels = output_channels, output_channels=output_channels)

    def forward(self, x):
        up2_shape = x.shape

        x = self.res1(x)

        # TRUNK branch
        trunk_x = self.res1_trunk(x)
        trunk_x = self.res2_trunk(trunk_x)

        # SOFT MASK branch
        ## Downsampling
        x = self.max1(x)
        x = self.res2(x)
        ## Here skip has shape [1, 32, 56, 56]
        skip = self.skip(x)
        up1_shape = skip.shape

        x = self.max2(x)
        x = self.res3(x)
        ## Here the x has shape [1, 32, 28, 28]
        
        ## Upsampling
        x = self.res4(x)
        # self.up1 = nn.Upsample(size=(up1_shape[2], up1_shape[3]))
        x = self.up1(x)

        x = x + skip

        x = self.res5(x)
        # self.up2 = nn.Upsample(size=(up2_shape[2], up2_shape[3]))
        x = self.up2(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = sigmoid(x)

        # Union of the two branches
        # x = torch.matmul(x, trunk_x)
        x = x * trunk_x
        x = x + trunk_x
        x = self.res_final(x)

        return x


if __name__ == "__main__":

    # m = ResidualBlock(input_channels = 4, output_channels=8)
    # input_tensor = torch.randn(1,4,10,10)
    # target_tensor = torch.randn(1,8,8,8)


    m = AttentionModule(input_channels=128, output_channels=128)
    input_tensor = torch.randn(1, 128, 55, 55)
    target_tensor = torch.randn(1, 128, 55, 55)
    criterion  = nn.MSELoss()

    optimizer = optim.Adam(m.parameters(), lr=0.01)

    m.train()  # Set the model to training mode
    # DEBUG
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")
    i = 0
    # DEBUG
    # Salva i pesi prima dell'ottimizzazione
    before_update = {}
    for name, param in m.named_parameters():
        before_update[name] = param.clone().detach()

    optimizer.zero_grad()  # Zero the parameter gradients

    # DEBUG
    # print("Images is on CUDA:", images.is_cuda)
    # print("points is on CUDA:", point_clouds.is_cuda)
    # check_model_device(model)

    outputs = m(input_tensor)  # Forward pass

    loss = criterion(outputs, target_tensor)  # Compute loss

    loss.backward()  # Backward pass

    # DEBUG
    # Stampa dei gradienti per verificare che non siano nulli
    # check_gradients(m)


    optimizer.step()  # Optimize the model

    # DEBUG
    # Confronta i pesi prima e dopo l'ottimizzazione
    # print_weight_updates(model, before_update)

    # loss_list.append(float(loss))

    print(f'---Running Loss: {float(loss):.4f}')  # Print running loss



