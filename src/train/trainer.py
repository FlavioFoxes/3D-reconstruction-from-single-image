import sys
sys.path.append('/home/flavio/Scrivania/3D-reconstruction-from-single-image/')
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.utils.utils import *
from dataset.dataset import ObjectsPointCloudDataset
from src.model.network import Network
from src.train.loss import SumOfDistancesLoss
from src.train.train import train



def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)  # Inizializzazione gaussiana
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)  # Inizializzazione gaussiana
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    
    # Load all configuration information
    config = load_config("config.yaml")

    # Load data from CSV
    column_names = ['Ignore'] + [f'Point_{i}_{axis}' for i in range(1024) for axis in ('x', 'y', 'z')]
    data = pd.read_csv(config['csv_path'], header=None, names=column_names)
    # print(train_data)
    # Split the data into train and test sets
    # Split the test set into test and eval sets
    train_data, eval_data, test_data = split_data(data)

    # Trasform data into tensors
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert the PIL image to a PyTorch tensor
    ])

    # Logger for debugging
    writer = SummaryWriter()


    # Create dataset objects
    train_dataset = ObjectsPointCloudDataset(train_data, config['dataset_path'], transform=transform)
    eval_dataset = ObjectsPointCloudDataset(eval_data, config['dataset_path'], transform=transform)
    test_dataset = ObjectsPointCloudDataset(test_data, config['dataset_path'], transform=transform)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Initialize the model
    model = Network(input_channels=3)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)  # Move your model to GPU if available
    model.apply(init_weights)

    # Define the loss function and optimizer
    # criterion = nn.MSELoss(reduction='sum')  # Mean Squared Error loss
    # criterion = nn.L1Loss()
    criterion  = SumOfDistancesLoss()
    weight_decay = 1e-6
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training
    num_epochs = 50

    total_loss = train(model, train_loader, eval_loader, optimizer, criterion, device, num_epochs, writer)
    writer.flush()
    writer.close()
    torch.save(model.state_dict(), config['save_model'])
