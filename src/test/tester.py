import sys
sys.path.append('../')
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.utils.utils import *
from src.dataset.dataset import ObjectsPointCloudDataset
from src.model.network import Network
from src.train.loss import SumOfDistancesLoss
from src.test.evaluate import evaluate



def tester():
    # Load all configuration information
    config = load_config("config.yaml")

    # Load data from CSV
    column_names = ['Ignore'] + [f'Point_{i}_{axis}' for i in range(1024) for axis in ('x', 'y', 'z')]
    data = pd.read_csv(config['csv_path'], header=None, names=column_names)
    
    # Take test data
    _, _, test_data = split_data(data)

    # Trasform data into tensors
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert the PIL image to a PyTorch tensor
    ])

    # Logger for debugging
    writer = SummaryWriter(log_dir=config['logs_path'])


    # Create test dataset objects
    test_dataset = ObjectsPointCloudDataset(test_data, config['dataset_path'], transform=transform)

    # Create test DataLoader objects
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Initialize the model
    model = Network(input_channels=3)
    model.load_state_dict(torch.load(config['load_model']))
    model.eval()
    
    # Move your model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)  
    
    # Define the loss function
    criterion  = SumOfDistancesLoss()
    
    # Evaluate the model on Test set
    test_loss = evaluate(model, test_loader, criterion, device, testing=True, writer=writer)

    print(f"Loss - Test set:    {test_loss}")
    writer.flush()
    writer.close()