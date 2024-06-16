import sys
sys.path.append('/home/flavio/Scrivania/3D-reconstruction-from-single-image/src')
from utils.utils import load_config
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, transform
import os
from utils.utils import load_config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from PIL import Image
from dataset.dataset import ObjectsPointCloudDataset
from model.network import Network
from torchvision import transforms
import torch.nn.init as init
from loss import SumOfDistancesLoss
from torch.utils.tensorboard import SummaryWriter
from train import train

# Load data from CSV
column_names = ['Ignore'] + [f'Point_{i}_{axis}' for i in range(1024) for axis in ('x', 'y', 'z')]
data = pd.read_csv('/home/flavio/Scrivania/dataset_two_classes.csv', header=None, names=column_names)
# print(train_data)
# Split the data into train and test sets
# Split the test set into test and eval sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
test_data, eval_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Trasform data into tensors
transform = transforms.Compose([
    transforms.ToTensor()  # Convert the PIL image to a PyTorch tensor
])

# Logger for debugging
writer = SummaryWriter()


# Create dataset objects
train_dataset = ObjectsPointCloudDataset(train_data, '/home/flavio/Documenti/Datasets/ShapeNetCore_TwoClasses', transform=transform)
eval_dataset = ObjectsPointCloudDataset(eval_data, '/home/flavio/Documenti/Datasets/ShapeNetCore_TwoClasses', transform=transform)
test_dataset = ObjectsPointCloudDataset(test_data, '/home/flavio/Documenti/Datasets/ShapeNetCore_TwoClasses', transform=transform)

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Initialize the model
model = Network(input_channels=3)
model.load_state_dict(torch.load('model_state.pth'))
model.train()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)  # Move your model to GPU if available
# model.apply(init_weights)

# Define the loss function and optimizer
criterion  = SumOfDistancesLoss()
weight_decay = 1e-6
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=weight_decay)

# Training
num_epochs = 20

total_loss = train(model, train_loader, eval_loader, optimizer, criterion, device, num_epochs, writer)
writer.flush()
writer.close()
torch.save(model.state_dict(), 'model_state_2.pth')
