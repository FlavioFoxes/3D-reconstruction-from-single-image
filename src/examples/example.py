import sys
sys.path.append('../../')
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from plotly.subplots import make_subplots
from PIL import Image
import plotly.graph_objs as go

from src.model.network import Network
from src.utils.mesh2point import display_point_cloud
from src.utils.utils import load_config

# Given the index idx, for the sample idx-th it plots both
# the Ground Truth (left) and the Prediction of the model(right) 
def plot_example(idx):
    
    # Load all configuration information
    config = load_config("config.yaml")
    # Load data from CSV
    column_names = ['Ignore'] + [f'Point_{i}_{axis}' for i in range(1024) for axis in ('x', 'y', 'z')]
    df = pd.read_csv(config['csv_path'], header=None, names=column_names)

    path = df.iloc[idx, 0]
    data = df.iloc[idx, 1:]

    # Convert the DataFrame to a numpy array
    data_array = data.values

    # Reshape the data to separate x, y, z coordinates
    # The shape will be (2, 3072) -> (2, 1024, 3)
    data_array = data_array.reshape(1, 1024, 3)

    # Ground Truth points
    points_gt = data_array[0]
    trace_gt = display_point_cloud(points_gt)

    # Image for model prediction
    image_path = config['dataset_path']+path
    image = Image.open(image_path).convert('RGB')

    # Apply the transformations
    transform = transforms.Compose([
      transforms.ToTensor()
    ])
    input_tensor = transform(image)

    # Add a batch dimension (as models expect a batch of images)
    input_tensor = input_tensor.unsqueeze(0)

    # Load the model and set it to evaluation mode
    model = Network(3)
    model.load_state_dict(torch.load(config["load_model"]))
    model.eval()  


    # Model prediction points
    with torch.no_grad():
        output = model(input_tensor)
        output = output.squeeze(0)

        points_pred = output.numpy()
        trace_pred = display_point_cloud(points_pred)


    # Make two plots: one for the ground truth, one for the prediction
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]], subplot_titles=["Ground Truth", "Model Prediction"])
    fig.append_trace(trace_gt, row=1, col=1)
    fig.append_trace(trace_pred, row=1, col=2)
    fig.show()


if __name__== "__main__":
    plot_example(44)