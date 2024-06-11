import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from src.utils.mesh2point import display_point_cloud
from src.utils.utils import load_config


def plot_ground_truth(idx):
    # Load all configuration information
    config = load_config("config.yaml")
    # Load data from CSV
    column_names = ['Ignore'] + [f'Point_{i}_{axis}' for i in range(1024) for axis in ('x', 'y', 'z')]
    df = pd.read_csv(config['csv_path'], header=None, names=column_names)

    # Take index idx and Drop the first column
    data = df.iloc[idx, 1:]

    # Convert the DataFrame to a numpy array
    data_array = data.values

    # Reshape the data to separate x, y, z coordinates
    # The shape will be (2, 3072) -> (2, 1024, 3)
    data_array = data_array.reshape(1, 1024, 3)

    # Select the first row (index 0) for plotting
    points = data_array[0]
    display_point_cloud(points)

if __name__== "__main__":
    plot_ground_truth(3610)
