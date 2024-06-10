import torch
from src.model.network import Network
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append('3D-reconstruction-from-single-image')
from src.utils.utils import *

config = load_config("config.yaml")
print(config['csv_path'])

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# # Read the CSV file
# # column_names = ['Ignore'] + [f'Point_{i}_{axis}' for i in range(1024) for axis in ('x', 'y', 'z')]

# # df = pd.read_csv('/home/flavio/Scrivania/dataset.csv', header=None)
# # image_path = (df.iloc[:,0]).to_string()
# # image_path = os.path.join(PATH,image_path)
# # print(image_path)
# # Load the image
# # Read the CSV file
# column_names = ['Ignore'] + [f'Point_{i}_{axis}' for i in range(1024) for axis in ('x', 'y', 'z')]

# df = pd.read_csv('/home/flavio/Scrivania/dataset_ordered_two_classes.csv', header=None, names=column_names)

# path = df.iloc[3610, 0]
# data = df.iloc[3610, 1:]

# image_path = '/home/flavio/Documenti/Datasets/ShapeNetCore_TwoClasses/'+path
# image = Image.open(image_path).convert('RGB')

# # Apply the transformations
# input_tensor = transform(image)

# # # Add a batch dimension (as models expect a batch of images)
# input_tensor = input_tensor.unsqueeze(0)

# checkpoint = torch.load('checkpoint.pt')

# model = Network(3)
# model.load_state_dict(torch.load('model_state.pth'))
# model.eval()  # Set the model to evaluation mode

# with torch.no_grad():
#     # Pass the input tensor through the model to get predictions
#     output = model(input_tensor)
#     output = output.squeeze(0)
#     print(output)
#     array = output.numpy()
#     x = array[:, 0]
#     y = array[:, 1]
#     z = array[:, 2]

#     # Create a 3D scatter plot using matplotlib
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x, y, z)
#     ax.set_aspect('equal', adjustable='box')

#     # Set labels
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')

#     # Show the plot
#     plt.show()

# # Output is a tensor containing the model's predictions
# # print("Model output:")
# # print(output)

