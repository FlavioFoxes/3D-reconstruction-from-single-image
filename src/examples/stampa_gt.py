import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
column_names = ['Ignore'] + [f'Point_{i}_{axis}' for i in range(1024) for axis in ('x', 'y', 'z')]

df = pd.read_csv('/home/flavio/Scrivania/dataset_ordered_two_classes.csv', header=None, names=column_names)

# Drop the first column
data = df.iloc[3610, 1:]

# Convert the DataFrame to a numpy array
data_array = data.values

# Reshape the data to separate x, y, z coordinates
# The shape will be (2, 3072) -> (2, 1024, 3)
data_array = data_array.reshape(1, 1024, 3)

# Select the first row (index 0) for plotting
points = data_array[0]

# Extract x, y, z coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z)
ax.set_aspect('equal', adjustable='box')

# Set labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Show the plot
plt.show()
