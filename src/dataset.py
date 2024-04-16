import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# ShapeNetCore Dataser
class ObjectsPointCloudDataset(Dataset):
    """Shape net core dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.file.iloc[idx, 0])
        image = io.imread(img_name)
        points = self.file.iloc[idx, 1:]
        points = np.array([points], dtype=float).reshape(-1, 3)
        sample = {'image': image, 'points': points}

        if self.transform:
            sample = self.transform(sample)

        return sample

# dataset = ObjectsPointCloudDataset(csv_file='/home/flavio/Scrivania/3D-reconstruction-from-single-image/dataset.csv',
#                                    root_dir='/home/flavio/Documenti/Datasets/ShapeNetCore/')
# fig = plt.figure()
# for i, sample in enumerate(dataset):
#     # print(i, sample['image'].shape, sample['points'].shape)
#     if i==0:
#         print(sample['points'])
    
    