import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# ShapeNetCore Dataset
class ObjectsPointCloudDataset(Dataset):
    """Shape net core dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Arguments:
            df (DataFrame): DataFrame returned from pd.read_csv(csv_file).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        # image = io.imread(img_name)
        image = Image.open(img_name).convert('RGBA')
        if self.transform:
            image = self.transform(image)
        points = self.df.iloc[idx, 1:].values
        points = points.astype('float32')
        # points = points.reshape(-1, 3) 
        points = np.array([points], dtype=float).reshape(-1, 3)
        # sample = {'image': image, 'points': points}


        return image, points

# df = pd.read_csv('/home/flavio/Scrivania/dataset.csv')
# dataset = ObjectsPointCloudDataset(df=df,
#                                    root_dir='/home/flavio/Documenti/Datasets/ShapeNetCore/')
# # fig = plt.figure()
# for i, sample in enumerate(dataset):
#     # print(i, sample['image'].shape, sample['points'].shape)
#     if i==0:
#         print(sample['points'])
    
    