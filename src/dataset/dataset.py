import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# ShapeNetCore Dataset
class ObjectsPointCloudDataset(Dataset):

    """
    Arguments:
        df (DataFrame): DataFrame returned from pd.read_csv(csv_file).
        root_dir (string): Directory with all the images.
        transform: Optional transform to be applied on a sample.
    """
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    # x: Path to the image
    # y: Ground Truth point cloud
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        points = self.df.iloc[idx, 1:-1].values
        points = np.array([points], dtype=np.float32).reshape(-1, 3)

        return image, points


    