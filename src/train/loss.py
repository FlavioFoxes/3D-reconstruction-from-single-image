import torch
import torch.nn as nn

"""
Loss fucntion:
Calculate the sum of Euclidean distances between corresponding points.

Arguments:
        pred: Predicted point cloud tensor of shape (B, 1024, 3)
        target: Target point cloud tensor of shape (B, 1024, 3)
Return: 
        Sum of distances
""" 
class SumOfDistancesLoss(nn.Module):
    def __init__(self):
        super(SumOfDistancesLoss, self).__init__()

    
    def forward(self, pred, target):
        # Compute the Euclidean distance for each pair of corresponding points
        distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))

        # Sum the distances of the points in the same point cloud
        total_distances = torch.sum(distances, dim=1)

        # Mean of the total distances between batches
        mean_distance = torch.mean(total_distances)

        return mean_distance

