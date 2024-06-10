import torch
import torch.nn as nn

class SumOfDistancesLoss(nn.Module):
    def __init__(self):
        super(SumOfDistancesLoss, self).__init__()

    def forward(self, pred, target):
        """
        Calculate the sum of Euclidean distances between corresponding points.
        :param pred: Predicted point cloud tensor of shape (B, 1024, 3)
        :param target: Target point cloud tensor of shape (B, 1024, 3)
        :return: Sum of distances
        """
        # Compute the Euclidean distance for each pair of corresponding points
        distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))

        # Sum the distances of the points in the same point cloud
        total_distances = torch.sum(distances, dim=1)

        # Mean of the total distances between batches
        mean_distance = torch.mean(total_distances)

        return mean_distance


# Example usage
# pred = torch.randn(2, 4, 3)  # Predicted point cloud
# target = torch.randn(2, 4, 3)  # Ground truth point cloud
# print("pred", pred)
# print("target", target)
# criterion = SumOfDistancesLoss()
# loss = criterion(pred, target)
# print(loss)