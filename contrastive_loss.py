import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, feature1, feature2, label):

        # Euclidean distance between feature vectors
        distance = torch.nn.functional.pairwise_distance(
            feature1,
            feature2
        )

        # Genuine pair (label = 1):
        # minimize the distance
        genuine_loss = label * torch.pow(distance, 2)

        # Forged pair (label = 0):
        # push distance beyond the margin
        forged_loss = (1 - label) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0),
            2
        )

        # Final contrastive loss
        loss = torch.mean(genuine_loss + forged_loss)

        return loss


# Test the loss function
if __name__ == "__main__":

    loss_function = ContrastiveLoss()

    feature1 = torch.randn(1, 128)
    feature2 = torch.randn(1, 128)

    # Test with genuine pair
    label = torch.tensor([1.0])

    loss = loss_function(feature1, feature2, label)

    print("Contrastive Loss:", loss.item())