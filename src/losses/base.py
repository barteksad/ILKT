import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self, margin=1.0, reduction="mean"):
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - target) * torch.pow(euclidean_distance, 2)
            + (target)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        if self.reduction == "mean":
            return loss_contrastive.mean()
        elif self.reduction == "sum":
            return loss_contrastive.sum()
        else:
            return loss_contrastive

    def __repr__(self):
        return self.__class__.__name__ + "(margin={})".format(self.margin)
