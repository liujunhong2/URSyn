import torch
from torch import nn


class Cosine_loss(nn.Module):
    """Margin loss on the cosine similarity of two batches of representations."""

    def __init__(self, margin=-0.5):
        super(Cosine_loss, self).__init__()
        self.cosine = nn.CosineSimilarity(dim=1)
        self.margin = margin

    def forward(self, x, y):
        # x, y: [B, D]
        similarity_matrix = self.cosine(x, y)
        similarity_loss = torch.mean(torch.relu(similarity_matrix - self.margin))
        return similarity_loss