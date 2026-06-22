"""Graph Convolutional Network classifier for facial expressions.

Takes a flat vector of 113 face-landmark 2D coordinates (226 features) and
predicts the corresponding robot routine ID. The graph topology is encoded
by a learnable, symmetrically-normalized adjacency matrix shared across the
batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FacialExpressionGCN(nn.Module):

    def __init__(self, input_size: int = 226, num_classes: int = 17,
                 dropout_rate: float = 0.3) -> None:
        super().__init__()

        self.num_nodes = input_size // 2
        self.in_features = 2

        self.adj = nn.Parameter(
            torch.eye(self.num_nodes)
            + torch.ones(self.num_nodes, self.num_nodes) / self.num_nodes
        )

        self.gcn1 = nn.Linear(self.in_features, 32)
        self.gcn2 = nn.Linear(32, 64)

        self.fc1 = nn.Linear(self.num_nodes * 64, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalization: D^(-1/2) · A · D^(-1/2)."""
        degree = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat = torch.diag(d_inv_sqrt)
        return torch.mm(torch.mm(d_mat, adj), d_mat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = x.view(batch_size, self.num_nodes, self.in_features)

        adj_norm = self._normalize_adj(self.adj)

        x = self.gcn1(x)
        x = torch.matmul(adj_norm, x)
        x = F.relu(x)

        x = self.gcn2(x)
        x = torch.matmul(adj_norm, x)
        x = F.relu(x)

        x = x.view(batch_size, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)
