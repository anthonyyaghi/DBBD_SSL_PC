import torch
import torch.nn as nn
from typing import List
from pointcept.models.dbbd.BaseClasses import AggregatorBase


class MLPMaxPoolAggregator(AggregatorBase):
    def __init__(self, input_dim: int = 96, hidden_dim: int = 96):
        """
        Applies MLP to the features and then aggregates using max pooling.

        :param input_dim: Number of input feature dimensions.
        :param hidden_dim: Output feature dimensions after MLP.
        """
        super(MLPMaxPoolAggregator, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor, dim=1) -> torch.Tensor:
        """
        Apply MLP followed by max pooling across the specified dimension.

        :param features: Input tensor of shape (B, N, D) or (N, D)
        :param dim: Dimension to apply max pooling on.
        :return: Aggregated tensor.
        """
        # Apply MLP (Linear + ReLU)
        features = self.linear(features)
        features = self.activation(features)

        # Max pooling
        aggregated_feature, _ = torch.max(features, dim=dim)
        return aggregated_feature

    @property
    def method_name(self) -> str:
        return 'mlp_max_pool_aggregator'
