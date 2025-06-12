import torch
import torch.nn as nn
from typing import List
from pointcept.models.dbbd.BaseClasses import AggregatorBase


class MLPMaxPoolAggregator(AggregatorBase):
    def __init__(self):
        """
        Applies MLP to the features and then aggregates using max pooling.

        :param input_dim: Number of input feature dimensions.
        """
        super(MLPMaxPoolAggregator, self).__init__()

    def forward(self, features: torch.Tensor, dim=1) -> torch.Tensor:
        """
        Apply MLP followed by max pooling across the specified dimension.

        :param features: Input tensor of shape (B, N, D) or (N, D)
        :param dim: Dimension to apply max pooling on.
        :return: Aggregated tensor.
        """
        # Pass through MLP
        processed = self.mlp(features)

        # Max pooling
        aggregated_feature, _ = torch.max(processed, dim=dim)
        return aggregated_feature

    def update_feature_dim(self, input_dim: int, output_dim: int) -> None:
        """
        Set the input and output dimension of the MLP used before pooling.
        """
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(), # Explore activation functions (LeakyReLU,etc.)
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(), # Explore activation functions (LeakyReLU,etc.)
            nn.Linear(128, output_dim),
            # nn.LayerNorm(output_dim) # Optional but can help for stability ?
        )
    
    @property
    def method_name(self) -> str:
        return 'mlp_max_pool_aggregator'
