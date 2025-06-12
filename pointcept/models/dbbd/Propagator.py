import torch
from torch import nn
from pointcept.models.dbbd.BaseClasses import FeaturePropagationBase


class ConcatPropagation(FeaturePropagationBase):
    def __init__(self):
        super(ConcatPropagation, self).__init__()

    def propagate(self, parent_feature: torch.Tensor, current_feature: torch.Tensor) -> torch.Tensor:
        if parent_feature is not None:
            # Concatenate along the feature dimension
            # parent_feature: shape = [96] [output_dim]
            # current_feature: shape = [5000, 3] [min_num_points_per_pointcloud, D]
            # shape: [500, 96] [min_num_points_per_pointcloud, output_dim]
            repeated_vector = parent_feature.unsqueeze(0).repeat(current_feature.size(0), 1)
            original_dim = 3  # Original feature size (before padding)
            #shape: [500, 3]
            current_feature = current_feature[:, :original_dim]  # Slice only the original features

            # Concatenate along the second dimension
            # shape: [500, 99] [min_num_points_per_pointcloud, (output_dim + D)]
            combined_feature = torch.cat((current_feature, repeated_vector), dim=1)
            
            # shape: [500, 96] [min_num_points_per_pointcloud, D]
            # Pass through MLP
            combined_feature = self.mlp(combined_feature)

        else:
            combined_feature = current_feature
        return combined_feature

    def update_feature_dim(self, input_dim: int, feature_dim: int) -> None:
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
            nn.Linear(128, feature_dim),
            # nn.LayerNorm(feature_dim)  # Optional but can help for stability ?
        )

    @property
    def method_name(self) -> str:
        return 'concat_propagation'