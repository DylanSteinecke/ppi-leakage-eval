"""Optional PyTorch modules for end-to-end PPI prediction."""

from __future__ import annotations

import torch
from torch import nn


class TorchSymmetricPairComposer(nn.Module):
    """Compose exchange-invariant protein-pair representations."""

    expansion_factor = 3

    def forward(
            self, protein_representations: torch.Tensor,
            protein_a_indices: torch.Tensor,
            protein_b_indices: torch.Tensor,
        ) -> torch.Tensor:
        if protein_representations.ndim != 2:
            raise ValueError("Protein representations must be a 2D tensor.")
        protein_a_indices = protein_a_indices.to(
            device=protein_representations.device,
            dtype=torch.long,
        ).reshape(-1)
        protein_b_indices = protein_b_indices.to(
            device=protein_representations.device,
            dtype=torch.long,
        ).reshape(-1)
        if protein_a_indices.shape != protein_b_indices.shape:
            raise ValueError("PPI endpoint index tensors must have equal shape.")
        if (
            torch.any(protein_a_indices < 0)
            or torch.any(protein_b_indices < 0)
            or torch.any(protein_a_indices >= protein_representations.shape[0])
            or torch.any(protein_b_indices >= protein_representations.shape[0])
        ):
            raise ValueError("PPI endpoint index is out of bounds.")
        protein_a = protein_representations[protein_a_indices]
        protein_b = protein_representations[protein_b_indices]
        return torch.cat((
            protein_a + protein_b,
            torch.abs(protein_a - protein_b),
            protein_a * protein_b,
        ), dim=-1)


class PPIPairHead(nn.Module):
    """Symmetric pair composition followed by a small binary head."""

    def __init__(
            self, representation_dim: int, hidden_dim: int = 64,
            dropout: float = 0.1,
        ):
        super().__init__()
        if representation_dim < 1 or hidden_dim < 1:
            raise ValueError("PPI head dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("PPI head dropout must be in [0, 1).")
        self.composer = TorchSymmetricPairComposer()
        self.classifier = nn.Sequential(
            nn.Linear(
                representation_dim * self.composer.expansion_factor,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
            self, protein_representations: torch.Tensor,
            protein_a_indices: torch.Tensor,
            protein_b_indices: torch.Tensor,
        ) -> torch.Tensor:
        pair_representations = self.composer(
            protein_representations,
            protein_a_indices,
            protein_b_indices,
        )
        return self.classifier(pair_representations).squeeze(-1)
