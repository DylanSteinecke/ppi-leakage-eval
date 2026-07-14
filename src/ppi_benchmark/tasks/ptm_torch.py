"""Optional PyTorch head for residue-level PTM prediction."""

from __future__ import annotations

import torch
from torch import nn


class PTMResidueHead(nn.Module):
    """Gather one aligned residue representation and classify it."""

    def __init__(self, representation_dim: int, output_dim: int = 1):
        super().__init__()
        if representation_dim < 1 or output_dim < 1:
            raise ValueError("PTM head dimensions must be positive.")
        self.classifier = nn.Linear(representation_dim, output_dim)

    def forward(
            self, token_representations: torch.Tensor,
            target_token_indices: torch.Tensor,
        ) -> torch.Tensor:
        if token_representations.ndim != 3:
            raise ValueError("PTM token representations must be 3D.")
        indices = target_token_indices.to(
            device=token_representations.device,
            dtype=torch.long,
        ).reshape(-1)
        if len(indices) != token_representations.shape[0]:
            raise ValueError("PTM target indices must have one row per batch.")
        if torch.any(indices < 0) or torch.any(
            indices >= token_representations.shape[1]
        ):
            raise ValueError("PTM target token index is out of bounds.")
        batch_indices = torch.arange(
            token_representations.shape[0],
            device=token_representations.device,
        )
        target_representations = token_representations[
            batch_indices,
            indices,
        ]
        logits = self.classifier(target_representations)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits
