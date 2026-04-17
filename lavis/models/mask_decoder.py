# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

class LayerNorm2d(nn.Module):
    """From Detectron2/ConvNeXt"""
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_mask_tokens: int = 1,
            mask_spatial_size: int = 224,
            hyper_in_dim: int = 22,
    ) -> None:
        """
        Predicts masks given image embeddings using a transformer architecture.
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.mask_spatial_size = mask_spatial_size

        # Initialize mask tokens
        self.mask_tokens = nn.Parameter(torch.zeros(1, num_mask_tokens, transformer_dim))
        self.mask_tokens.data.normal_(mean=0.0, std=0.02)

        # Upscaling and hypernetwork MLPs
        self.output_upscaling_mlp = MLP(
            transformer_dim, transformer_dim, mask_spatial_size ** 2, 3
        )
        self.output_hypernetworks_mlps = MLP(
            transformer_dim, transformer_dim, hyper_in_dim, 3
        )

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
          image_embeddings (torch.Tensor): [batch_size, seq_len, transformer_dim]
        Returns:
          torch.Tensor: predicted masks [batch_size, mask_spatial_size, mask_spatial_size]
        """
        # Expand tokens to batch size
        batch_size = image_embeddings.size(0)
        tokens = self.mask_tokens.expand(batch_size, -1, -1)

        # Run the transformer
        # Assuming transformer returns (hidden_states, updated_src)
        hs, src = self.transformer(image_embeddings, image_embeddings, tokens)

        # Upscale and project
        upscaled_embedding = self.output_upscaling_mlp(src)
        hyper_in = self.output_hypernetworks_mlps(hs)
        
        # Matrix multiplication to get masks and reshape
        b, _, _ = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding).view(b, self.mask_spatial_size, self.mask_spatial_size)

        return masks

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)  # Updated from F.sigmoid
        return x