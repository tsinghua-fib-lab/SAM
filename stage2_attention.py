import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class Stage2AttentionFusion(nn.Module):
    """
    Stage 2: Interest-aware attention (Q=interests; K=semantic item embeddings, V=ID item embeddings).
    Fuse K interests via stacked Transformer blocks:
      Pre-Norm -> Multi-Head Attention -> Residual -> Pre-Norm -> FFN -> Residual
    Inputs:
      - generated_interests: [B, K, Q]
      - interest_to_hidden: project interest Q to ID dim H
      - key_embeddings: [B, L, H]
      - value_embeddings: [B, L, H]
      - seq_mask: [B, L] True for valid positions
      - interests_mask: [B, K] True for valid interests
    Outputs:
      - final_user_representation: [B, H]
      - interest_views: [B, K, H]
      - interest_fused: [B, K, H]
    """

    def __init__(
        self,
        interest_to_hidden: nn.Linear,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.interest_to_hidden = interest_to_hidden
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, int(num_layers))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                'norm2': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(dropout),
            })
            for _ in range(self.num_layers)
        ])

    def run(
        self,
        user_features: Optional[torch.Tensor] = None,
        generated_interests: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        interests_mask: Optional[torch.Tensor] = None,
        key_embeddings: Optional[torch.Tensor] = None,
        value_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        keys = key_embeddings  # [B,L,H]
        values = value_embeddings  # [B,L,H]

        # Q: project interests to H
        q_proj = self.interest_to_hidden(generated_interests)  # [B, K, H]

        # attention weights: Q*K^T
        scores = torch.einsum('bkh,blh->bkl', q_proj, keys) / math.sqrt(self.hidden_dim)
        if seq_mask is not None and seq_mask.dim() == 2:
            mask_bkl = seq_mask[:, None, :].expand(-1, q_proj.size(1), -1)  # [B,K,L]
            scores = scores.masked_fill(~mask_bkl, -1e9)
        attn = torch.softmax(scores, dim=-1)  # [B, K, L]
        views = torch.einsum('bkl,blh->bkh', attn, values)  # [B, K, H]

        # Interest fusion (Transformer over K-dim sequence)
        fused = views
        key_padding_mask = None
        if interests_mask is not None and interests_mask.dim() == 2:
            key_padding_mask = ~interests_mask.bool()  # True means to ignore
        for layer in self.layers:
            q_norm = layer['norm1'](fused)
            attn_out, _ = layer['self_attn'](q_norm, fused, fused, key_padding_mask=key_padding_mask)
            attn_out = layer['dropout'](attn_out)
            x = fused + attn_out
            ffn_in = layer['norm2'](x)
            ffn_out = layer['ffn'](ffn_in)
            ffn_out = layer['dropout'](ffn_out)
            fused = x + ffn_out

        # Aggregate over K to get final user representation
        if interests_mask is not None and interests_mask.dim() == 2:
            denom = interests_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            final_user_representation = (fused * interests_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            final_user_representation = fused.mean(dim=1)

        return {
            'final_user_representation': final_user_representation,
            'interest_views': views,
            'interest_fused': fused,
        } 