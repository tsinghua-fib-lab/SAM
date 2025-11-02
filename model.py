import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from stage2_attention import Stage2AttentionFusion


class Model(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 item_dim: int,
                 data_dir: str,
                 max_seq_len: int,
                 num_interest_fusion_layers: int,
                 stage2_num_heads: int,
                 interests_pt_path: Optional[str],
                 dropout_prob: float,
                 interest_dim: int,
                 stage2_dropout: float):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.item_dim = item_dim
        self.max_seq_len = int(max_seq_len)
        self.data_dir = data_dir

        # ID & positional embeddings (lookup)
        self.item_embedding = nn.Embedding(self.num_items, item_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_len, item_dim)
        self.emb_scale = float(item_dim) ** 0.5

        # Load Stage1 interests; build W_Q: interest_dim -> item_dim
        self.user_interest_cls: Dict[Any, torch.Tensor] = {}
        self.interest_dim: int = int(interest_dim)
        self.interest_to_hidden: nn.Linear = nn.Linear(self.interest_dim, self.item_dim)
        self._load_stage1_interests(interests_pt_path)

        # Stage2 multi-head attention fusion
        self.stage2 = Stage2AttentionFusion(
            interest_to_hidden=self.interest_to_hidden,
            hidden_dim=self.item_dim,
            num_heads=int(stage2_num_heads),
            dropout=float(stage2_dropout),
            num_layers=int(num_interest_fusion_layers),
        )

    def _load_stage1_interests(self, interests_pt_path: Optional[str]) -> None:
        data = torch.load(interests_pt_path, map_location='cpu')
        self.user_interest_cls = data

    def _build_generated_interests(self, raw_user_ids: Optional[List[Any]], device: torch.device) -> torch.Tensor:
        """Return [B, K, interest_dim]"""
        mats: List[torch.Tensor] = []
        max_k = 0
        for uid in raw_user_ids:
            mat = self.user_interest_cls.get(uid)
            mats.append(mat)
            max_k = max(max_k, int(mat.size(0)))
        padded_list: List[torch.Tensor] = []
        for mat in mats:
            # mat: [k, interest_dim]
            if mat.size(0) < max_k:
                pad = torch.zeros(max_k - mat.size(0), self.interest_dim, dtype=mat.dtype, device=mat.device)
                mat = torch.cat([mat, pad], dim=0)
            padded_list.append(mat.unsqueeze(0))
        generated_interests = torch.cat(padded_list, dim=0).to(device)  # [B, K, interest_dim]
        return generated_interests

    def forward(self,
                behavior_sequences: torch.Tensor,
                target_items: Optional[torch.Tensor] = None,
                raw_user_ids: Optional[List[Any]] = None,
                seq_mask: Optional[torch.Tensor] = None,
                neg_items: Optional[torch.Tensor] = None
                ) -> Dict[str, Any]:
        batch_size, seq_len = behavior_sequences.shape
        device = behavior_sequences.device

        # Sequence encoding (E_u^(ID) = e_k + p_k)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        item_emb = self.item_embedding(behavior_sequences) * self.emb_scale
        pos_emb = self.position_embedding(position_ids)
        user_features = item_emb + pos_emb

        # Generate interests [B, K, interest_dim]
        generated_interests = self._build_generated_interests(raw_user_ids, device)

        # Stage2 fusion call
        out2 = self.stage2.run(
            generated_interests=generated_interests,
            seq_mask=seq_mask,
            interests_mask=None,
            key_embeddings=user_features,
            value_embeddings=item_emb,
        )
        final_user_representation = out2['final_user_representation']
        interest_views = out2.get('interest_views')
        interest_fused = out2.get('interest_fused')

        # Dot-product scoring + pointwise BCE (one positive, one negative)
        item_table = self.item_embedding.weight  # [N,item_dim]
        recommendation_logits = torch.matmul(final_user_representation, item_table.t())  # [B,N]
        losses: Dict[str, torch.Tensor] = {}
        if target_items is not None and neg_items is not None:
            pos_vec = item_table.index_select(0, target_items.view(-1))  # [B,item_dim]
            neg_vec = item_table.index_select(0, neg_items.view(-1))     # [B,item_dim]
            pos_score = torch.sum(final_user_representation * pos_vec, dim=-1)  # [B]
            neg_score = torch.sum(final_user_representation * neg_vec, dim=-1)  # [B]
            loss = - torch.log(torch.sigmoid(pos_score) + 1e-24) - torch.log(1 - torch.sigmoid(neg_score) + 1e-24)
            loss = loss.mean()
            losses['total_loss'] = loss

        return {
            'recommendation_logits': recommendation_logits,
            'final_user_representation': final_user_representation,
            'interest_views': interest_views,
            'interest_fused': interest_fused,
            'losses': losses,
        }