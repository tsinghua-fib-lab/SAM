import torch
import numpy as np
import math
import time
from collections import defaultdict
from typing import Dict, List, Any


class Evaluator:
    """Evaluator"""

    def __init__(self, num_items: int):
        self.num_items = num_items

    def evaluate_recommendations(
        self,
        recommendation_scores: torch.Tensor,
        target_items: torch.Tensor,
        k=20,
    ) -> Dict[str, float]:
        batch_size = recommendation_scores.size(0)
        _, top_k_items = torch.topk(recommendation_scores, k, dim=1)
        metrics = {}

        top_k = top_k_items[:, :k]
        hits = torch.any(top_k == target_items.unsqueeze(1), dim=1).float()
        metrics[f'hr@{k}'] = hits.mean().item()
        ndcg_scores = []
        for i in range(batch_size):
            target = target_items[i].item()
            if target in top_k[i]:
                pos = (top_k[i] == target).nonzero(as_tuple=True)[0][0].item()
                ndcg_scores.append(1.0 / np.log2(pos + 2.0))
            else:
                ndcg_scores.append(0.0)
        metrics[f'ndcg@{k}'] = float(np.mean(ndcg_scores))
        # MRR
        mrr_scores = []
        for i in range(batch_size):
            target = target_items[i].item()
            row = top_k_items[i]
            if target in row:
                pos = (row == target).nonzero(as_tuple=True)[0][0].item()
                mrr_scores.append(1.0 / (pos + 1))
            else:
                mrr_scores.append(0.0)
        metrics['mrr'] = float(np.mean(mrr_scores))
        metrics[f'mrr@{k}'] = float(np.mean(mrr_scores))
        return metrics

class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay'),
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma'],
        )
        dataset_info = config.get('dataset_info', {})
        num_items = dataset_info.get('num_items')
        self.evaluator = Evaluator(num_items)

    def safe_item(self, value):
        if isinstance(value, torch.Tensor):
            return value.mean().item() if value.numel() > 1 else value.item()
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def train_epoch(self, epoch_idx: int = -1) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        successful_batches = 0
        total_batches = len(self.train_loader)
        bar_width = 30
        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            behavior_sequences = batch['behavior_sequences']
            target_items = batch['target_items']
            raw_user_ids = batch.get('raw_user_ids') or batch.get('raw_user_id')
            # Negative samples: one per sample; exclude history and the positive item
            with torch.no_grad():
                num_items = getattr(self.model, 'num_items', 0)
                neg_items = torch.randint(low=1, high=max(2, num_items), size=target_items.shape, device=self.device)
                if isinstance(behavior_sequences, torch.Tensor):
                    hist = behavior_sequences
                    for b in range(hist.size(0)):
                        banned = set(hist[b].tolist())
                        banned.add(int(target_items[b].item()))
                        for _ in range(10):
                            if int(neg_items[b].item()) not in banned and int(neg_items[b].item()) != 0:
                                break
                            neg_items[b] = torch.randint(low=1, high=max(2, num_items), size=(1,), device=self.device)
            if behavior_sequences.size(0) == 0:
                continue
            batch_size = behavior_sequences.size(0)
            if target_items.size(0) != batch_size:
                continue
            outputs = self.model(
                behavior_sequences=behavior_sequences,
                target_items=target_items,
                raw_user_ids=raw_user_ids,
                seq_mask=batch.get('seq_mask'),
                neg_items=neg_items,
            )
            losses = outputs.get('losses', {})
            if not losses:
                filled = int((batch_idx + 1) / max(1, total_batches) * bar_width)
                bar = '#' * filled + '.' * (bar_width - filled)
                print(f"\rtrain [{bar}] {batch_idx+1}/{total_batches}", end='', flush=True)
                continue
            loss = losses.get('total_loss')
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                filled = int((batch_idx + 1) / max(1, total_batches) * bar_width)
                bar = '#' * filled + '.' * (bar_width - filled)
                print(f"\rtrain [{bar}] {batch_idx+1}/{total_batches}", end='', flush=True)
                continue
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += self.safe_item(loss)
            successful_batches += 1
            filled = int((batch_idx + 1) / max(1, total_batches) * bar_width)
            bar = '#' * filled + '.' * (bar_width - filled)
            print(f"\rtrain [{bar}] {batch_idx+1}/{total_batches}", end='', flush=True)
        avg_loss = total_loss / max(1, successful_batches)
        return {'train_loss': avg_loss}

    def evaluate_loader(self, dataloader, epoch_idx: int = -1, prefix: str = 'test_') -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_metrics = defaultdict(list)
        with torch.no_grad():
            num_eval_batches = len(dataloader)
            bar_width = 30
            for batch_idx, batch in enumerate(dataloader):
                try:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    behavior_sequences = batch['behavior_sequences']
                    target_items = batch['target_items']
                    raw_user_ids = batch.get('raw_user_ids') or batch.get('raw_user_id')

                    # Forward to get user representation
                    outputs = self.model(
                        behavior_sequences=behavior_sequences,
                        target_items=target_items,
                        raw_user_ids=raw_user_ids,
                        seq_mask=batch.get('seq_mask'),
                    )
                    user_repr = outputs.get('final_user_representation')  # [B,H]

                    # SASRec-style sampled eval: 1 positive + 100 negatives
                    num_items_model = getattr(self.model, 'num_items', 0)
                    B = behavior_sequences.size(0)
                    hr20 = 0.0
                    ndcg20 = 0.0
                    mrr20 = 0.0
                    for b in range(B):
                        pos = int(target_items[b].item())
                        rated = set(behavior_sequences[b][behavior_sequences[b] != 0].tolist())
                        rated.add(0)
                        candidates = [pos]
                        # sample 100 negatives
                        while len(candidates) < 101:
                            t = int(torch.randint(low=1, high=max(2, num_items_model), size=(1,), device=self.device).item())
                            if t not in rated and t != pos:
                                candidates.append(t)
                        item_table = self.model.item_embedding.weight[:num_items_model]
                        idx = torch.tensor(candidates, device=self.device)
                        emb = item_table.index_select(0, idx)
                        scores = torch.matmul(user_repr[b:b+1], emb.t()).view(-1)
                        # rank: index 0 is the positive
                        rank_idx = torch.argsort(-scores)
                        pos_rank = int((rank_idx == 0).nonzero(as_tuple=True)[0].item())
                        if pos_rank < 20:
                            hr20 += 1.0
                            ndcg20 += 1.0 / math.log2(pos_rank + 2)
                            mrr20 += 1.0 / (pos_rank + 1)
                    if B > 0:
                        hr20 /= B
                        ndcg20 /= B
                        mrr20 /= B

                    all_metrics['hr@20'].append(hr20)
                    all_metrics['ndcg@20'].append(ndcg20)
                    all_metrics['mrr@20'].append(mrr20)

                    # aggregate loss if exists
                    losses = outputs.get('losses', {})
                    if 'total_loss' in losses:
                        total_loss += self.safe_item(losses['total_loss'])

                    num_batches += 1
                    filled = int((batch_idx + 1) / max(1, num_eval_batches) * bar_width)
                    bar = '#' * filled + '.' * (bar_width - filled)
                    print(f"\reval [{bar}] {batch_idx+1}/{num_eval_batches}", end='', flush=True)
                except Exception:
                    continue
        avg_metrics = {
            f'{prefix}loss': total_loss / max(num_batches, 1),
        }
        for metric, values in all_metrics.items():
            if values:
                avg_metrics[f'{prefix}{metric}'] = float(np.mean(values))
        return avg_metrics

    def train(self, num_epochs: int = 1) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {'train_loss': []}
        best_metric = float('-inf')
        wait = 0
        patience = 10  # early stop when valid_ndcg@20 no improvement for 10 evals
        for epoch in range(1, int(num_epochs) + 1):
            t0 = time.time()
            metrics = self.train_epoch(epoch_idx=epoch)
            train_time = time.time() - t0
            epoch_loss = float(metrics.get('train_loss', 0.0))
            history['train_loss'].append(epoch_loss)
            print(f"\nEpoch {epoch}/{num_epochs} - train_loss: {epoch_loss:.6f} - train_time: {train_time:.1f}s")
            if epoch % 5 == 0:
                valid_metrics = self.evaluate_loader(self.valid_loader, epoch_idx=epoch, prefix='valid_')
                test_metrics = self.evaluate_loader(self.test_loader, epoch_idx=epoch, prefix='test_')
                keys_to_show = ['valid_hr@20', 'valid_ndcg@20', 'valid_mrr@20', 'test_hr@20', 'test_ndcg@20', 'test_mrr@20']
                show_parts = []
                for key in keys_to_show:
                    src = valid_metrics if key.startswith('valid_') else test_metrics
                    if key in src:
                        show_parts.append(f"{key}: {src[key]:.4f}")
                if show_parts:
                    print(" | ".join(show_parts))
                for k, v in {**valid_metrics, **test_metrics}.items():
                    if k not in history:
                        history[k] = []
                    history[k].append(float(v))
                # early stopping on valid_ndcg@20
                current = float(valid_metrics.get('valid_ndcg@20', float('-inf')))
                if np.isfinite(current) and current > best_metric:
                    best_metric = current
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping: valid NDCG@20 no improvement for {patience} evaluations")
                        break
            try:
                self.scheduler.step()
            except Exception:
                pass
        print("Training finished.")
        return history

def create_trainer(config: Dict[str, Any], device: torch.device):
    from model import Model
    from utils import data_loaders
    train_loader, valid_loader, test_loader, dataset_stats = data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        min_seq_len=config['min_seq_len'],
        num_workers=config['num_workers'],
        txt_path=config['txt_path'],
    )
    print(f"DataLoaders created")
    print(f"train batches: {len(train_loader)} | valid batches: {len(valid_loader)} | test batches: {len(test_loader)}")
    try:
        num_items_global = max(
            getattr(getattr(train_loader, 'dataset', None), 'num_items', 0),
            getattr(getattr(test_loader, 'dataset', None), 'num_items', 0),
            int(dataset_stats.get('num_items', 0)),
        )
    except Exception:
        num_items_global = int(dataset_stats.get('num_items', 0))
    config['dataset_info'] = {
        'num_users': dataset_stats['num_users'],
        'num_items': num_items_global,
    }
    model = Model(
        num_users=config['dataset_info']['num_users'],
        num_items=config['dataset_info']['num_items'],
        item_dim=config['item_dim'],
        data_dir=config['data_dir'],
        max_seq_len=config['max_seq_len'],
        num_interest_fusion_layers=config['num_interest_fusion_layers'],
        stage2_num_heads=config['stage2_num_heads'],
        interests_pt_path=config['interests_pt_path'],
        dropout_prob=config['dropout'],
        interest_dim=config['interest_dim'],
        stage2_dropout=config['stage2_dropout'],
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        config=config,
    )
    return trainer