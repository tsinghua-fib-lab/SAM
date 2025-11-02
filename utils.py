import os
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


MAX_LEN_DEFAULT: int = 20
PAD_ITEM_ID: int = 0


def load_users_from_txt(txt_path: str) -> Dict[int, List[int]]:
    """Read dataset.txt; each line: user_id \t item_id \t title \t tags (at least first 2).
    Return {uid: [item_id, ...]} in chronological order.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Dataset file not found: {txt_path}")
    users: Dict[int, List[int]] = {}
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split()
                if len(parts) < 2:
                    continue
            try:
                uid = int(parts[0]); iid = int(parts[1])
            except Exception:
                continue
            users.setdefault(uid, []).append(iid)
    return users


class SequenceDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], max_len: int = MAX_LEN_DEFAULT):
        self.samples = samples
        self.max_len = max_len
        # Collect user/item statistics
        self.user_to_id: Dict[Any, int] = {}
        self.num_items = 0
        for s in self.samples:
            if s['raw_user_id'] not in self.user_to_id:
                self.user_to_id[s['raw_user_id']] = len(self.user_to_id)
            for x in s['hist']:
                self.num_items = max(self.num_items, int(x))
            self.num_items = max(self.num_items, int(s['target']))
        self.num_items += 1  # include 0 as PAD

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        uid = self.user_to_id.get(s['raw_user_id'], 0)
        hist = list(s['hist'])[-self.max_len:]
        pad_len = self.max_len - len(hist)
        if pad_len > 0:
            hist = [PAD_ITEM_ID] * pad_len + hist
        return {
            'user_id': torch.tensor(uid, dtype=torch.long),
            'behavior_sequences': torch.tensor(hist, dtype=torch.long),
            'target_items': torch.tensor(int(s['target']), dtype=torch.long),
            'raw_user_id': s['raw_user_id'],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    user_ids = torch.stack([b['user_id'] for b in batch])
    behavior_sequences = torch.stack([b['behavior_sequences'] for b in batch])
    target_items = torch.stack([b['target_items'] for b in batch])
    raw_user_ids = [b['raw_user_id'] for b in batch]
    # Sequence mask: True=real position, False=PAD
    seq_mask = (behavior_sequences != PAD_ITEM_ID)
    return {
        'user_id': user_ids,
        'behavior_sequences': behavior_sequences,
        'target_items': target_items,
        'raw_user_id': raw_user_ids,  # backward compatibility
        'raw_user_ids': raw_user_ids, # used by model/trainer
        'seq_mask': seq_mask,
    }


def data_loaders(
    data_dir: str,
    batch_size: int = 128,
    max_seq_len: int = MAX_LEN_DEFAULT,
    min_seq_len: int = 2,
    num_workers: int = 4,
    txt_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Main entry (SASRec-aligned)
    - Split per user: train=[:-2], valid=[-2], test=[-1] (relax if not enough)
    - Train sample (no sliding): hist=last max_seq_len of train[:-1], target=train[-1]
    - Valid sample: hist=last max_seq_len of train, target=valid
    - Test sample: hist=last max_seq_len of (train+valid), target=test
    Return (train_loader, valid_loader, test_loader, stats)
    """
    if not (isinstance(txt_path, str) and len(txt_path) > 0 and os.path.exists(txt_path)):
        raise FileNotFoundError(f"Must provide a valid txt_path: {txt_path}")

    users_seq = load_users_from_txt(txt_path)

    train_samples: List[Dict[str, Any]] = []
    valid_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    for uid, seq in users_seq.items():
        # Drop users with too short sequences
        if not isinstance(seq, list) or len(seq) < 2:
            continue
        # SASRec-style split
        if len(seq) >= 3:
            train_part = seq[:-2]
            valid_part = [seq[-2]]
            test_part  = [seq[-1]]
        else:
            train_part = seq[:-1]
            valid_part = []
            test_part  = [seq[-1]]

        # Train (no sliding): hist from train_part[:-1] tail, target=train_part[-1]
        if len(train_part) >= 2:
            hist = train_part[:-1][-max_seq_len:]
            target = train_part[-1]
            train_samples.append({'raw_user_id': uid, 'hist': hist, 'target': target})

        # Valid: hist from train_part tail, target=valid
        if len(train_part) >= 1 and len(valid_part) == 1:
            v_hist = train_part[-max_seq_len:]
            v_target = valid_part[0]
            valid_samples.append({'raw_user_id': uid, 'hist': v_hist, 'target': v_target})

        # Test: hist from (train_part+valid_part) tail, target=test
        if len(test_part) == 1 and (len(train_part) + len(valid_part)) >= 1:
            hist_full = (train_part + valid_part)[-max_seq_len:]
            test_target = test_part[0]
            test_samples.append({'raw_user_id': uid, 'hist': hist_full, 'target': test_target})

    train_dataset = SequenceDataset(train_samples, max_len=max_seq_len)
    valid_dataset = SequenceDataset(valid_samples, max_len=max_seq_len)
    test_dataset = SequenceDataset(test_samples, max_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    stats = {
        'num_users': len(train_dataset.user_to_id),
        'num_items': train_dataset.num_items,
        'num_train_seqs': len(train_dataset),
        'num_valid_seqs': len(valid_dataset),
        'num_test_seqs': len(test_dataset),
    }
    return train_loader, valid_loader, test_loader, stats