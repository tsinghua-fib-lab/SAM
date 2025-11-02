import os
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import csv


class InterestGroupIdentificationModule(nn.Module):
	"""Interest group identification module"""
	
	def __init__(self, hidden_dim: int, K_min: int, K_max: int, num_iterations: int, alpha: float, beta: float):
		super(InterestGroupIdentificationModule, self).__init__()
		self.hidden_dim = hidden_dim
		self.K_min = K_min
		self.K_max = K_max
		self.num_iterations = num_iterations
		# Consistency/diversity weighting coefficients
		self.alpha = float(alpha)
		self.beta = float(beta)
		# 胶囊网络参数
		self.num_capsules = K_max
		self.stop_grad = True
		# Map each item embedding into capsule space
		self.linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
		
	def squash(self, s: torch.Tensor) -> torch.Tensor:
		"""Squash activation"""
		s_norm = torch.norm(s, dim=-1, keepdim=True)
		return (s_norm ** 2) / (1 + s_norm ** 2) * s / (s_norm + 1e-8)
	
	def dynamic_routing(self, user_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Dynamic routing (B2I). Returns optimal K per sample and scores.
		Input: user_features [B, L, H]
		"""
		batch_size, seq_len, hidden = user_features.shape
		device = user_features.device
		
		# Map items into capsule space and replicate for K interests
		item_eb = user_features  # [B, L, H]
		item_eb_hat = self.linear(item_eb)  # [B, L, H]
		item_eb_hat = item_eb_hat.repeat(1, 1, self.num_capsules)  # [B, L, H*K]
		item_eb_hat = item_eb_hat.view(batch_size, seq_len, self.num_capsules, hidden)
		item_eb_hat = item_eb_hat.transpose(1, 2).contiguous()  # [B, K, L, H]
		item_eb_hat_iter = item_eb_hat.detach() if self.stop_grad else item_eb_hat
		
		# Initialize routing logits b[b,k,l]
		b = torch.randn(batch_size, self.num_capsules, seq_len, device=device)
		mask = torch.ones(batch_size, seq_len, device=device)  # 无显式mask时视为全1
		
		# Dynamic routing iterations: softmax over items per interest and update b[b,k,l]
		for t in range(self.num_iterations):
			atten_mask = mask.unsqueeze(1).repeat(1, self.num_capsules, 1)  # [B, K, L]
			paddings = torch.zeros_like(atten_mask, dtype=torch.float)
			c = torch.softmax(b, dim=-1)  # softmax over items
			c = torch.where(atten_mask == 0, paddings, c)
			c = c.unsqueeze(2)  # [B, K, 1, L]
			
			if t < self.num_iterations - 1:
				# s = c * u_hat -> v（squash）
				interest_capsule = torch.matmul(c, item_eb_hat_iter)  # [B, K, 1, H]
				cap_norm = torch.sum(interest_capsule.pow(2), dim=-1, keepdim=True)  # [B, K, 1, 1]
				scalar = cap_norm / (1.0 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
				interest_capsule = scalar * interest_capsule  # squash
				# Update b: b += u_hat · v
				delta = torch.matmul(item_eb_hat_iter, interest_capsule.transpose(2, 3).contiguous())  # [B, K, L, 1]
				delta = delta.view(batch_size, self.num_capsules, seq_len)
				b = b + delta
			else:
				# Last iteration: aggregate using un-truncated u_hat
				interest_capsule = torch.matmul(c, item_eb_hat)  # [B, K, 1, H]
				cap_norm = torch.sum(interest_capsule.pow(2), dim=-1, keepdim=True)
				scalar = cap_norm / (1.0 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
				interest_capsule = scalar * interest_capsule
		
		# Output interest capsules and final coupling weights
		interest_capsule = interest_capsule.view(batch_size, self.num_capsules, hidden)  # [B, K, H]
		final_c = c.squeeze(2)  # [B, K, L]
		
		# Compute Consistency(K) and Diversity(K)
		k_scores: List[torch.Tensor] = []
		alpha, beta = self.alpha, self.beta

		# Normalize raw item embeddings for cosine similarity
		e = item_eb  # [B, L, H]
		e_norm = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
		mu_all = interest_capsule  # [B, K, H]

		# Select top-k interests by strength (||v||)
		strength = torch.norm(mu_all, dim=-1)  # [B, K]
		for k in range(self.K_min, self.K_max + 1):
			top_k_indices = torch.topk(strength, k, dim=1)[1]  # [B, k]
			score_per_batch: List[torch.Tensor] = []
			for bi in range(batch_size):
				idx = top_k_indices[bi]  # [k]
				# Select corresponding weights and centers
				c_k = final_c[bi].index_select(0, idx)  # [k, L]
				denom = c_k.sum(dim=1, keepdim=True) + 1e-8  # [k,1]
				# Re-compute cluster centers μ_j in the original embedding space using c_k
				mu_k = torch.einsum('kl,lh->kh', c_k, e[bi]) / denom  # [k, H]
				mu_k_norm = mu_k / (mu_k.norm(dim=-1, keepdim=True) + 1e-8)
				# Consistency: 1/(K*L) sum_j sum_i c_ji * cos(e_i, μ_j)
				cos_e_mu = torch.einsum('lh,kh->kl', e_norm[bi], mu_k_norm)  # [k, L]
				consistency = (c_k * cos_e_mu).sum() / float(max(1, k) * max(1, seq_len))
				# Diversity: 1 - 2/(K(K-1)) sum_{j<k} cos(μ_j, μ_k)
				if k > 1:
					pair = torch.matmul(mu_k_norm, mu_k_norm.t())  # [k, k]
					upper = torch.triu(pair, diagonal=1)
					sum_pairs = upper.sum()
					diversity = 1.0 - (2.0 / (k * (k - 1))) * sum_pairs
				else:
					diversity = torch.tensor(1.0, device=device)
				total = alpha * consistency + beta * diversity
				score_per_batch.append(total.unsqueeze(0))
			k_scores.append(torch.cat(score_per_batch, dim=0))  # [B]
		
		# Select optimal K
		k_scores_tensor = torch.stack(k_scores, dim=1)  # [B, num_k_values]
		optimal_k_indices = torch.argmax(k_scores_tensor, dim=1)
		optimal_k = optimal_k_indices + self.K_min
		
		return optimal_k, k_scores_tensor
	
	def forward(self, user_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Forward"""
		return self.dynamic_routing(user_features)


class LLMInterestGenerator(nn.Module):

	def __init__(self, hidden_dim: int, interest_dim: int, vocab_size: int, K_max: int, api_key: str):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.interest_dim = interest_dim
		self.vocab_size = vocab_size
		self.K_max = K_max
		self.api_key = api_key
		self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
		self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
		self.interest_encoder = nn.Linear(hidden_dim, interest_dim)
		self.interest_projection = nn.Linear(interest_dim, vocab_size)
		self.interest_cache: Dict[str, List[str]] = {}
	
	def _prompt(self, behavior_sequence: List[str], optimal_k: int) -> str:
		return (
			"The user's historical behavior sequence is as follows:\n\n"
			f"{', '.join(behavior_sequence)}\n"
			f" please infer the user's interest preference based on the user's historical behavior sequence and output exact {optimal_k} interests. \n\n"
			"Return only the list of interests, one per line, without any numbering or extra commentary."
		)
	
	def _call_api_with_retry(self, prompt: str, max_retries: int = 3, backoff: float = 0.8) -> str:
		import requests, time
		exc = None
		for i in range(max(1, max_retries)):
			try:
				resp = requests.post(
					self.api_url,
					headers=self.headers,
					json={
						"model": "qwen-turbo",
						"input": {"messages": [{"role": "user", "content": prompt}]},
						"parameters": {"max_tokens": 500, "temperature": 0.3, "top_p": 0.8}
					},
					timeout=30,
				)
				if resp.status_code == 200:
					data = resp.json()
					if 'output' in data and 'text' in data['output']:
						return data['output']['text']
					elif 'choices' in data and data['choices']:
						return data['choices'][0]['message']['content']
					raise Exception(f"bad payload {data}")
				else:
					try:
						print(f"[LLM][HTTP {resp.status_code}] {resp.text[:200]}")
					except Exception:
						pass
			except Exception as e:
				exc = e
				time.sleep(backoff * (i + 1))
		if exc:
			raise exc
		return ""
	
	def _parse_lines(self, content: str, num: int) -> List[str]:
		lines = [l.strip() for l in (content or "").split('\n') if l.strip()]
		qs: List[str] = []
		for line in lines:
			qs.append(line.split('. ',1)[1].strip() if '. ' in line else line)
		return qs[:max(1, num)]
		
	def forward(self, user_representation: torch.Tensor, optimal_k: torch.Tensor,
				 user_behavior_sequences: Optional[List[List[str]]] = None) -> Dict[str, Any]:
		B = user_representation.size(0)
		texts: List[List[str]] = []
		for i in range(B):
			k = int(optimal_k[i].item())
			beh = user_behavior_sequences[i] if user_behavior_sequences else [f"item_{j}" for j in range(10)]
			cache_key = f"{k}|{' '.join(beh)[:128]}"
			if cache_key in self.interest_cache:
				texts.append(self.interest_cache[cache_key])
				continue
			content = self._call_api_with_retry(self._prompt(beh, k))
			qs = self._parse_lines(content, k)
			self.interest_cache[cache_key] = qs
			texts.append(qs)
		features = self.interest_encoder(user_representation)
		logits = self.interest_projection(features)
		return {"interest_texts": texts, "interest_logits": logits}
 

def read_txt(path: str) -> List[Tuple[int, int, str]]:
    """Read dataset.txt
    Each line (tab-delimited): user_id, item_id, title, tags
    Returns list [(user, item, title)]
    """
    rows: List[Tuple[int, int, str]] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                parts = line.split()
                if len(parts) < 3:
                    continue
            try:
                uid = int(parts[0])
                iid = int(parts[1])
                # If split by spaces, title may contain spaces; merge from the 3rd field to the end
                title = parts[2].strip() if '\t' in line else ' '.join(parts[2:]).strip()
            except Exception:
                continue
            rows.append((uid, iid, title))
    return rows


def partition(rows: List[Tuple[int, int, str]]):
    """Split into train/valid/test.
    Rule: last -> test, second-last -> valid, the rest -> train; relax when not enough.
    Returns: user_train, user_valid, user_test, usernum, itemnum, user_titles
    user_titles: {uid: [title1, title2, ...]} aligned with interactions
    """
    by_user: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    usernum = 0
    itemnum = 0
    for uid, iid, title in rows:
        by_user[uid].append((iid, title))
        usernum = max(usernum, uid)
        itemnum = max(itemnum, iid)
    user_train: Dict[int, List[int]] = {}
    user_valid: Dict[int, List[int]] = {}
    user_test: Dict[int, List[int]] = {}
    user_titles: Dict[int, List[str]] = {}
    for uid, lst in by_user.items():
        items = [iid for iid, _ in lst]
        titles = [t for _, t in lst]
        n = len(items)
        # Test: last
        user_test[uid] = [items[-1]] if n >= 1 else []
        # Valid: second last
        user_valid[uid] = [items[-2]] if n >= 2 else []
        # Train: the rest
        user_train[uid] = items[:-2] if n >= 3 else []
        user_titles[uid] = titles
    return user_train, user_valid, user_test, usernum, itemnum, user_titles

def batch_encode_texts(tokenizer: BertTokenizer, bert: BertModel, texts: List[str], device: torch.device) -> torch.Tensor:
    """BERT CLS encoding. Returns [N, 768]."""
    if len(texts) == 0:
        return torch.zeros(0, bert.config.hidden_size, device=device)
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    with torch.no_grad():
        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
    return cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty', help='dataset name; read from Data/<dataset>.txt')
    parser.add_argument('--k_min', type=int, default=2, help='min K for dynamic routing')
    parser.add_argument('--k_max', type=int, default=10, help='max K for dynamic routing')
    parser.add_argument('--routing_iters', type=int, default=3, help='routing iterations')
    parser.add_argument('--alpha', type=float, default=0.6, help='consistency weight')
    parser.add_argument('--beta', type=float, default=0.4, help='diversity weight')
    parser.add_argument('--maxlen', type=int, default=20, help='max sequence length')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--api_key', type=str, default=os.getenv('DASHSCOPE_API_KEY',''))
    parser.add_argument('--debug', action='store_true', help='debug mode: print prompt and raw response snippets')
    parser.add_argument('--hidden_dim',  type=int, default=128,help='hidden dimension')
    parser.add_argument('--vocab_size',  type=int, default=1000, help='vocabulary size')
    parser.add_argument('--interest_dim', type=int, default=64, help='interest vector dimension')

    args = parser.parse_args()

    dataset_path = os.path.join(os.path.dirname(__file__), 'Data', f"{args.dataset}.txt")
    rows = read_txt(dataset_path)
    train, valid, test, usernum, itemnum, user_titles = partition(rows)
    print(f'usernum={usernum}, itemnum={itemnum}')

    # Initialize BERT
    bert_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'LLMs', 'bert-base-uncased'))
    tokenizer = BertTokenizer.from_pretrained(bert_src, local_files_only=True)
    bert = BertModel.from_pretrained(bert_src, local_files_only=True).to(args.device)
    bert.eval()

    # Initialize LLM interest generator
    api_key = args.api_key
    llm = LLMInterestGenerator(hidden_dim=args.hidden_dim, interest_dim=args.interest_dim, vocab_size=args.vocab_size, K_max=args.k_max, api_key=api_key)
    llm = llm.to(args.device)  # ensure same device
    if args.debug:
        try:
            print(f"[DEBUG] Using API Key, device={args.device}")
        except Exception:
            pass

    # Interest group identification module
    routing = InterestGroupIdentificationModule(hidden_dim=bert.config.hidden_size, K_min=args.k_min, K_max=args.k_max, num_iterations=args.routing_iters, alpha=args.alpha, beta=args.beta)
    routing = routing.to(args.device)

    # Generate interests for each training user and encode via BERT
    user_to_interest: Dict[int, List[str]] = {}
    user_to_embed: Dict[int, torch.Tensor] = {}
    users = list(range(1, usernum + 1))
    total = len(users)
    for idx, uid in enumerate(users, 1):
        hist = train.get(uid, [])
        if len(hist) == 0:
            continue
        # Build behavior title sequence (last maxlen)
        titles = user_titles.get(uid, [])
        # Align titles with interactions and truncate to len(train)
        t_hist = titles[:len(hist)]
        t_hist = t_hist[-args.maxlen:]
        # Generate interests

        # Encode titles via BERT to [1, L, H]
        title_emb = batch_encode_texts(tokenizer, bert, t_hist, device=args.device)  # [L, 768]
        user_feat = title_emb.unsqueeze(0)  # [1, L, H]
        with torch.no_grad():
            k_tensor, _ = routing(user_feat)
        k_user = int(k_tensor.view(-1)[0].item())
        k_user = max(args.k_min, min(args.k_max, k_user))
        if args.debug:
            print(f"[DEBUG] uid={uid} k_user={k_user} t_hist_len={len(t_hist)}")
        if args.debug and idx <= 3:
            prompt = llm._prompt(behavior_sequence=t_hist, optimal_k=k_user)
            print(f"[DEBUG] uid={uid} Prompt=\n{prompt[:400]}...\n")
            raw = llm._call_api_with_retry(prompt, max_retries=1)
            print(f"[DEBUG] uid={uid} RawResp=\n{str(raw)[:400]}...\n")

        out = llm(
            user_representation=torch.zeros(1, args.hidden_dim, device=args.device),
            optimal_k=torch.tensor([k_user], device=args.device),
            user_behavior_sequences=[t_hist],
        )
        qs = out.get('interest_texts', [[]])[0]
        if args.debug:
            print(f"[DEBUG] uid={uid} parsed_qs={len(qs)}")
                
        user_to_interest[uid] = qs
        # BERT CLS encoding (768d)
        texts = [q for q in qs if isinstance(q, str) and q.strip()]
        if len(texts) == 0:
            emb = torch.zeros(0, bert.config.hidden_size, device=args.device)
        else:
            emb = batch_encode_texts(tokenizer, bert, texts, device=args.device)
        user_to_embed[uid] = emb.detach().cpu()
        
        # Print per-user preview
        kk = k_user
        preview = ' | '.join([q.strip() for q in (qs[:kk])])
        print(f"uid={uid}  K={len(qs)}  interests: {preview}")
        sys.stdout.flush()
        if idx % 500 == 0:
            print(f'Progress: {idx}/{total}')
            sys.stdout.flush()

    # Save: CSV, PT
    os.makedirs('./Interests', exist_ok=True)
    csv_out = f"./Interests/interests_{args.dataset}.csv"
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['userID', 'interest'])
        for uid in range(1, usernum + 1):
            qs = user_to_interest.get(uid, [])
            writer.writerow([uid, ' || '.join(qs)])
    out_pt_path = f"./Interests/interests_{args.dataset}.pt"
    torch.save(user_to_embed, out_pt_path)
    print(f"Saved interest texts: {csv_out}\nSaved interest vectors: {out_pt_path}")

if __name__ == '__main__':
    main() 