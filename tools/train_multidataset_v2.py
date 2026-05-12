"""Train context CNN on merged Few-NERD + WNUT + CoNLL — direct file loading."""
import torch, torch.nn as nn, numpy as np, os, pickle
from datasets import load_dataset
from tqdm import tqdm

# ── Load from cache/disk (no HuggingFace API calls) ──────────────────────
print("Loading Few-NERD from cache...")
fewnerd = load_dataset('DFKI-SLT/few-nerd', 'supervised')

print("Loading WNUT from pickle...")
wnut = pickle.load(open('/tmp/wnut.pkl', 'rb'))

# CoNLL direct parse
def load_conll(path):
    sents, labels = [], []; tokens, tags = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('-DOCSTART-'):
                if tokens: sents.append(tokens); labels.append(tags); tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) >= 2: tokens.append(parts[0]); tags.append(parts[-1])
    if tokens: sents.append(tokens); labels.append(tags)
    return sents, labels

print("Loading CoNLL from disk...")
conll_train_s, conll_train_t = load_conll('/tmp/train.conll')
conll_dev_s, conll_dev_t = load_conll('/tmp/dev.conll')
print(f"CoNLL: {len(conll_train_s)} train, {len(conll_dev_s)} dev")

# ── Tag mapping ────────────────────────────────────────────────────────
fewnerd_tags = fewnerd['train'].features['ner_tags'].feature.names
tag2id = {t: i for i, t in enumerate(fewnerd_tags)}  # O=0, art=1, building=2, event=3, location=4, organization=5, other=6, person=7, product=8
print(f"Few-NERD tags: {fewnerd_tags}")

conll_map = {'O': 0, 'B-PER': 7, 'I-PER': 7, 'B-ORG': 5, 'I-ORG': 5, 'B-LOC': 4, 'I-LOC': 4, 'B-MISC': 6, 'I-MISC': 6}
wnut_map = {0: 0, 1: 5, 2: 5, 3: 1, 4: 1, 5: 6, 6: 6, 7: 4, 8: 4, 9: 7, 10: 7, 11: 8, 12: 8}  # WNUT tag IDs → Few-NERD

# ── Merge all into simple token+tag format ──────────────────────────────
def to_examples(sentences, tags):
    return [{'tokens': s, 'ner_tags': t} for s, t in zip(sentences, tags)]

def hf_to_examples(dataset):
    return [{'tokens': ex['tokens'], 'ner_tags': ex['ner_tags']} for ex in dataset]

# CoNLL with mapped tags
conll_train_ex = []
for s, t in zip(conll_train_s, conll_train_t):
    conll_train_ex.append({'tokens': s, 'ner_tags': [conll_map.get(tag, 0) for tag in t]})
conll_dev_ex = []
for s, t in zip(conll_dev_s, conll_dev_t):
    conll_dev_ex.append({'tokens': s, 'ner_tags': [conll_map.get(tag, 0) for tag in t]})

# WNUT with mapped tags
wnut_train_ex = []
for ex in wnut['train']:
    wnut_train_ex.append({'tokens': ex['tokens'], 'ner_tags': [wnut_map.get(t, 0) for t in ex['ner_tags']]})
wnut_dev_ex = []
for ex in wnut['validation']:
    wnut_dev_ex.append({'tokens': ex['tokens'], 'ner_tags': [wnut_map.get(t, 0) for t in ex['ner_tags']]})

# Merge
combined_train = hf_to_examples(fewnerd['train']) + wnut_train_ex + conll_train_ex
combined_val = hf_to_examples(fewnerd['validation']) + wnut_dev_ex + conll_dev_ex
print(f"Merged: {len(combined_train)} train, {len(combined_val)} val")

# ── Model ───────────────────────────────────────────────────────────────
def murmurhash3(key, seed=0):
    h = seed; data = key.encode('utf-8') if isinstance(key, str) else key
    for i in range(0, len(data) - len(data) % 4, 4):
        k = data[i] | (data[i+1] << 8) | (data[i+2] << 16) | (data[i+3] << 24)
        k = (k * 0xcc9e2d51) & 0xFFFFFFFF; k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1b873593) & 0xFFFFFFFF; h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF; h = ((h * 5) + 0xe6546b64) & 0xFFFFFFFF
    tail = len(data) % 4
    if tail:
        k = 0
        for i in range(tail): k |= data[len(data) - tail + i] << (8 * i)
        k = (k * 0xcc9e2d51) & 0xFFFFFFFF; k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1b873593) & 0xFFFFFFFF; h ^= k
    h ^= len(data); h ^= h >> 16; h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= h >> 13; h = (h * 0xc2b2ae35) & 0xFFFFFFFF; h ^= h >> 16
    return h

def compute_hash_keys(tokens):
    S = len(tokens); keys = np.zeros((S, 16), dtype=np.uint64)
    for i, tok in enumerate(tokens):
        for col, seed in enumerate([8, 9, 10, 11]):
            for k in range(4): keys[i, col*4+k] = murmurhash3(tok.lower(), seed*4+k)
    return keys

class HashEmbed(nn.Module):
    def __init__(self, nV, embed_dim): super().__init__(); self.nV = nV; self.embed = nn.Embedding(nV, embed_dim)
    def forward(self, col_slice):
        S = col_slice.shape[0]; out = torch.zeros(S, self.embed.weight.shape[1], device=col_slice.device)
        for k in range(4): idx = (col_slice[:, k] % self.nV).long(); out += self.embed(idx)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, dim, dilation=1): super().__init__(); self.conv = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation); self.norm = nn.LayerNorm(dim); self.drop = nn.Dropout(0.1)
    def forward(self, x):
        residual = x; out = self.conv(x); out = out.transpose(1,2); out = self.norm(out); out = out.transpose(1,2); out = torch.relu(out); out = self.drop(out); return out + residual

class ContextCNN(nn.Module):
    def __init__(self, embed_dim=96, n_classes=9):
        super().__init__()
        self.hash_embeds = nn.ModuleList([HashEmbed(5000, embed_dim), HashEmbed(1000, embed_dim), HashEmbed(2500, embed_dim), HashEmbed(2500, embed_dim)])
        self.proj = nn.Linear(embed_dim*4, embed_dim); self.norm_in = nn.LayerNorm(embed_dim)
        self.local = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.LayerNorm(embed_dim))
        self.global_res = nn.ModuleList([ResidualBlock(embed_dim, 1), ResidualBlock(embed_dim, 3), ResidualBlock(embed_dim, 5)])
        self.gate = nn.Linear(embed_dim*2, embed_dim); self.ner = nn.Linear(embed_dim, n_classes); self.drop = nn.Dropout(0.1)
    def forward(self, x):
        B, S, _ = x.shape
        embeds = []
        for col, he in enumerate(self.hash_embeds):
            col_slice = x[:,:,col*4:(col+1)*4].reshape(B*S,4); emb = he(col_slice); embeds.append(emb.view(B,S,-1))
        emb = self.proj(torch.cat(embeds, dim=-1)); emb = self.norm_in(emb); emb_t = emb.transpose(1,2)
        local = torch.relu(self.local[0](emb_t)); local = local.transpose(1,2); local = self.local[1](local); local = local.transpose(1,2)
        g = emb_t
        for res in self.global_res: g = res(g)
        local_t, g_t = local.transpose(1,2), g.transpose(1,2)
        gate_w = torch.sigmoid(self.gate(self.drop(torch.cat([local_t, g_t], dim=-1))))
        fused = gate_w * local_t + (1 - gate_w) * g_t
        return self.ner(self.drop(fused))

# ── Training ────────────────────────────────────────────────────────────
device = torch.device('cuda')
model = ContextCNN(embed_dim=96, n_classes=9).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

def prepare_batch(examples, indices):
    bs = len(indices); ml = max(len(examples[i]['tokens']) for i in indices)
    x = torch.zeros(bs, ml, 16, dtype=torch.long); y = torch.full((bs, ml), -1, dtype=torch.long)
    for j, idx in enumerate(indices):
        ex = examples[idx]; keys = compute_hash_keys(ex['tokens'])
        x[j,:len(ex['tokens'])] = torch.from_numpy(keys)
        y[j,:len(ex['tokens'])] = torch.tensor(ex['ner_tags'])
    return x.to(device), y.to(device)

print("\nTraining on merged dataset...")
best_acc = 0
for epoch in range(10):
    model.train(); total_loss, n = 0, 0
    indices = np.random.permutation(len(combined_train))
    for i in tqdm(range(0, len(indices), 16), desc=f'Epoch {epoch+1}'):
        x, y = prepare_batch(combined_train, indices[i:i+16])
        optimizer.zero_grad()
        loss = criterion(model(x).reshape(-1, 9), y.reshape(-1))
        loss.backward(); optimizer.step()
        total_loss += loss.item(); n += 1
    scheduler.step()
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(combined_val), 32):
            x, y = prepare_batch(combined_val, list(range(i, min(i+32, len(combined_val)))))
            preds = model(x).argmax(-1); mask = y != -1
            correct += (preds[mask] == y[mask]).sum().item(); total += mask.sum().item()
    acc = correct / total
    print(f'Epoch {epoch+1}: loss={total_loss/n:.4f} val_acc={acc:.4f}')
    if acc > best_acc: best_acc = acc; torch.save(model.state_dict(), '/home/max/Projects/mesh-actors/models/multidataset_best.pt')

print(f'\nBest: {best_acc:.4f}')
