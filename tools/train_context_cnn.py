"""Context CNN with hash-based embeddings — same input as bloom ONNX.
Input: (seq_len, 16) uint64 hash keys
No vocab table — MurmurHash-based like our C code."""
import torch, torch.nn as nn, numpy as np, os
from collections import Counter

def load_conll(path):
    sentences, labels = [], []
    tokens, tags = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('-DOCSTART-'):
                if tokens: sentences.append(tokens); labels.append(tags)
                tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) >= 2: tokens.append(parts[0]); tags.append(parts[-1])
    if tokens: sentences.append(tokens); labels.append(tags)
    return sentences, labels

print("Loading CoNLL-2003...")
train_s, train_t = load_conll('/tmp/train.conll')
dev_s, dev_t = load_conll('/tmp/dev.conll')
print(f"Train: {len(train_s)} sents, {sum(len(s) for s in train_s)} tokens")

# MurmurHash3 — python implementation matching our C code
def murmurhash3(key, seed=0):
    """MurmurHash3_x86_32 — matches spaCy/thinc implementation"""
    h = seed
    data = key.encode('utf-8') if isinstance(key, str) else key
    for i in range(0, len(data) - len(data) % 4, 4):
        k = data[i] | (data[i+1] << 8) | (data[i+2] << 16) | (data[i+3] << 24)
        k = (k * 0xcc9e2d51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1b873593) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = ((h * 5) + 0xe6546b64) & 0xFFFFFFFF
    tail = len(data) % 4
    if tail:
        k = 0
        for i in range(tail):
            k |= data[len(data) - tail + i] << (8 * i)
        k = (k * 0xcc9e2d51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1b873593) & 0xFFFFFFFF
        h ^= k
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= h >> 16
    return h

def compute_hash_keys(tokens, seeds=[8,9,10,11], n_cols=4):
    """Compute (seq_len, 16) hash keys matching bloom ONNX input.
    4 columns × 4 seeds each = 16 keys per token."""
    S = len(tokens)
    keys = np.zeros((S, 16), dtype=np.uint64)
    for i, tok in enumerate(tokens):
        text = tok.lower()
        # 4 keys per column (matching spaCy's MultiHashEmbed behavior)
        for col in range(n_cols):
            base = col * 4
            # Generate 4 different hash seeds per column
            for k in range(4):
                keys[i, base + k] = murmurhash3(text, seeds[col] * 4 + k)
    return keys

tag_set = sorted(set(t for tags in train_t for t in tags))
tag2id = {t: i for i, t in enumerate(tag_set)}
n_classes = len(tag_set)
print(f"Tags: {n_classes} classes")

class HashEmbed(nn.Module):
    """Hash-based embedding — same as bloom ONNX.
    col_slice: (S, 4) uint64 → Mod → Gather → Sum → (S, embed_dim)"""
    def __init__(self, nV, embed_dim, seed):
        super().__init__()
        self.nV = nV
        self.embed = nn.Embedding(nV, embed_dim)
        self.seed = seed
        
    def forward(self, col_slice):
        # col_slice: (S, 4) — 4 hash keys per token
        S = col_slice.shape[0]
        out = torch.zeros(S, self.embed.weight.shape[1], device=col_slice.device)
        for k in range(4):
            idx = (col_slice[:, k] % self.nV).long()
            out += self.embed(idx)
        return out

class ContextHashCNN(nn.Module):
    def __init__(self, embed_dim=128, n_classes=18):
        super().__init__()
        # 4 hash tables (same as bloom: NORM=5000, PREFIX=1000, SUFFIX=2500, SHAPE=2500)
        self.hash_embeds = nn.ModuleList([
            HashEmbed(5000, embed_dim, 8),
            HashEmbed(1000, embed_dim, 9),
            HashEmbed(2500, embed_dim, 10),
            HashEmbed(2500, embed_dim, 11),
        ])
        
        # Projection: 4*embed_dim → embed_dim
        self.proj = nn.Linear(embed_dim * 4, embed_dim)
        
        # Local + Global paths (same as before)
        self.local_conv = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.global_d1 = nn.Conv1d(embed_dim, embed_dim, 3, dilation=1, padding=1)
        self.global_d3 = nn.Conv1d(embed_dim, embed_dim, 3, dilation=3, padding=3)
        self.global_d5 = nn.Conv1d(embed_dim, embed_dim, 3, dilation=5, padding=5)
        
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.ner = nn.Linear(embed_dim, n_classes)
        self.drop = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: (B, S, 16) uint64 hash keys
        B, S, _ = x.shape
        
        # Hash embed for each column
        embeds = []
        for col, he in enumerate(self.hash_embeds):
            col_slice = x[:, :, col*4:(col+1)*4].reshape(B*S, 4)
            emb = he(col_slice)  # (B*S, E)
            embeds.append(emb.view(B, S, -1))
        
        emb = self.proj(torch.cat(embeds, dim=-1))  # (B, S, E)
        emb = self.drop(emb)
        
        # Local path
        emb_t = emb.transpose(1, 2)  # (B, E, S)
        local = torch.relu(self.local_conv(emb_t))
        
        # Global path
        g = torch.relu(self.global_d1(emb_t))
        g = torch.relu(self.global_d3(g))
        g = torch.relu(self.global_d5(g))
        
        # Gated fusion
        local_t = local.transpose(1, 2)
        g_t = g.transpose(1, 2)
        gate_w = torch.sigmoid(self.gate(self.drop(torch.cat([local_t, g_t], dim=-1))))
        fused = gate_w * local_t + (1 - gate_w) * g_t
        
        return self.ner(self.drop(fused))

# Pre-compute hash keys for entire dataset
print("Computing hash keys...")
import pickle
cache_file = '/tmp/conll_hash_keys.pkl'
if os.path.exists(cache_file):
    train_keys, dev_keys = pickle.load(open(cache_file, 'rb'))
else:
    train_keys = [compute_hash_keys(s) for s in train_s]
    dev_keys = [compute_hash_keys(s) for s in dev_s]
    pickle.dump((train_keys, dev_keys), open(cache_file, 'wb'))
print(f"Cached {len(train_keys)} train, {len(dev_keys)} dev")

device = torch.device('cuda')
model = ContextHashCNN(embed_dim=128, n_classes=n_classes).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

def batches(sents, tags, keys, bs=16):
    pairs = sorted(zip(sents, tags, keys), key=lambda x: len(x[0]))
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i+bs]
        ml = max(len(s) for s, _, _ in batch)
        x = torch.zeros(len(batch), ml, 16, dtype=torch.long, device=device)
        y = torch.full((len(batch), ml), -1, dtype=torch.long, device=device)
        for j, (tokens, labels, hk) in enumerate(batch):
            S = min(ml, len(tokens))
            x[j, :S, :] = torch.from_numpy(hk[:S])
            for k, lab in enumerate(labels[:S]):
                y[j, k] = tag2id[lab]
        yield x, y

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

print("\nTraining on GPU...")
for epoch in range(20):
    model.train()
    total_loss, n = 0, 0
    for x, y in batches(train_s, train_t, train_keys):
        optimizer.zero_grad()
        out = model(x)
        loss = nn.CrossEntropyLoss(ignore_index=-1)(out.reshape(-1, n_classes), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item(); n += 1
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in batches(dev_s, dev_t, dev_keys, bs=32):
            out = model(x)
            preds = out.argmax(-1)
            mask = y != -1
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    print(f"Epoch {epoch+1:2d}  loss={total_loss/n:.4f}  dev_acc={correct/total:.4f}")

# Export to ONNX
print("\nExporting to ONNX...")
model.eval()
dummy = torch.zeros(1, 20, 16, dtype=torch.long).to(device)
torch.onnx.export(model, dummy,
    '/home/max/Projects/mesh-actors/models/ner_context_hash.onnx',
    input_names=['token_hashes'], output_names=['ner_logits'],
    dynamic_axes={'token_hashes': {0:'batch',1:'seq_len'}, 'ner_logits': {0:'batch',1:'seq_len'}},
    opset_version=18)
size_kb = os.path.getsize('/home/max/Projects/mesh-actors/models/ner_context_hash.onnx') / 1024
print(f"Size: {size_kb:.0f} KB")
