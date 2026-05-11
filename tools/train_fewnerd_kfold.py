"""K-fold cross-validation for Context CNN v2 on Few-NERD."""
import torch, torch.nn as nn, numpy as np, os, sys
from datasets import load_dataset
from tqdm import tqdm


ds = load_dataset('DFKI-SLT/few-nerd', 'supervised')
tag_names = ds['train'].features['ner_tags'].feature.names
n_classes = len(tag_names)
print(f"Few-NERD: {len(ds['train'])} train, {n_classes} classes")
print(f"Tags: {tag_names}")

# Hash + Model (same as v2 but with __main__ guard)
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

class ContextHashCNN_v2(nn.Module):
    def __init__(self, embed_dim=128, n_classes=n_classes):
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

def prepare_batch(dataset, indices):
    bs = len(indices); ml = max(len(dataset[int(i)]['tokens']) for i in indices)
    x = torch.zeros(bs, ml, 16, dtype=torch.long); y = torch.full((bs, ml), -1, dtype=torch.long)
    for j, idx in enumerate(indices):
        ex = dataset[int(idx)]
        keys = compute_hash_keys(ex['tokens'])
        x[j,:len(ex['tokens'])] = torch.from_numpy(keys)
        y[j,:len(ex['tokens'])] = torch.tensor(ex['ner_tags'])
    return x, y

def evaluate(model, dataset, indices, bs=32):
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(indices), bs):
            batch = indices[i:i+bs]
            x, y = prepare_batch(dataset, batch); x, y = x.to(device), y.to(device)
            preds = model(x).argmax(-1); mask = y != -1
            correct += (preds[mask] == y[mask]).sum().item(); total += mask.sum().item()
    return correct / total if total else 0

def train_fold(fold, train_idx, val_idx):
    device = torch.device('cuda')
    model = ContextHashCNN_v2(embed_dim=128, n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    best_acc = 0
    for epoch in range(6):
        model.train(); total_loss, n = 0, 0
        np.random.shuffle(train_idx)
        for i in tqdm(range(0, len(train_idx), 16), desc=f'Fold {fold+1} E{epoch+1}', leave=False):
            x, y = prepare_batch(ds['train'], train_idx[i:i+16]); x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).reshape(-1, n_classes), y.reshape(-1))
            loss.backward(); optimizer.step()
            total_loss += loss.item(); n += 1
        scheduler.step()
        acc = evaluate(model, ds['train'], val_idx)
        if acc > best_acc: best_acc = acc
        print(f'Fold {fold+1} Epoch {epoch+1}: loss={total_loss/n:.4f} val_acc={acc:.4f} best={best_acc:.4f}')
    return best_acc

if __name__ == '__main__':
    device = torch.device('cuda')
    indices = np.arange(len(ds["train"]))
    np.random.seed(42); np.random.shuffle(indices)
    fold_size = len(indices) // 5
    indices = np.arange(len(ds['train']))
    
    scores = []
    print("\n=== 5-Fold Cross-Validation ===\n")
    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size if fold < 4 else len(indices)
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        acc = train_fold(fold, train_idx, val_idx)
        scores.append(acc)
        print(f'Fold {fold+1} done: {acc:.4f}\n')
    
    print(f'=== Results ===')
    print(f'Scores: {[f"{s:.4f}" for s in scores]}')
    print(f'Mean:   {np.mean(scores):.4f}')
    print(f'Std:    {np.std(scores):.4f}')
    print(f'Range:  {np.min(scores):.4f} - {np.max(scores):.4f}')
