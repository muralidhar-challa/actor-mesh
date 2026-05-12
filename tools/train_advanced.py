"""Context CNN v3: +RoPE +SE +Depthwise +HardSwish. 128-dim, ~2M params."""

import math
import os

import numpy as np
import torch
import torch.nn as nn
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm


# ── Rotatry Position Embedding ──────────────────────────────────────────
class RoPE(nn.Module):
    """Rotary Position Embedding — zero params."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B, S, E) — apply rotation to pairs of dims
        B, S, E = x.shape
        device = x.device
        # Position indices
        pos = torch.arange(S, device=device).float()
        # Frequency bands
        freqs = 1.0 / (10000 ** (torch.arange(0, E, 2, device=device).float() / E))
        # (S, E/2)
        theta = pos.unsqueeze(1) * freqs.unsqueeze(0)  # (S, E/2)
        cos = theta.cos().unsqueeze(0)  # (1, S, E/2)
        sin = theta.sin().unsqueeze(0)

        x_rot = torch.zeros_like(x)
        x_rot[:, :, 0::2] = x[:, :, 0::2] * cos - x[:, :, 1::2] * sin
        x_rot[:, :, 1::2] = x[:, :, 0::2] * sin + x[:, :, 1::2] * cos
        return x_rot


# ── Squeeze-and-Excitation (SE) Block ──────────────────────────────────
class SEBlock(nn.Module):
    """Channel attention — always pools over spatial dim, gates channels."""

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.Hardswish(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, S, E) — pool over S, gate over E
        pooled = x.mean(dim=1)  # (B, E)
        gate = self.se(pooled).unsqueeze(1)  # (B, 1, E)
        return x * gate


# ── Depthwise Separable Conv ───────────────────────────────────────────
class SepConv(nn.Module):
    """Depthwise(3×3) + Pointwise(1×1). 8× fewer params."""

    def __init__(self, dim, dilation=1):
        super().__init__()
        padding = dilation
        self.depthwise = nn.Conv1d(
            dim, dim, 3, padding=padding, dilation=dilation, groups=dim
        )
        self.pointwise = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# ── Residual Block with SE + LayerNorm ─────────────────────────────────
class SepResBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.conv = SepConv(dim, dilation)
        self.norm = nn.LayerNorm(dim)
        self.se = SEBlock(dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.se(out)  # SE in (B, S, E)
        out = out.transpose(1, 2)
        out = torch.nn.functional.hardswish(out)
        out = self.drop(out)
        return out + residual


# ── Hash Embed ─────────────────────────────────────────────────────────
class HashEmbed(nn.Module):
    def __init__(self, nV, embed_dim):
        super().__init__()
        self.nV = nV
        self.embed = nn.Embedding(nV, embed_dim)

    def forward(self, col_slice):
        S = col_slice.shape[0]
        out = torch.zeros(S, self.embed.weight.shape[1], device=col_slice.device)
        for k in range(4):
            idx = (col_slice[:, k] % self.nV).long()
            out += self.embed(idx)
        return out


# ── Full Model ─────────────────────────────────────────────────────────
class AdvancedNER(nn.Module):
    def __init__(self, embed_dim=128, n_classes=33):
        super().__init__()
        self.hash_embeds = nn.ModuleList(
            [
                HashEmbed(5000, embed_dim),
                HashEmbed(1000, embed_dim),
                HashEmbed(2500, embed_dim),
                HashEmbed(2500, embed_dim),
            ]
        )
        self.proj = nn.Linear(embed_dim * 4, embed_dim)
        self.norm_in = nn.LayerNorm(embed_dim)
        self.rope = RoPE(embed_dim)

        # Local path: sep conv + SE
        self.local_conv = SepConv(embed_dim)
        self.local_norm = nn.LayerNorm(embed_dim)
        self.local_se = SEBlock(embed_dim)

        # Global path: dilated sep convs + SE
        self.global_res = nn.ModuleList(
            [
                SepResBlock(embed_dim, dilation=1),
                SepResBlock(embed_dim, dilation=3),
                SepResBlock(embed_dim, dilation=5),
            ]
        )

        # Gated fusion
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.ner = nn.Linear(embed_dim, n_classes)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        B, S, _ = x.shape
        # Hash embed + project
        embeds = []
        for col, he in enumerate(self.hash_embeds):
            col_slice = x[:, :, col * 4 : (col + 1) * 4].reshape(B * S, 4)
            emb = he(col_slice)
            embeds.append(emb.view(B, S, -1))
        emb = self.proj(torch.cat(embeds, dim=-1))
        emb = self.norm_in(emb)
        # SE moved to conv blocks (better with local context)  # channel attention on embeddings
        emb = self.rope(emb)  # position encoding

        emb_t = emb.transpose(1, 2)

        # Local
        local = self.local_conv(emb_t).transpose(1, 2)
        local = self.local_norm(local)
        local = self.local_se(local).transpose(1, 2)
        local = torch.nn.functional.hardswish(local)

        # Global (dilated)
        g = emb_t
        for res in self.global_res:
            g = res(g)

        # Gated fusion
        local_t = local.transpose(1, 2)
        g_t = g.transpose(1, 2)
        gate_w = torch.sigmoid(self.gate(self.drop(torch.cat([local_t, g_t], dim=-1))))
        fused = gate_w * local_t + (1 - gate_w) * g_t

        return self.ner(self.drop(fused))


# ── Data loading ────────────────────────────────────────────────────────
def murmurhash3(key, seed=0):
    h = seed
    data = key.encode("utf-8") if isinstance(key, str) else key
    for i in range(0, len(data) - len(data) % 4, 4):
        k = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) | (data[i + 3] << 24)
        k = (k * 0xCC9E2D51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1B873593) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = ((h * 5) + 0xE6546B64) & 0xFFFFFFFF
    tail = len(data) % 4
    if tail:
        k = 0
        for i in range(tail):
            k |= data[len(data) - tail + i] << (8 * i)
        k = (k * 0xCC9E2D51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1B873593) & 0xFFFFFFFF
        h ^= k
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def compute_hash_keys(tokens):
    S = len(tokens)
    keys = np.zeros((S, 16), dtype=np.uint64)
    for i, tok in enumerate(tokens):
        for col, seed in enumerate([8, 9, 10, 11]):
            for k in range(4):
                keys[i, col * 4 + k] = murmurhash3(tok.lower(), seed * 4 + k)
    return keys


def load_conll(path):
    sents, labels = [], []
    tokens, tags = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if tokens:
                    sents.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[-1])
    if tokens:
        sents.append(tokens)
        labels.append(tags)
    return sents, labels


print("Loading data...")
import pickle

fewnerd = load_dataset("DFKI-SLT/few-nerd", "supervised")
wnut = pickle.load(open("/tmp/wnut.pkl", "rb"))
conll_train_s, conll_train_t = load_conll("/tmp/train.conll")
conll_dev_s, conll_dev_t = load_conll("/tmp/dev.conll")

fewnerd_tags = fewnerd["train"].features["ner_tags"].feature.names
tag2id = {t: i for i, t in enumerate(fewnerd_tags)}
conll_map = {
    "O": 0,
    "B-PER": 7,
    "I-PER": 7,
    "B-ORG": 5,
    "I-ORG": 5,
    "B-LOC": 4,
    "I-LOC": 4,
    "B-MISC": 6,
    "I-MISC": 6,
}
wnut_map = {
    0: 0,
    1: 5,
    2: 5,
    3: 1,
    4: 1,
    5: 6,
    6: 6,
    7: 4,
    8: 4,
    9: 7,
    10: 7,
    11: 8,
    12: 8,
}


def hf_to_examples(dataset):
    return [{"tokens": ex["tokens"], "ner_tags": ex["ner_tags"]} for ex in dataset]


combined_train = (
    hf_to_examples(fewnerd["train"])
    + [
        {
            "tokens": ex["tokens"],
            "ner_tags": [wnut_map.get(t, 0) for t in ex["ner_tags"]],
        }
        for ex in wnut["train"]
    ]
    + [
        {"tokens": s, "ner_tags": [conll_map.get(t, 0) for t in t]}
        for s, t in zip(conll_train_s, conll_train_t)
    ]
)

combined_val = (
    hf_to_examples(fewnerd["validation"])
    + [
        {
            "tokens": ex["tokens"],
            "ner_tags": [wnut_map.get(t, 0) for t in ex["ner_tags"]],
        }
        for ex in wnut["validation"]
    ]
    + [
        {"tokens": s, "ner_tags": [conll_map.get(t, 0) for t in t]}
        for s, t in zip(conll_dev_s, conll_dev_t)
    ]
)


# ── IO → BILOU conversion ──────────────────────────────────────────────
# Converts: [PER, PER, O, ORG, ORG, ORG] → [B-PER, L-PER, O, B-ORG, I-ORG, L-ORG]
def io_to_bilou(tags, tag_names):
    """Convert IO tags to BILOU format. Returns (bilou_tags, bilou_classes)."""
    bilou = []
    n = len(tags)
    i = 0
    while i < n:
        t = tags[i]
        if t == 0:  # O
            bilou.append(0)
            i += 1
        else:
            # Start of entity
            start = i
            while i < n and tags[i] == t:
                i += 1
            length = i - start
            entity_type = t  # 1..N-1
            if length == 1:
                bilou.append((entity_type - 1) * 4 + 3 + 1)  # U-entity
            else:
                bilou.append((entity_type - 1) * 4 + 0 + 1)  # B-entity
                for j in range(1, length - 1):
                    bilou.append((entity_type - 1) * 4 + 1 + 1)  # I-entity
                bilou.append((entity_type - 1) * 4 + 2 + 1)  # L-entity
    return bilou


# BILOU class layout: O=0, then for each of 8 entity types: B=+0, I=+1, L=+2, U=+3
N_ENTITY_TYPES = (
    8  # art, building, event, location, organization, other, person, product
)
BILOU_CLASSES = N_ENTITY_TYPES * 4 + 1  # 0..32 = 33 classes  # 33
print(f"BILOU classes: {BILOU_CLASSES} (O + 8 types × 4 BILOU tags)")

# Convert all examples to BILOU
for ex in combined_train:
    ex["ner_tags"] = io_to_bilou(ex["ner_tags"], fewnerd_tags)
for ex in combined_val:
    ex["ner_tags"] = io_to_bilou(ex["ner_tags"], fewnerd_tags)

print(f"Data: {len(combined_train)} train, {len(combined_val)} val (BILOU)")


# ── Focal Loss ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weights. gamma=2 down-weights easy examples."""

    def __init__(self, gamma=2.0, weight=None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # class weights, applied inside focal term
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,)
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce)  # p_t for the correct class
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ── Entity span extraction (for F1 scoring) ────────────────────────────
def extract_spans(tags):
    """Extract (start, end, type) spans from BILOU tags."""
    spans = set()
    i = 0
    n = len(tags)
    while i < n:
        t = tags[i]
        if t == 0:
            i += 1
        else:
            pos = (t - 1) % 4
            etype = (t - 1) // 4
            if pos == 3:
                spans.add((i, i, etype))
                i += 1  # U
            elif pos == 0:  # B
                start = i
                i += 1
                while i < n and tags[i] != 0 and (tags[i] - 1) % 4 in (1, 2):
                    if (tags[i] - 1) % 4 == 2:
                        spans.add((start, i, etype))
                        i += 1
                        break
                    i += 1
            else:
                i += 1
    return spans


def span_f1(gold_tags, pred_tags):
    """Entity-level exact span match F1 for a list of tag sequences."""
    total_gold, total_pred, total_correct = 0, 0, 0
    for g, p in zip(gold_tags, pred_tags):
        gs = extract_spans(g)
        ps = extract_spans(p)
        total_gold += len(gs)
        total_pred += len(ps)
        total_correct += len(gs & ps)
    p = total_correct / total_pred if total_pred > 0 else 0
    r = total_correct / total_gold if total_gold > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f


# ── Training ───────────────────────────────────────────────────────────
device = torch.device("cuda")
model = AdvancedNER(embed_dim=128, n_classes=33).to(device)
n_classes = 33
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
criterion = FocalLoss(gamma=2.0, ignore_index=-1)


def prepare_batch(keys_list, tags_list, indices):
    bs = len(indices)
    ml = max(len(tags_list[i]) for i in indices)
    x = torch.zeros(bs, ml, 16, dtype=torch.long)
    y = torch.full((bs, ml), -1, dtype=torch.long)
    for j, idx in enumerate(indices):
        n = len(tags_list[idx])
        x[j, :n] = torch.from_numpy(keys_list[idx])
        y[j, :n] = torch.tensor(tags_list[idx])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# Precompute all hash keys once (CPU, runs once — not every epoch)
print("Precomputing hash keys...")
train_keys = [
    compute_hash_keys(ex["tokens"]) for ex in tqdm(combined_train, desc="train keys")
]
val_keys = [
    compute_hash_keys(ex["tokens"]) for ex in tqdm(combined_val, desc="val keys")
]
train_tags = [ex["ner_tags"] for ex in combined_train]
val_tags = [ex["ner_tags"] for ex in combined_val]

# Pre-compute val tag sequences for span F1
val_gold_tags = [ex["ner_tags"] for ex in combined_val]

print(f"\nTraining Advanced NER (FocalLoss γ=2, 128-dim, batch=64, 80ep)...")
BATCH_SIZE = 64
EPOCHS = 80
best_span_f1 = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss, n = 0, 0
    indices = np.random.permutation(len(combined_train))
    for i in tqdm(range(0, len(indices), BATCH_SIZE), desc=f"Epoch {epoch + 1}"):
        x, y = prepare_batch(train_keys, train_tags, indices[i : i + BATCH_SIZE])
        optimizer.zero_grad()
        loss = criterion(model(x).reshape(-1, 33), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    scheduler.step()

    # Validation: token accuracy + entity-level span F1 + per-class F1
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    val_pred_tags = []
    with torch.no_grad():
        for i in range(0, len(combined_val), 64):
            x, y = prepare_batch(
                val_keys, val_tags, list(range(i, min(i + 64, len(combined_val))))
            )
            logits = model(x)
            preds = logits.argmax(-1)
            mask = y != -1
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
            all_preds.extend(preds[mask].cpu().tolist())
            all_labels.extend(y[mask].cpu().tolist())
            # Collect per-sequence preds for span F1
            for j in range(x.shape[0]):
                n_tok = len(combined_val[i + j]["tokens"])
                val_pred_tags.append(preds[j, :n_tok].cpu().tolist())

    acc = correct / total
    sp, sr, sf1 = span_f1(val_gold_tags, val_pred_tags)

    # Per-class F1 for entity types (BILOU collapsed to 9 types)
    def bilou_to_etype(t):
        return 0 if t == 0 else ((t - 1) // 4) + 1

    etype_preds = [bilou_to_etype(p) for p in all_preds]
    etype_labels = [bilou_to_etype(l) for l in all_labels]
    from sklearn.metrics import f1_score as f1s

    per_class_f1 = {}
    for cls_idx in range(9):
        mask_l = [1 if l == cls_idx else 0 for l in etype_labels]
        mask_p = [1 if p == cls_idx else 0 for p in etype_preds]
        if sum(mask_l) > 0:
            per_class_f1[cls_idx] = f1s(mask_l, mask_p, zero_division=0)
    f1_str = " ".join(
        f"{fewnerd_tags[c][:4]}:{per_class_f1.get(c, 0):.3f}" for c in range(9)
    )

    print(
        f"Epoch {epoch + 1}: loss={total_loss / n:.4f} acc={acc:.4f} span_P={sp:.4f} span_R={sr:.4f} span_F1={sf1:.4f}"
    )
    print(f"          per-class F1: {f1_str}")

    if sf1 > best_span_f1:
        best_span_f1 = sf1
        torch.save(
            model.state_dict(),
            "/home/max/Projects/mesh-actors/models/advanced_128_best.pt",
        )
        print(f"  ↑ best (span_F1={sf1:.4f})")

print(f"\nBest span F1: {best_span_f1:.4f}")
