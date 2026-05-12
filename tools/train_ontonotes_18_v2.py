"""Train 18-type NER v2: rich features (NORM/PREFIX/SUFFIX/SHAPE) + multi-scale (local+global+wide)."""

import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════
# Architecture v2 — same backbone, richer features, multi-scale fusion
# ══════════════════════════════════════════════════════════════════════════


class RoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        B, S, E = x.shape
        device = x.device
        pos = torch.arange(S, device=device).float()
        freqs = 1.0 / (10000 ** (torch.arange(0, E, 2, device=device).float() / E))
        theta = pos.unsqueeze(1) * freqs.unsqueeze(0)
        cos = theta.cos().unsqueeze(0)
        sin = theta.sin().unsqueeze(0)
        x_rot = torch.zeros_like(x)
        x_rot[:, :, 0::2] = x[:, :, 0::2] * cos - x[:, :, 1::2] * sin
        x_rot[:, :, 1::2] = x[:, :, 1::2] * cos + x[:, :, 0::2] * sin
        return x_rot


class SEBlock(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.Hardswish(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled = x.mean(dim=1)
        return x * self.se(pooled).unsqueeze(1)


class SepConv(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            dim, dim, 3, padding=dilation, dilation=dilation, groups=dim
        )
        self.pointwise = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SepResBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.conv = SepConv(dim, dilation)
        self.norm = nn.LayerNorm(dim)
        self.se = SEBlock(dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        r = x
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.se(x)
        x = x.transpose(1, 2)
        x = torch.nn.functional.hardswish(x)
        return self.drop(x) + r


class HashEmbed(nn.Module):
    def __init__(self, nV, embed_dim):
        super().__init__()
        self.nV = nV
        self.embed = nn.Embedding(nV, embed_dim)

    def forward(self, col_slice):
        S = col_slice.shape[0]
        out = torch.zeros(S, self.embed.weight.shape[1], device=col_slice.device)
        for k in range(4):
            out += self.embed((col_slice[:, k] % self.nV).long())
        return out


class AdvancedNERv2(nn.Module):
    """v2: rich features + 3-path fusion (local, global, wide)."""

    def __init__(self, embed_dim=128, n_classes=73):
        super().__init__()
        # Rich feature hash embeds: NORM, PREFIX, SUFFIX, SHAPE
        self.hash_embeds = nn.ModuleList(
            [
                HashEmbed(5000, embed_dim),  # NORM
                HashEmbed(1000, embed_dim),  # PREFIX (small vocab — single chars)
                HashEmbed(2500, embed_dim),  # SUFFIX
                HashEmbed(2500, embed_dim),  # SHAPE
            ]
        )
        self.proj = nn.Linear(embed_dim * 4, embed_dim)
        self.norm_in = nn.LayerNorm(embed_dim)
        self.rope = RoPE(embed_dim)

        # Local path: single conv, small field
        self.local_conv = SepConv(embed_dim)
        self.local_norm = nn.LayerNorm(embed_dim)
        self.local_se = SEBlock(embed_dim)

        # Global path: 3 dilated blocks (d=1,3,5) → ±9 context
        self.global_res = nn.ModuleList(
            [
                SepResBlock(embed_dim, 1),
                SepResBlock(embed_dim, 3),
                SepResBlock(embed_dim, 5),
            ]
        )

        # Wide path: 2 heavily dilated blocks (d=9,15) → ±24 context
        self.wide_res = nn.ModuleList(
            [
                SepResBlock(embed_dim, 9),
                SepResBlock(embed_dim, 15),
            ]
        )

        # 3-way gated fusion
        self.gate = nn.Linear(embed_dim * 3, embed_dim)
        self.ner = nn.Linear(embed_dim, n_classes)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        B, S, _ = x.shape
        # Rich feature embedding
        embeds = []
        for col, he in enumerate(self.hash_embeds):
            s = x[:, :, col * 4 : (col + 1) * 4].reshape(B * S, 4)
            embeds.append(he(s).view(B, S, -1))
        e = self.proj(torch.cat(embeds, -1))
        e = self.norm_in(e)
        e = self.rope(e)
        et = e.transpose(1, 2)

        # Local
        local = self.local_conv(et).transpose(1, 2)
        local = self.local_norm(local)
        local = self.local_se(local).transpose(1, 2)
        local = torch.nn.functional.hardswish(local)

        # Global
        g = et
        for res in self.global_res:
            g = res(g)

        # Wide
        w = et
        for res in self.wide_res:
            w = res(w)

        # 3-way gated fusion
        lt = local.transpose(1, 2)
        gt = g.transpose(1, 2)
        wt = w.transpose(1, 2)
        gw = torch.softmax(self.gate(self.drop(torch.cat([lt, gt, wt], -1))), dim=-1)
        fused = gw[:, :, 0:1] * lt + gw[:, :, 1:2] * gt + gw[:, :, 2:3] * wt
        return self.ner(self.drop(fused))


# ══════════════════════════════════════════════════════════════════════════
# Focal Loss
# ══════════════════════════════════════════════════════════════════════════


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ══════════════════════════════════════════════════════════════════════════
# Span F1 helpers
# ══════════════════════════════════════════════════════════════════════════


def extract_spans(tags):
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
                i += 1
            elif pos == 0:
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
    tg, tp, tc = 0, 0, 0
    for g, p in zip(gold_tags, pred_tags):
        gs = extract_spans(g)
        ps = extract_spans(p)
        tg += len(gs)
        tp += len(ps)
        tc += len(gs & ps)
    p = tc / tp if tp > 0 else 0
    r = tc / tg if tg > 0 else 0
    return p, r, 2 * p * r / (p + r) if (p + r) > 0 else 0


# ══════════════════════════════════════════════════════════════════════════
# Rich feature extraction (C-compatible: NORM/PREFIX/SUFFIX/SHAPE)
# ══════════════════════════════════════════════════════════════════════════


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


def ner_shape(tok):
    out = []
    last = "\0"
    seq = 0
    for c in tok:
        if c.isupper():
            sc = "X"
        elif c.islower():
            sc = "x"
        elif c.isdigit():
            sc = "d"
        else:
            sc = c
        if sc == last:
            seq += 1
        else:
            seq = 0
            last = sc
        if seq < 4:
            out.append(sc)
    return "".join(out)


def compute_hash_keys_rich(tokens):
    """Rich features: NORM(seed=8), PREFIX(seed=9), SUFFIX(seed=10), SHAPE(seed=11).
    Each column gets 4 murmurhash3 variants: seed*4+k for k=0..3."""
    S = len(tokens)
    keys = np.zeros((S, 16), dtype=np.uint64)
    for i, tok in enumerate(tokens):
        # NORM: full lowercased token
        norm = tok.lower()
        for k in range(4):
            keys[i, 0 * 4 + k] = murmurhash3(norm, 8 * 4 + k)
        # PREFIX: first character
        pfx = tok[0] if tok else " "
        for k in range(4):
            keys[i, 1 * 4 + k] = murmurhash3(pfx, 9 * 4 + k)
        # SUFFIX: last 3 characters
        sfx = tok[-3:] if len(tok) >= 3 else tok
        for k in range(4):
            keys[i, 2 * 4 + k] = murmurhash3(sfx, 10 * 4 + k)
        # SHAPE: orthographic pattern
        shape = ner_shape(tok)
        for k in range(4):
            keys[i, 3 * 4 + k] = murmurhash3(shape, 11 * 4 + k)
    return keys


# ══════════════════════════════════════════════════════════════════════════
# Data loading (same as train_ontonotes_18.py)
# ══════════════════════════════════════════════════════════════════════════


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


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


print("Loading data...")

SPACY_TYPES = [
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
]
TYPE2ID = {t: i + 1 for i, t in enumerate(SPACY_TYPES)}
print(f"Entity types: {len(SPACY_TYPES)}, BILOU classes: {len(SPACY_TYPES) * 4 + 1}")

# Few-NERD
fewnerd = load_dataset("DFKI-SLT/few-nerd", "supervised")
FEWNERD_MAP = {
    0: 0,
    1: TYPE2ID["WORK_OF_ART"],
    2: TYPE2ID["FAC"],
    3: TYPE2ID["EVENT"],
    4: TYPE2ID["LOC"],
    5: TYPE2ID["ORG"],
    6: 0,
    7: TYPE2ID["PERSON"],
    8: TYPE2ID["PRODUCT"],
}

# WNUT
wnut = pickle.load(open("/tmp/wnut.pkl", "rb"))
WNUT_MAP = {
    0: 0,
    1: TYPE2ID["ORG"],
    2: TYPE2ID["ORG"],
    3: TYPE2ID["WORK_OF_ART"],
    4: TYPE2ID["WORK_OF_ART"],
    5: TYPE2ID["ORG"],
    6: TYPE2ID["ORG"],
    7: TYPE2ID["LOC"],
    8: TYPE2ID["LOC"],
    9: TYPE2ID["PERSON"],
    10: TYPE2ID["PERSON"],
    11: TYPE2ID["PRODUCT"],
    12: TYPE2ID["PRODUCT"],
}

# CoNLL
conll_train_s, conll_train_t = load_conll("/tmp/train.conll")
conll_dev_s, conll_dev_t = load_conll("/tmp/dev.conll")
CONLL_MAP = {
    "O": 0,
    "B-PER": TYPE2ID["PERSON"],
    "I-PER": TYPE2ID["PERSON"],
    "B-ORG": TYPE2ID["ORG"],
    "I-ORG": TYPE2ID["ORG"],
    "B-LOC": TYPE2ID["LOC"],
    "I-LOC": TYPE2ID["LOC"],
    "B-MISC": 0,
    "I-MISC": 0,
}

# OntoNotes
repo = "tner/ontonotes5"
rt = "dataset"
onto_labels = json.load(open(hf_hub_download(repo, "dataset/label.json", repo_type=rt)))
ONTO_MAP = {}
for ln, lid in onto_labels.items():
    if ln == "O":
        ONTO_MAP[lid] = 0
    else:
        _, et = ln.split("-")
        ONTO_MAP[lid] = TYPE2ID[et] if et in TYPE2ID else 0

onto_train = []
for fn in ["train00.json", "train01.json", "train02.json", "train03.json"]:
    onto_train.extend(load_jsonl(hf_hub_download(repo, f"dataset/{fn}", repo_type=rt)))
onto_train = [
    {"tokens": ex["tokens"], "ner_tags": [ONTO_MAP.get(t, 0) for t in ex["tags"]]}
    for ex in onto_train
]

onto_val_path = hf_hub_download(repo, "dataset/valid.json", repo_type=rt)
onto_val = [
    {"tokens": ex["tokens"], "ner_tags": [ONTO_MAP.get(t, 0) for t in ex["tags"]]}
    for ex in load_jsonl(onto_val_path)
]

print(f"Few-NERD: {len(fewnerd['train'])} train / {len(fewnerd['validation'])} val")
print(f"WNUT:     {len(wnut['train'])} train / {len(wnut['validation'])} val")
print(f"OntoNotes:{len(onto_train)} train / {len(onto_val)} val")


# Merge
def hf_to_examples(ds, m):
    return [
        {"tokens": ex["tokens"], "ner_tags": [m[t] for t in ex["ner_tags"]]}
        for ex in ds
    ]


combined_train = (
    hf_to_examples(fewnerd["train"], FEWNERD_MAP)
    + [
        {
            "tokens": ex["tokens"],
            "ner_tags": [WNUT_MAP.get(t, 0) for t in ex["ner_tags"]],
        }
        for ex in wnut["train"]
    ]
    + [
        {"tokens": s, "ner_tags": [CONLL_MAP.get(t, 0) for t in t]}
        for s, t in zip(conll_train_s, conll_train_t)
    ]
    + onto_train
)
combined_val = (
    hf_to_examples(fewnerd["validation"], FEWNERD_MAP)
    + [
        {
            "tokens": ex["tokens"],
            "ner_tags": [WNUT_MAP.get(t, 0) for t in ex["ner_tags"]],
        }
        for ex in wnut["validation"]
    ]
    + [
        {"tokens": s, "ner_tags": [CONLL_MAP.get(t, 0) for t in t]}
        for s, t in zip(conll_dev_s, conll_dev_t)
    ]
    + onto_val
)


# IO → BILOU
def io_to_bilou(tags, n_types):
    bilou = []
    n = len(tags)
    i = 0
    while i < n:
        t = tags[i]
        if t == 0:
            bilou.append(0)
            i += 1
        else:
            start = i
            while i < n and tags[i] == t:
                i += 1
            length = i - start
            etype = t - 1
            if length == 1:
                bilou.append(etype * 4 + 3 + 1)
            else:
                bilou.append(etype * 4 + 0 + 1)
                for _ in range(1, length - 1):
                    bilou.append(etype * 4 + 1 + 1)
                bilou.append(etype * 4 + 2 + 1)
    return bilou


N_TYPES = len(SPACY_TYPES)
N_CLASSES = N_TYPES * 4 + 1
for ex in combined_train:
    ex["ner_tags"] = io_to_bilou(ex["ner_tags"], N_TYPES)
for ex in combined_val:
    ex["ner_tags"] = io_to_bilou(ex["ner_tags"], N_TYPES)

total_tok = sum(len(ex["tokens"]) for ex in combined_train)
print(f"\nTrain: {len(combined_train)} ex, {total_tok:,} tokens")
print(
    f"Val:   {len(combined_val)} ex, {sum(len(ex['tokens']) for ex in combined_val):,} tokens"
)
print(f"Classes: {N_CLASSES}")

# ══════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════

device = torch.device("cuda")
model = AdvancedNERv2(embed_dim=128, n_classes=N_CLASSES).to(device)
print(f"\nParams: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
criterion = FocalLoss(gamma=2.0, ignore_index=-1)

# Precompute rich features
print("Precomputing rich features (NORM/PREFIX/SUFFIX/SHAPE)...")
train_keys = [
    compute_hash_keys_rich(ex["tokens"])
    for ex in tqdm(combined_train, desc="train keys")
]
val_keys = [
    compute_hash_keys_rich(ex["tokens"]) for ex in tqdm(combined_val, desc="val keys")
]
train_tags = [ex["ner_tags"] for ex in combined_train]
val_tags_list = [ex["ner_tags"] for ex in combined_val]
val_gold_tags = val_tags_list


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


BATCH = 64
EPOCHS = 80
best_span_f1 = 0
print(
    f"\nTraining v2 (rich features, 3-path multi-scale, batch={BATCH}, {EPOCHS}ep)..."
)

for epoch in range(EPOCHS):
    model.train()
    total_loss, n = 0, 0
    indices = np.random.permutation(len(combined_train))
    for i in tqdm(range(0, len(indices), BATCH), desc=f"Epoch {epoch + 1}"):
        x, y = prepare_batch(train_keys, train_tags, indices[i : i + BATCH])
        optimizer.zero_grad()
        loss = criterion(model(x).reshape(-1, N_CLASSES), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    scheduler.step()

    # Validation
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    val_pred_tags = []
    with torch.no_grad():
        for i in range(0, len(combined_val), 64):
            x, y = prepare_batch(
                val_keys, val_tags_list, list(range(i, min(i + 64, len(combined_val))))
            )
            logits = model(x)
            preds = logits.argmax(-1)
            mask = y != -1
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
            all_preds.extend(preds[mask].cpu().tolist())
            all_labels.extend(y[mask].cpu().tolist())
            for j in range(x.shape[0]):
                val_pred_tags.append(
                    preds[j, : len(combined_val[i + j]["tokens"])].cpu().tolist()
                )

    acc = correct / total
    sp, sr, sf1 = span_f1(val_gold_tags, val_pred_tags)

    def bilou_to_etype(t):
        return 0 if t == 0 else ((t - 1) // 4) + 1

    etype_preds = [bilou_to_etype(p) for p in all_preds]
    etype_labels = [bilou_to_etype(l) for l in all_labels]
    from sklearn.metrics import f1_score as f1s

    type_names = ["O"] + SPACY_TYPES
    per_class_f1 = {}
    for cls_idx in range(19):
        mask_l = [1 if l == cls_idx else 0 for l in etype_labels]
        mask_p = [1 if p == cls_idx else 0 for p in etype_preds]
        if sum(mask_l) > 0:
            per_class_f1[cls_idx] = f1s(mask_l, mask_p, zero_division=0)
    f1_str = " ".join(
        f"{type_names[c][:4]}:{per_class_f1.get(c, 0):.3f}"
        for c in range(min(10, len(type_names)))
    )

    print(
        f"Epoch {epoch + 1}: loss={total_loss / n:.4f} acc={acc:.4f} span_P={sp:.4f} span_R={sr:.4f} span_F1={sf1:.4f}"
    )
    print(f"          F1: {f1_str}")

    if sf1 > best_span_f1:
        best_span_f1 = sf1
        torch.save(
            model.state_dict(),
            "/home/max/Projects/mesh-actors/models/onto18_v2_best.pt",
        )
        print(f"  ↑ best (span_F1={sf1:.4f})")

print(f"\nBest span F1: {best_span_f1:.4f}")
