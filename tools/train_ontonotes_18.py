"""Train on Few-NERD + WNUT + CoNLL + OntoNotes5 → 18 spaCy-compatible types, 73 BILOU classes."""

import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


# ── Architecture (same as train_advanced.py) ───────────────────────────
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
        gate = self.se(pooled).unsqueeze(1)
        return x * gate


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
        residual = x
        out = self.conv(x)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.se(out)
        out = out.transpose(1, 2)
        out = torch.nn.functional.hardswish(out)
        out = self.drop(out)
        return out + residual


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


class AdvancedNER(nn.Module):
    def __init__(self, embed_dim=128, n_classes=73):
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
        self.local_conv = SepConv(embed_dim)
        self.local_norm = nn.LayerNorm(embed_dim)
        self.local_se = SEBlock(embed_dim)
        self.global_res = nn.ModuleList(
            [
                SepResBlock(embed_dim, 1),
                SepResBlock(embed_dim, 3),
                SepResBlock(embed_dim, 5),
            ]
        )
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.ner = nn.Linear(embed_dim, n_classes)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        B, S, _ = x.shape
        embeds = []
        for col, he in enumerate(self.hash_embeds):
            col_slice = x[:, :, col * 4 : (col + 1) * 4].reshape(B * S, 4)
            emb = he(col_slice)
            embeds.append(emb.view(B, S, -1))
        emb = self.proj(torch.cat(embeds, dim=-1))
        emb = self.norm_in(emb)
        emb = self.rope(emb)
        emb_t = emb.transpose(1, 2)
        local = self.local_conv(emb_t).transpose(1, 2)
        local = self.local_norm(local)
        local = self.local_se(local).transpose(1, 2)
        local = torch.nn.functional.hardswish(local)
        g = emb_t
        for res in self.global_res:
            g = res(g)
        local_t = local.transpose(1, 2)
        g_t = g.transpose(1, 2)
        gate_w = torch.sigmoid(self.gate(self.drop(torch.cat([local_t, g_t], dim=-1))))
        fused = gate_w * local_t + (1 - gate_w) * g_t
        return self.ner(self.drop(fused))


# ── Focal Loss ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Span extraction ────────────────────────────────────────────────────
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
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f


# ── Hash functions ─────────────────────────────────────────────────────
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


# ── Data loading ───────────────────────────────────────────────────────
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
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


print("Loading data...")

# ── 18-type definition (spaCy compatible) ──────────────────────────────
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
TYPE2ID = {t: i + 1 for i, t in enumerate(SPACY_TYPES)}  # 1..18, 0=O
print(f"Entity types ({len(SPACY_TYPES)}): {SPACY_TYPES}")
print(f"BILOU classes: {len(SPACY_TYPES) * 4 + 1}")

# ── Few-NERD (8 types → 18) ────────────────────────────────────────────
fewnerd = load_dataset("DFKI-SLT/few-nerd", "supervised")
fewnerd_tags = fewnerd["train"].features["ner_tags"].feature.names
# Map: O=0, art→WORK_OF_ART, building→FAC, event→EVENT, location→LOC, organization→ORG, other→O, person→PERSON, product→PRODUCT
FEWNERD_MAP = {
    0: 0,  # O
    1: TYPE2ID["WORK_OF_ART"],  # art
    2: TYPE2ID["FAC"],  # building
    3: TYPE2ID["EVENT"],  # event
    4: TYPE2ID["LOC"],  # location
    5: TYPE2ID["ORG"],  # organization
    6: 0,  # other → O
    7: TYPE2ID["PERSON"],  # person
    8: TYPE2ID["PRODUCT"],  # product
}

# ── WNUT (same mapping as before, adapted to 18 types) ─────────────────
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

# ── CoNLL ──────────────────────────────────────────────────────────────
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

# ── OntoNotes 5 ────────────────────────────────────────────────────────
repo = "tner/ontonotes5"
repo_type = "dataset"
onto_labels = json.load(
    open(hf_hub_download(repo, "dataset/label.json", repo_type=repo_type))
)
# Build BIO tag → type ID mapping (collapse B/I → same type)
ONTO_MAP = {}
for label_name, label_id in onto_labels.items():
    if label_name == "O":
        ONTO_MAP[label_id] = 0
    else:
        prefix, etype = label_name.split("-")
        if etype in TYPE2ID:
            ONTO_MAP[label_id] = TYPE2ID[etype]
        else:
            ONTO_MAP[label_id] = 0  # unknown → O

# Load OntoNotes train files
onto_train = []
for fname in ["train00.json", "train01.json", "train02.json", "train03.json"]:
    path = hf_hub_download(repo, f"dataset/{fname}", repo_type=repo_type)
    onto_train.extend(load_jsonl(path))
onto_train = [
    {"tokens": ex["tokens"], "ner_tags": [ONTO_MAP.get(t, 0) for t in ex["tags"]]}
    for ex in onto_train
]

# Load OntoNotes valid
onto_val_path = hf_hub_download(repo, "dataset/valid.json", repo_type=repo_type)
onto_val = [
    {"tokens": ex["tokens"], "ner_tags": [ONTO_MAP.get(t, 0) for t in ex["tags"]]}
    for ex in load_jsonl(onto_val_path)
]

print(f"OntoNotes: {len(onto_train)} train, {len(onto_val)} val")


# ── Merge all ──────────────────────────────────────────────────────────
def hf_to_examples(dataset, tag_map):
    return [
        {"tokens": ex["tokens"], "ner_tags": [tag_map[t] for t in ex["ner_tags"]]}
        for ex in dataset
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


# ── IO → BILOU conversion ──────────────────────────────────────────────
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
            etype = t - 1  # 0..17
            if length == 1:
                bilou.append(etype * 4 + 3 + 1)  # U
            else:
                bilou.append(etype * 4 + 0 + 1)  # B
                for j in range(1, length - 1):
                    bilou.append(etype * 4 + 1 + 1)  # I
                bilou.append(etype * 4 + 2 + 1)  # L
    return bilou


N_TYPES = len(SPACY_TYPES)
N_CLASSES = N_TYPES * 4 + 1

for ex in combined_train:
    ex["ner_tags"] = io_to_bilou(ex["ner_tags"], N_TYPES)
for ex in combined_val:
    ex["ner_tags"] = io_to_bilou(ex["ner_tags"], N_TYPES)

# ── Stats ──────────────────────────────────────────────────────────────
from collections import Counter

tag_counts = Counter()
for ex in combined_train:
    for t in ex["ner_tags"]:
        tag_counts[t] += 1
total_tok = sum(tag_counts.values())
o_count = tag_counts[0]
print(f"\nTrain: {len(combined_train)} ex, {total_tok:,} tokens")
print(
    f"Val:   {len(combined_val)} ex, {sum(len(ex['tokens']) for ex in combined_val):,} tokens"
)
print(f"Classes: {N_CLASSES}")
print(f"O: {o_count:,} ({100 * o_count / total_tok:.1f}%)")

# Show per-type counts
print("\nPer-type token distribution:")
for i, name in enumerate(SPACY_TYPES):
    b = tag_counts.get(i * 4 + 1, 0)
    ii = tag_counts.get(i * 4 + 2, 0)
    l = tag_counts.get(i * 4 + 3, 0)
    u = tag_counts.get(i * 4 + 4, 0)
    total = b + ii + l + u
    if total > 0:
        print(
            f"  {name:15s}: B={b:>6,} I={ii:>6,} L={l:>6,} U={u:>6,}  total={total:>8,}"
        )

# ── Training ───────────────────────────────────────────────────────────
device = torch.device("cuda")
model = AdvancedNER(embed_dim=128, n_classes=N_CLASSES).to(device)
print(f"\nParams: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
criterion = FocalLoss(gamma=2.0, ignore_index=-1)

# Precompute hash keys
print("Precomputing hash keys...")
train_keys = [
    compute_hash_keys(ex["tokens"]) for ex in tqdm(combined_train, desc="train keys")
]
val_keys = [
    compute_hash_keys(ex["tokens"]) for ex in tqdm(combined_val, desc="val keys")
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


BATCH_SIZE = 64
EPOCHS = 80
best_span_f1 = 0
print(f"\nTraining 18-type NER (FocalLoss, 128-dim, batch={BATCH_SIZE}, {EPOCHS}ep)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss, n = 0, 0
    indices = np.random.permutation(len(combined_train))
    for i in tqdm(range(0, len(indices), BATCH_SIZE), desc=f"Epoch {epoch + 1}"):
        x, y = prepare_batch(train_keys, train_tags, indices[i : i + BATCH_SIZE])
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
                n_tok = len(combined_val[i + j]["tokens"])
                val_pred_tags.append(preds[j, :n_tok].cpu().tolist())

    acc = correct / total
    sp, sr, sf1 = span_f1(val_gold_tags, val_pred_tags)

    # Per-class F1 (collapsed to 19 types for readability)
    def bilou_to_etype(t):
        return 0 if t == 0 else ((t - 1) // 4) + 1

    etype_preds = [bilou_to_etype(p) for p in all_preds]
    etype_labels = [bilou_to_etype(l) for l in all_labels]
    from sklearn.metrics import f1_score as f1s

    per_class_f1 = {}
    for cls_idx in range(19):
        mask_l = [1 if l == cls_idx else 0 for l in etype_labels]
        mask_p = [1 if p == cls_idx else 0 for p in etype_preds]
        if sum(mask_l) > 0:
            per_class_f1[cls_idx] = f1s(mask_l, mask_p, zero_division=0)

    type_names = ["O"] + SPACY_TYPES
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
            model.state_dict(), "/home/max/Projects/mesh-actors/models/onto18_best.pt"
        )
        print(f"  ↑ best (span_F1={sf1:.4f})")

print(f"\nBest span F1: {best_span_f1:.4f}")
