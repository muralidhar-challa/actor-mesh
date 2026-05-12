"""Train 18-type NER v3: CRF + rich features + 3-path multi-scale, no RoPE."""

import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torchcrf import CRF
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════
# Architecture v3: same as v2 but no RoPE, CRF on top
# ══════════════════════════════════════════════════════════════════════════


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


class NERv3(nn.Module):
    """v3: rich features + 3-path + CRF, no RoPE."""

    def __init__(self, embed_dim=128, n_classes=73):
        super().__init__()
        self.n_classes = n_classes
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
        # no RoPE

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
        self.wide_res = nn.ModuleList(
            [SepResBlock(embed_dim, 9), SepResBlock(embed_dim, 15)]
        )
        self.gate = nn.Linear(embed_dim * 3, embed_dim)
        self.ner = nn.Linear(embed_dim, n_classes)
        self.drop = nn.Dropout(0.1)
        self.crf = CRF(n_classes, batch_first=True)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        embeds = []
        for col, he in enumerate(self.hash_embeds):
            s = x[:, :, col * 4 : (col + 1) * 4].reshape(B * S, 4)
            embeds.append(he(s).view(B, S, -1))
        e = self.proj(torch.cat(embeds, -1))
        e = self.norm_in(e)
        # no RoPE
        et = e.transpose(1, 2)
        local = self.local_conv(et).transpose(1, 2)
        local = self.local_norm(local)
        local = self.local_se(local).transpose(1, 2)
        local = torch.nn.functional.hardswish(local)
        g = et
        for res in self.global_res:
            g = res(g)
        w = et
        for res in self.wide_res:
            w = res(w)
        lt = local.transpose(1, 2)
        gt = g.transpose(1, 2)
        wt = w.transpose(1, 2)
        gw = torch.softmax(self.gate(self.drop(torch.cat([lt, gt, wt], -1))), dim=-1)
        fused = gw[:, :, 0:1] * lt + gw[:, :, 1:2] * gt + gw[:, :, 2:3] * wt
        logits = self.ner(self.drop(fused))
        if mask is not None:
            return logits  # during training, return logits for CRF loss
        return logits


# ══════════════════════════════════════════════════════════════════════════
# Span F1
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
# Rich features
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
    S = len(tokens)
    keys = np.zeros((S, 16), dtype=np.uint64)
    for i, tok in enumerate(tokens):
        norm = tok.lower()
        for k in range(4):
            keys[i, 0 * 4 + k] = murmurhash3(norm, 8 * 4 + k)
        pfx = tok[0] if tok else " "
        for k in range(4):
            keys[i, 1 * 4 + k] = murmurhash3(pfx, 9 * 4 + k)
        sfx = tok[-3:] if len(tok) >= 3 else tok
        for k in range(4):
            keys[i, 2 * 4 + k] = murmurhash3(sfx, 10 * 4 + k)
        shape = ner_shape(tok)
        for k in range(4):
            keys[i, 3 * 4 + k] = murmurhash3(shape, 11 * 4 + k)
    return keys


# ══════════════════════════════════════════════════════════════════════════
# Data loading
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

# ══════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════

device = torch.device("cuda")
model = NERv3(embed_dim=128, n_classes=N_CLASSES).to(device)
print(f"\nParams: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

print("Precomputing rich features...")
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
print(f"\nTraining v3 (CRF, no RoPE, batch={BATCH}, {EPOCHS}ep)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss, n = 0, 0
    indices = np.random.permutation(len(combined_train))
    for i in tqdm(range(0, len(indices), BATCH), desc=f"Epoch {epoch + 1}"):
        x, y = prepare_batch(train_keys, train_tags, indices[i : i + BATCH])
        mask = y != -1
        optimizer.zero_grad()
        logits = model(x)
        loss = -model.crf(logits, y, mask=mask, reduction="mean")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    scheduler.step()

    # Validation with Viterbi
    model.eval()
    val_pred_tags = []
    with torch.no_grad():
        for i in range(0, len(combined_val), 64):
            x, y = prepare_batch(
                val_keys, val_tags_list, list(range(i, min(i + 64, len(combined_val))))
            )
            mask = y != -1
            logits = model(x)
            preds = model.crf.decode(logits, mask=mask)
            for j in range(x.shape[0]):
                seq_preds = preds[j]
                val_pred_tags.append(seq_preds)

    # Accuracy and span F1 (use CRF-decoded tags)
    correct, total = 0, 0
    for gold, pred in zip(val_gold_tags, val_pred_tags):
        for g, p in zip(gold, pred):
            if g != -1:
                if g == p:
                    correct += 1
                total += 1
    acc = correct / total if total > 0 else 0
    sp, sr, sf1 = span_f1(val_gold_tags, val_pred_tags)

    # Per-class F1
    all_preds = [t for seq in val_pred_tags for t in seq]
    all_labels = [t for seq in val_gold_tags for t in seq if t != -1]
    all_preds = all_preds[: len(all_labels)]

    from sklearn.metrics import f1_score as f1s

    def bilou_to_etype(t):
        return 0 if t == 0 else ((t - 1) // 4) + 1

    etype_preds = [bilou_to_etype(p) for p in all_preds]
    etype_labels = [bilou_to_etype(l) for l in all_labels]
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
            "/home/max/Projects/mesh-actors/models/onto18_v3_best.pt",
        )
        print(f"  ↑ best (span_F1={sf1:.4f})")

print(f"\nBest span F1: {best_span_f1:.4f}")
