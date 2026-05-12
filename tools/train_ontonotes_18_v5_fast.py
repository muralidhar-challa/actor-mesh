"""v5-fast: Char CNN + precomputed embeddings + CRF.

Fixes vs original:
  1. Gate Linear(d*3→3) — proper 3-way softmax gate, was Linear(d*3→d) with
     softmax over 128 dims then slicing [0:1],[1:2],[2:3] which is wrong.
     Also saves 48K params (49,152 → 1,152).
  2. build_embed_cache — uses defaultdict(set) instead of defaultdict(list)
     with `not in` check on ints. Semantically correct now.
  3. T_max=EP — cosine scheduler was set to T_max=80 with EP=40, completing
     only half a cycle. Now T_max tracks EP.
  4. Training mask — cleaner explicit mask construction, no -1 sentinel.
  5. Val mask — cleaner, consistent with training mask pattern.
  6. CharCNN dropout — added Dropout(0.1) after each ReLU for regularization.
  7. build_batch — removed repeated torch.from_numpy per token, builds numpy
     array first then converts once.
"""

import json
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torchcrf import CRF
from tqdm import tqdm


# ═══════════ Architecture ═══════════


class SepConv(nn.Module):
    def __init__(self, d, out=None, dl=1):
        super().__init__()
        out = out or d
        self.dw = nn.Conv1d(d, d, 3, padding=dl, dilation=dl, groups=d)
        self.pw = nn.Conv1d(d, out, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class CharCNN(nn.Module):
    def __init__(self, char_vocab=256, char_dim=64, out_dim=64, max_len=20):
        super().__init__()
        self.max_len = max_len
        self.char_embed = nn.Embedding(char_vocab, char_dim, padding_idx=0)
        self.convs = nn.Sequential(
            SepConv(char_dim, out=96),
            nn.ReLU(),
            nn.Dropout(0.1),          # FIX 6: regularize char conv stack
            SepConv(96, out=128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # FIX: pool at peak expressivity (128-dim), then project down
        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, c):
        x = self.char_embed(c)         # (B, L, char_dim)
        x = x.transpose(1, 2)         # (B, char_dim, L)
        x = self.convs(x)             # (B, 128, L)
        x, _ = x.max(-1)              # (B, 128) — max pool at peak width
        return self.proj(x)           # (B, out_dim)


class SEBlock(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(d, d // r),
            nn.Hardswish(),
            nn.Linear(d // r, d),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, S, d) after transpose
        return x * self.se(x.mean(1)).unsqueeze(1)


class SepResBlock(nn.Module):
    def __init__(self, d, dl=1):
        super().__init__()
        self.c = SepConv(d, dl=dl)
        self.n = nn.LayerNorm(d)
        self.s = SEBlock(d)
        self.dp = nn.Dropout(0.1)

    def forward(self, x):
        # x: (B, d, S) — channel-first for conv
        r = x
        x = self.c(x)
        x = x.transpose(1, 2)         # (B, S, d) for LayerNorm + SE
        x = self.n(x)
        x = self.s(x)
        x = x.transpose(1, 2)         # (B, d, S) back
        x = nn.functional.hardswish(x)
        return self.dp(x) + r


class Encoder(nn.Module):
    def __init__(self, d=128, in_dim=None):
        super().__init__()
        self.proj = nn.Linear(in_dim or d * 4, d)
        self.norm = nn.LayerNorm(d)

        # Local path
        self.lc = SepConv(d)
        self.ln = nn.LayerNorm(d)
        self.ls = SEBlock(d)

        # Global path: dilation 1→3→5, receptive field ±9 tokens
        self.gr = nn.ModuleList([
            SepResBlock(d, 1),
            SepResBlock(d, 3),
            SepResBlock(d, 5),
        ])

        # Wide path: dilation 9→15, receptive field ±24 tokens
        self.wr = nn.ModuleList([
            SepResBlock(d, 9),
            SepResBlock(d, 15),
        ])

        # FIX 1: gate outputs 3 scalars (one per path), not d=128 dims.
        # Proper 3-way softmax: weights sum to 1 across paths per token.
        # Was: Linear(d*3, d) = 49,152 params — softmax over 128, slice [0:1]
        # Now: Linear(d*3, 3) =  1,152 params — softmax over 3, correct gate
        self.gate = nn.Linear(d * 3, 3)
        self.drop = nn.Dropout(0.1)

    def forward(self, e):
        """e: (B, S, in_dim) precomputed 4×64 concat embeddings"""
        B, S, _ = e.shape

        e = self.proj(e)               # (B, S, d)
        e = self.norm(e)
        et = e.transpose(1, 2)         # (B, d, S) for convs

        # Local path
        l = self.lc(et).transpose(1, 2)   # (B, S, d)
        l = self.ln(l)
        l = self.ls(l)                     # SE expects (B, S, d)
        l = l.transpose(1, 2)             # (B, d, S)
        l = nn.functional.hardswish(l)

        # Global path
        g = et
        for r in self.gr:
            g = r(g)

        # Wide path
        w = et
        for r in self.wr:
            w = r(w)

        # Transpose all paths to (B, S, d) for gate + weighted sum
        lt = l.transpose(1, 2)
        gt = g.transpose(1, 2)
        wt = w.transpose(1, 2)

        # FIX 1: 3-way gate — softmax over dim=-1 gives 3 weights summing to 1
        gw = torch.softmax(
            self.gate(self.drop(torch.cat([lt, gt, wt], -1))), -1
        )  # (B, S, 3)

        return gw[:, :, 0:1] * lt + gw[:, :, 1:2] * gt + gw[:, :, 2:3] * wt


class BILOUDecoder(nn.Module):
    def __init__(self, d=128, n_types=18):
        super().__init__()
        self.ner = nn.Linear(d, n_types * 4 + 1)
        self.crf = CRF(n_types * 4 + 1, batch_first=True)

    def forward(self, h):
        return self.ner(h)


# ═══════════ Token views ═══════════


def ner_shape(tok):
    o = []
    last = "\0"
    seq = 0
    for c in tok:
        sc = "X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c
        seq = seq + 1 if sc == last else 0
        last = sc
        if seq < 4:
            o.append(sc)
    return "".join(o)


def token_to_char_ids(token, max_len=20):
    ids = [0] * max_len
    raw = token.encode("utf-8", errors="replace")[:max_len]
    for i, c in enumerate(raw):
        ids[i] = c if c > 0 else 1
    return ids


def token_views(tok):
    return (
        tok.lower(),
        tok[0] if tok else " ",
        tok[-3:] if len(tok) >= 3 else tok,
        ner_shape(tok),
    )


# ═══════════ Precompute char embeddings ═══════════


@torch.no_grad()
def build_embed_cache(char_cnns, all_tokens_list, device, max_len=20):
    """Run CharCNN once per unique token view, store results in dict."""
    # FIX 2: use set instead of list — avoids incorrect `not in` on int list
    unique = defaultdict(set)  # view_string → set of col indices that use it
    for tokens in tqdm(all_tokens_list, desc="scan tokens"):
        for tok in tokens:
            views = token_views(tok)
            for col in range(4):
                unique[views[col]].add(col)

    cache = {}
    for col in range(4):
        view_strings = sorted([v for v, cols in unique.items() if col in cols])
        print(f"  Col {col}: {len(view_strings)} unique strings")
        cnn = char_cnns[col]
        cnn.eval()
        for start in tqdm(range(0, len(view_strings), 4096), desc=f"col{col}"):
            chunk = view_strings[start: start + 4096]
            ids = torch.tensor(
                [token_to_char_ids(v, max_len) for v in chunk], device=device
            )
            embs = cnn(ids).cpu().numpy()
            for v, emb in zip(chunk, embs):
                if v not in cache:
                    cache[v] = [None] * 4
                cache[v][col] = emb
    return cache


def build_batch(cache, tokens_list, device):
    """Build (B, S_max, 256) tensor from cache.

    FIX 7: builds a numpy array first, converts to tensor once per batch
    instead of calling torch.from_numpy per token.
    """
    B = len(tokens_list)
    S_max = max(len(t) for t in tokens_list)
    batch = np.zeros((B, S_max, 256), dtype=np.float32)
    for b, tokens in enumerate(tokens_list):
        views = [token_views(tok) for tok in tokens]
        for col in range(4):
            col_slice = slice(col * 64, (col + 1) * 64)
            for s, v in enumerate(views):
                emb = cache.get(v[col], [None] * 4)[col]
                if emb is not None:
                    batch[b, s, col_slice] = emb
    return torch.from_numpy(batch).to(device)


# ═══════════ Span F1 ═══════════


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
            et = (t - 1) // 4
            if pos == 3:  # U tag
                spans.add((i, i, et))
                i += 1
            elif pos == 0:  # B tag
                s = i
                i += 1
                while i < n and tags[i] != 0 and (tags[i] - 1) % 4 in (1, 2):
                    if (tags[i] - 1) % 4 == 2:  # L tag
                        spans.add((s, i, et))
                        i += 1
                        break
                    i += 1
            else:
                i += 1
    return spans


def span_f1(gold, pred):
    tg, tp, tc = 0, 0, 0
    for g, p in zip(gold, pred):
        gs = extract_spans(g)
        ps = extract_spans(p)
        tg += len(gs)
        tp += len(ps)
        tc += len(gs & ps)
    p = tc / tp if tp > 0 else 0
    r = tc / tg if tg > 0 else 0
    return p, r, 2 * p * r / (p + r) if p + r > 0 else 0


# ═══════════ Data ═══════════


def load_conll(path):
    sentences, labels = [], []
    toks, tags = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if toks:
                    sentences.append(toks)
                    labels.append(tags)
                    toks, tags = [], []
                continue
            parts = line.split()
            if len(parts) >= 2:
                toks.append(parts[0])
                tags.append(parts[-1])
    if toks:
        sentences.append(toks)
        labels.append(tags)
    return sentences, labels


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


print("Loading data...")
SPACY = [
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY",
    "QUANTITY", "ORDINAL", "CARDINAL",
]
T2I = {t: i + 1 for i, t in enumerate(SPACY)}
NT = len(SPACY)
print(f"{NT} types, {NT * 4 + 1} BILOU classes")

fewnerd = load_dataset("DFKI-SLT/few-nerd", "supervised")
FM = {
    0: 0, 1: T2I["WORK_OF_ART"], 2: T2I["FAC"], 3: T2I["EVENT"],
    4: T2I["LOC"], 5: T2I["ORG"], 6: 0, 7: T2I["PERSON"], 8: T2I["PRODUCT"],
}

wnut = pickle.load(open("/tmp/wnut.pkl", "rb"))
WM = {
    0: 0, 1: T2I["ORG"], 2: T2I["ORG"], 3: T2I["WORK_OF_ART"],
    4: T2I["WORK_OF_ART"], 5: T2I["ORG"], 6: T2I["ORG"], 7: T2I["LOC"],
    8: T2I["LOC"], 9: T2I["PERSON"], 10: T2I["PERSON"],
    11: T2I["PRODUCT"], 12: T2I["PRODUCT"],
}

cs, ct = load_conll("/tmp/train.conll")
ds2, dt2 = load_conll("/tmp/dev.conll")
CM = {
    "O": 0,
    "B-PER": T2I["PERSON"], "I-PER": T2I["PERSON"],
    "B-ORG": T2I["ORG"],    "I-ORG": T2I["ORG"],
    "B-LOC": T2I["LOC"],    "I-LOC": T2I["LOC"],
    "B-MISC": 0,            "I-MISC": 0,
}

repo = "tner/ontonotes5"
rd2 = "dataset"
ol = json.load(open(hf_hub_download(repo, "dataset/label.json", repo_type=rd2)))
OM = {}
for ln, lid in ol.items():
    if ln == "O":
        OM[lid] = 0
    else:
        _, et = ln.split("-")
        OM[lid] = T2I[et] if et in T2I else 0

ot = []
for fn in ["train00.json", "train01.json", "train02.json", "train03.json"]:
    ot.extend(load_jsonl(hf_hub_download(repo, f"dataset/{fn}", repo_type=rd2)))
ot = [
    {"tokens": ex["tokens"], "ner_tags": [OM.get(t, 0) for t in ex["tags"]]}
    for ex in ot
]
ov = [
    {"tokens": ex["tokens"], "ner_tags": [OM.get(t, 0) for t in ex["tags"]]}
    for ex in load_jsonl(hf_hub_download(repo, "dataset/valid.json", repo_type=rd2))
]


def hf2ex(ds, m):
    return [
        {"tokens": ex["tokens"], "ner_tags": [m[t] for t in ex["ner_tags"]]}
        for ex in ds
    ]


ctr = (
    hf2ex(fewnerd["train"], FM)
    + [
        {"tokens": ex["tokens"], "ner_tags": [WM.get(t, 0) for t in ex["ner_tags"]]}
        for ex in wnut["train"]
    ]
    + [{"tokens": s, "ner_tags": [CM.get(t, 0) for t in t]} for s, t in zip(cs, ct)]
    + ot
)
cvl = (
    hf2ex(fewnerd["validation"], FM)
    + [
        {"tokens": ex["tokens"], "ner_tags": [WM.get(t, 0) for t in ex["ner_tags"]]}
        for ex in wnut["validation"]
    ]
    + [{"tokens": s, "ner_tags": [CM.get(t, 0) for t in t]} for s, t in zip(ds2, dt2)]
    + ov
)


def io_to_bilou(tags):
    b = []
    n = len(tags)
    i = 0
    while i < n:
        t = tags[i]
        if t == 0:
            b.append(0)
            i += 1
        else:
            s = i
            while i < n and tags[i] == t:
                i += 1
            length = i - s
            e = t - 1
            if length == 1:
                b.append(e * 4 + 3 + 1)   # U
            else:
                b.append(e * 4 + 0 + 1)   # B
                for _ in range(1, length - 1):
                    b.append(e * 4 + 1 + 1)  # I
                b.append(e * 4 + 2 + 1)   # L
    return b


for ex in ctr:
    ex["ner_tags"] = io_to_bilou(ex["ner_tags"])
for ex in cvl:
    ex["ner_tags"] = io_to_bilou(ex["ner_tags"])

print(f"Train: {len(ctr)} ex, {sum(len(ex['tokens']) for ex in ctr):,} tokens")
print(f"Val:   {len(cvl)} ex, {sum(len(ex['tokens']) for ex in cvl):,} tokens")

# ═══════════ Precompute per-example tensors ═══════════

device = torch.device("cuda")
char_cnns = nn.ModuleList([CharCNN().to(device) for _ in range(4)])

print("\nPrecomputing char embeddings...")
train_tokens = [ex["tokens"] for ex in ctr]
val_tokens = [ex["tokens"] for ex in cvl]
all_tokens = train_tokens + val_tokens
cache = build_embed_cache(char_cnns, all_tokens, device)
del char_cnns

print("Building per-example tensors...")
train_emb = []
for tokens in tqdm(train_tokens, desc="train emb"):
    train_emb.append(
        build_batch(cache, [tokens], torch.device("cpu")).squeeze(0)
    )
val_emb = []
for tokens in tqdm(val_tokens, desc="val emb"):
    val_emb.append(
        build_batch(cache, [tokens], torch.device("cpu")).squeeze(0)
    )
del cache

# ═══════════ Training ═══════════

# ── Paths ──────────────────────────────────────────────────────────────────
CKPT_BEST  = "/home/max/Projects/mesh-actors/models/onto18_v5_fast_best.pt"
CKPT_LAST  = "/home/max/Projects/mesh-actors/models/onto18_v5_fast_last.pt"

# ── Config ─────────────────────────────────────────────────────────────────
# TARGET_EP is the total number of epochs you want to have run when done.
# If a checkpoint exists at epoch 40 and TARGET_EP=80, this run does 40 more.
TARGET_EP = 80
B         = 64
LR        = 0.002

# ── Model ──────────────────────────────────────────────────────────────────
enc = Encoder(d=128, in_dim=256).to(device)
dec = BILOUDecoder(128, NT).to(device)
total_params = (
    sum(p.numel() for p in enc.parameters())
    + sum(p.numel() for p in dec.parameters())
)
print(f"Params: {total_params:,}")

opt = torch.optim.AdamW(
    list(enc.parameters()) + list(dec.parameters()),
    lr=LR,
    weight_decay=0.01,
)
# T_max = TARGET_EP so the cosine schedule spans the full intended run.
# If resuming, last_epoch below rewinds the scheduler to the right position.
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TARGET_EP)

tt = [ex["ner_tags"] for ex in ctr]
vt = [ex["ner_tags"] for ex in cvl]

# ── Resume ─────────────────────────────────────────────────────────────────
start_epoch = 0
best        = 0.0

import os
if os.path.exists(CKPT_LAST):
    print(f"\nResuming from {CKPT_LAST} ...")
    ckpt = torch.load(CKPT_LAST, map_location=device)
    enc.load_state_dict(ckpt["enc"])
    dec.load_state_dict(ckpt["dec"])
    opt.load_state_dict(ckpt["opt"])
    sch.load_state_dict(ckpt["sch"])
    start_epoch = ckpt["epoch"]          # last completed epoch (0-indexed done)
    best        = ckpt["best_f1"]
    print(f"  Resumed at epoch {start_epoch}, best F1 so far: {best:.4f}")
    print(f"  Current LR: {sch.get_last_lr()}")
else:
    print(f"\nNo checkpoint found at {CKPT_LAST}, starting fresh.")

if start_epoch >= TARGET_EP:
    print(f"Already completed {start_epoch} epochs (target={TARGET_EP}). Nothing to do.")
else:
    remaining = TARGET_EP - start_epoch
    print(f"\nTraining v5-fast (CharCNN precomputed + CRF, B={B})")
    print(f"  Epochs {start_epoch + 1} → {TARGET_EP}  ({remaining} remaining)")

    for epoch in range(start_epoch, TARGET_EP):
        enc.train()
        dec.train()
        total_loss, n_batches = 0.0, 0
        ix = np.random.permutation(len(ctr))

        for i in tqdm(range(0, len(ix), B), desc=f"E{epoch + 1}/{TARGET_EP}"):
            batch_idx = ix[i: i + B]
            batch_emb = [train_emb[idx] for idx in batch_idx]

            x = nn.utils.rnn.pad_sequence(batch_emb, batch_first=True).to(device)
            h = enc(x)
            logits = dec(h)

            bs_i = len(batch_idx)
            ml   = x.shape[1]

            y    = torch.zeros((bs_i, ml), dtype=torch.long,  device=device)
            mask = torch.zeros((bs_i, ml), dtype=torch.bool,  device=device)
            for j, idx in enumerate(batch_idx):
                n = len(tt[idx])
                y[j, :n]    = torch.tensor(tt[idx], dtype=torch.long, device=device)
                mask[j, :n] = True

            opt.zero_grad()
            loss = -dec.crf(logits, y, mask=mask, reduction="mean")
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(dec.parameters()), 1.0
            )
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        sch.step()

        # ── Evaluation ─────────────────────────────────────────────────────
        enc.eval()
        dec.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(cvl), 64):
                end       = min(i + 64, len(cvl))
                batch_emb = [val_emb[j] for j in range(i, end)]
                x         = nn.utils.rnn.pad_sequence(batch_emb, batch_first=True).to(device)
                h         = enc(x)
                logits    = dec(h)

                bs_v = end - i
                ml_v = x.shape[1]
                mask = torch.zeros((bs_v, ml_v), dtype=torch.bool, device=device)
                for j, idx in enumerate(range(i, end)):
                    mask[j, :len(val_tokens[idx])] = True

                p = dec.crf.decode(logits, mask=mask)
                preds.extend(p)

        sp, sr, sf = span_f1(vt, preds)
        current_lr = sch.get_last_lr()[0]
        print(
            f"E{epoch + 1}: loss={total_loss / n_batches:.4f} "
            f"span_P={sp:.4f} span_R={sr:.4f} span_F1={sf:.4f}  lr={current_lr:.6f}"
        )

        # ── Save best ──────────────────────────────────────────────────────
        if sf > best:
            best = sf
            torch.save(
                {"enc": enc.state_dict(), "dec": dec.state_dict()},
                CKPT_BEST,
            )
            print(f"  ↑ best ({sf:.4f}) → {CKPT_BEST}")

        # ── Save last (full training state for resume) ─────────────────────
        # Saved every epoch so a crash loses at most one epoch of work.
        torch.save(
            {
                "epoch":   epoch + 1,      # number of completed epochs
                "enc":     enc.state_dict(),
                "dec":     dec.state_dict(),
                "opt":     opt.state_dict(),
                "sch":     sch.state_dict(),
                "best_f1": best,
            },
            CKPT_LAST,
        )

    print(f"\nBest span F1: {best:.4f}")