"""v5: Char CNN encoder replaces hash embeddings. Same decoder as v2 (BILOU+CRF)."""

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

# ═══════════ Char CNN Embedding ═══════════


class CharCNN(nn.Module):
    """Character-level CNN → fixed-size token embedding.
    Processes: lowercased token, prefix, suffix, or shape pattern."""

    def __init__(self, char_vocab=256, char_dim=32, out_dim=128, max_len=20):
        super().__init__()
        self.max_len = max_len
        self.char_embed = nn.Embedding(char_vocab, char_dim, padding_idx=0)
        self.convs = nn.Sequential(
            nn.Conv1d(char_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, out_dim, 3, padding=1),
        )

    def forward(self, char_ids):
        """char_ids: (B*S, max_len) int tensor, 0=padded"""
        x = self.char_embed(char_ids)  # (N, max_len, char_dim)
        x = x.transpose(1, 2)  # (N, char_dim, max_len)
        x = self.convs(x)  # (N, out_dim, max_len)
        x, _ = x.max(dim=-1)  # (N, out_dim) — global max pool
        return x


# ═══════════ Encoder (same CNN backbone, char-CNN features) ═══════════


class SEBlock(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(d, d // r), nn.Hardswish(), nn.Linear(d // r, d), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x.mean(1)).unsqueeze(1)


class SepConv(nn.Module):
    def __init__(self, d, dl=1):
        super().__init__()
        self.dw = nn.Conv1d(d, d, 3, padding=dl, dilation=dl, groups=d)
        self.pw = nn.Conv1d(d, d, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class SepResBlock(nn.Module):
    def __init__(self, d, dl=1):
        super().__init__()
        self.c = SepConv(d, dl)
        self.n = nn.LayerNorm(d)
        self.s = SEBlock(d)
        self.dp = nn.Dropout(0.1)

    def forward(self, x):
        r = x
        x = self.c(x)
        x = x.transpose(1, 2)
        x = self.n(x)
        x = self.s(x)
        x = x.transpose(1, 2)
        x = nn.functional.hardswish(x)
        return self.dp(x) + r


class Encoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.char_cnns = nn.ModuleList([CharCNN(out_dim=d) for _ in range(4)])
        self.proj = nn.Linear(d * 4, d)
        self.norm = nn.LayerNorm(d)
        self.lc = SepConv(d)
        self.ln = nn.LayerNorm(d)
        self.ls = SEBlock(d)
        self.gr = nn.ModuleList(
            [SepResBlock(d, 1), SepResBlock(d, 3), SepResBlock(d, 5)]
        )
        self.wr = nn.ModuleList([SepResBlock(d, 9), SepResBlock(d, 15)])
        self.gate = nn.Linear(d * 3, d)
        self.drop = nn.Dropout(0.1)

    def forward(self, char_batch):
        """char_batch: (B, S, 4, max_len) — 4 views per token"""
        B, S, _, L = char_batch.shape
        es = []
        for col in range(4):
            ids = char_batch[:, :, col, :].reshape(B * S, L)
            es.append(self.char_cnns[col](ids).view(B, S, -1))
        e = self.proj(torch.cat(es, -1))
        e = self.norm(e)
        et = e.transpose(1, 2)
        l = self.lc(et).transpose(1, 2)
        l = self.ln(l)
        l = self.ls(l).transpose(1, 2)
        l = nn.functional.hardswish(l)
        g = et
        for r in self.gr:
            g = r(g)
        w = et
        for r in self.wr:
            w = r(w)
        lt = l.transpose(1, 2)
        gt = g.transpose(1, 2)
        wt = w.transpose(1, 2)
        gw = torch.softmax(self.gate(self.drop(torch.cat([lt, gt, wt], -1))), -1)
        return gw[:, :, 0:1] * lt + gw[:, :, 1:2] * gt + gw[:, :, 2:3] * wt


# ═══════════ Decoder (BILOU + CRF) ═══════════


class BILOUDecoder(nn.Module):
    def __init__(self, d=128, n_types=18):
        super().__init__()
        self.ner = nn.Linear(d, n_types * 4 + 1)
        self.crf = CRF(n_types * 4 + 1, batch_first=True)

    def forward(self, h):
        return self.ner(h)


# ═══════════ Features: text → char IDs ═══════════


def token_to_char_ids(token, max_len=20):
    """Convert token text to char IDs (1..255, 0=pad)."""
    ids = [0] * max_len
    raw = token.encode("utf-8", errors="replace")[:max_len]
    for i, c in enumerate(raw):
        ids[i] = c if c > 0 else 1
    return ids


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


def tokens_to_char_batch(tokens_list, max_len=20):
    """Convert list of token lists → (B, S_max, 4, max_len) char id tensor.
    4 views: norm, prefix, suffix, shape."""
    max_S = max(len(t) for t in tokens_list)
    B = len(tokens_list)
    batch = np.zeros((B, max_S, 4, max_len), dtype=np.int64)
    for b, tokens in enumerate(tokens_list):
        for s, tok in enumerate(tokens):
            batch[b, s, 0] = token_to_char_ids(tok.lower(), max_len)
            batch[b, s, 1] = token_to_char_ids(tok[0] if tok else " ", max_len)
            sfx = tok[-3:] if len(tok) >= 3 else tok
            batch[b, s, 2] = token_to_char_ids(sfx, max_len)
            batch[b, s, 3] = token_to_char_ids(ner_shape(tok), max_len)
    return torch.from_numpy(batch)


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
            if pos == 3:
                spans.add((i, i, et))
                i += 1
            elif pos == 0:
                s = i
                i += 1
                while i < n and tags[i] != 0 and (tags[i] - 1) % 4 in (1, 2):
                    if (tags[i] - 1) % 4 == 2:
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


def load_conll(p):
    s, l = [], []
    t, g = [], []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if t:
                    s.append(t)
                    l.append(g)
                    t, g = [], []
                continue
            ps = line.split()
            if len(ps) >= 2:
                t.append(ps[0])
                g.append(ps[-1])
    if t:
        s.append(t)
        l.append(g)
    return s, l


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


print("Loading data...")
SPACY = [
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
T2I = {t: i + 1 for i, t in enumerate(SPACY)}
NT = len(SPACY)
NC = NT * 4 + 1
print(f"{NT} types, {NC} BILOU classes")

fewnerd = load_dataset("DFKI-SLT/few-nerd", "supervised")
FM = {
    0: 0,
    1: T2I["WORK_OF_ART"],
    2: T2I["FAC"],
    3: T2I["EVENT"],
    4: T2I["LOC"],
    5: T2I["ORG"],
    6: 0,
    7: T2I["PERSON"],
    8: T2I["PRODUCT"],
}
wnut = pickle.load(open("/tmp/wnut.pkl", "rb"))
WM = {
    0: 0,
    1: T2I["ORG"],
    2: T2I["ORG"],
    3: T2I["WORK_OF_ART"],
    4: T2I["WORK_OF_ART"],
    5: T2I["ORG"],
    6: T2I["ORG"],
    7: T2I["LOC"],
    8: T2I["LOC"],
    9: T2I["PERSON"],
    10: T2I["PERSON"],
    11: T2I["PRODUCT"],
    12: T2I["PRODUCT"],
}
cs, ct = load_conll("/tmp/train.conll")
ds2, dt2 = load_conll("/tmp/dev.conll")
CM = {
    "O": 0,
    "B-PER": T2I["PERSON"],
    "I-PER": T2I["PERSON"],
    "B-ORG": T2I["ORG"],
    "I-ORG": T2I["ORG"],
    "B-LOC": T2I["LOC"],
    "I-LOC": T2I["LOC"],
    "B-MISC": 0,
    "I-MISC": 0,
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
            l = i - s
            e = t - 1
            if l == 1:
                b.append(e * 4 + 3 + 1)
            else:
                b.append(e * 4 + 0 + 1)
                for _ in range(1, l - 1):
                    b.append(e * 4 + 1 + 1)
                b.append(e * 4 + 2 + 1)
    return b


for ex in ctr:
    ex["ner_tags"] = io_to_bilou(ex["ner_tags"])
for ex in cvl:
    ex["ner_tags"] = io_to_bilou(ex["ner_tags"])
print(f"Train: {len(ctr)} ex, {sum(len(ex['tokens']) for ex in ctr):,} tokens")
print(f"Val:   {len(cvl)} ex, {sum(len(ex['tokens']) for ex in cvl):,} tokens")

# ═══════════ Training ═══════════

device = torch.device("cuda")
enc = Encoder(128).to(device)
dec = BILOUDecoder(128, NT).to(device)
print(
    f"Params: {sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters()):,}"
)

opt = torch.optim.AdamW(
    list(enc.parameters()) + list(dec.parameters()), lr=0.002, weight_decay=0.01
)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
ce = nn.CrossEntropyLoss(ignore_index=-1)

print("Precomputing char features...")
train_tokens = [ex["tokens"] for ex in ctr]
val_tokens = [ex["tokens"] for ex in cvl]
tt = [ex["ner_tags"] for ex in ctr]
vb = [ex["ner_tags"] for ex in cvl]

B = 32
EP = 40
best = 0
print(f"\nTraining v5 (CharCNN+CRF, B={B}, {EP}ep)...")

for epoch in range(EP):
    enc.train()
    dec.train()
    tl, tn = 0, 0
    ix = np.random.permutation(len(ctr))
    for i in tqdm(range(0, len(ix), B), desc=f"E{epoch + 1}"):
        batch_tokens = [train_tokens[idx] for idx in ix[i : i + B]]
        x = tokens_to_char_batch(batch_tokens).to(device)
        h = enc(x)
        logits = dec(h)
        bs_i = len(ix[i : i + B])
        ml = x.shape[1]
        y = torch.full((bs_i, ml), -1, dtype=torch.long)
        for j, idx in enumerate(ix[i : i + B]):
            n = len(tt[idx])
            y[j, :n] = torch.tensor(tt[idx])
        y = y.to(device)
        mask = y != -1
        opt.zero_grad()
        loss = -dec.crf(logits, y, mask=mask, reduction="mean")
        loss.backward()
        opt.step()
        tl += loss.item()
        tn += 1
    sch.step()

    # Eval
    enc.eval()
    dec.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(cvl), 64):
            batch_tokens = [val_tokens[idx] for idx in range(i, min(i + 64, len(cvl)))]
            x = tokens_to_char_batch(batch_tokens).to(device)
            h = enc(x)
            logits = dec(h)
            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=device)
            for j, idx in enumerate(range(i, min(i + 64, len(cvl)))):
                mask[j, len(val_tokens[idx]) :] = False
            p = dec.crf.decode(logits, mask=mask)
            for j in range(len(p)):
                preds.append(p[j])

    sp, sr, sf = span_f1(vb, preds)
    print(
        f"E{epoch + 1}: loss={tl / tn:.4f} span_P={sp:.4f} span_R={sr:.4f} span_F1={sf:.4f}"
    )
    if sf > best:
        best = sf
        torch.save(
            {"enc": enc.state_dict(), "dec": dec.state_dict()},
            "/home/max/Projects/mesh-actors/models/onto18_v5_best.pt",
        )
        print(f"  ↑ best ({sf:.4f})")
print(f"\nBest: {best:.4f}")
