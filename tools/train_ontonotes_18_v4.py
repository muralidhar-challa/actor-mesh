"""v4: Token start/end NER. Simpler than BILOU/CRF, faster than span-sampling."""

import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ═══════════ Encoder (same as v2/v3) ═══════════


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


class HashEmbed(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.V = V
        self.e = nn.Embedding(V, d)

    def forward(self, cs):
        S = cs.shape[0]
        o = torch.zeros(S, self.e.weight.shape[1], device=cs.device)
        for k in range(4):
            o += self.e((cs[:, k] % self.V).long())
        return o


class Encoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.he = nn.ModuleList(
            [
                HashEmbed(5000, d),
                HashEmbed(1000, d),
                HashEmbed(2500, d),
                HashEmbed(2500, d),
            ]
        )
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

    def forward(self, x):
        B, S, _ = x.shape
        es = []
        for c, he in enumerate(self.he):
            cs = x[:, :, c * 4 : (c + 1) * 4].reshape(B * S, 4)
            es.append(he(cs).view(B, S, -1))
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


# ═══════════ Decoder: token start/end ═══════════


class StartEndDecoder(nn.Module):
    def __init__(self, d=128, n_types=18):
        super().__init__()
        self.start = nn.Linear(d, n_types + 1)
        self.end = nn.Linear(d, n_types + 1)

    def forward(self, h):
        return self.start(h), self.end(h)


# ═══════════ Loss + Labels ═══════════


def bilou_to_start_end(tags):
    S = len(tags)
    sl = [0] * S
    el = [0] * S
    i = 0
    while i < S:
        t = tags[i]
        if t == 0:
            i += 1
        else:
            pos = (t - 1) % 4
            et = (t - 1) // 4
            if pos == 3:
                sl[i] = et + 1
                el[i] = et + 1
                i += 1
            elif pos == 0:
                sl[i] = et + 1
                j = i + 1
                while j < S and tags[j] != 0 and (tags[j] - 1) % 4 in (1, 2):
                    if (tags[j] - 1) % 4 == 2:
                        el[j] = et + 1
                        break
                    j += 1
                i = j + 1
            else:
                i += 1
    return sl, el


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


def span_f1(gold_tags, pred_spans):
    tg, tp, tc = 0, 0, 0
    for g, p in zip(gold_tags, pred_spans):
        gs = extract_spans(g)
        ps = set((s, e, t) for s, e, t in p)
        tg += len(gs)
        tp += len(ps)
        tc += len(gs & ps)
    p = tc / tp if tp > 0 else 0
    r = tc / tg if tg > 0 else 0
    return p, r, 2 * p * r / (p + r) if p + r > 0 else 0


def logits_to_spans(sl, el):
    """Fast O(S·C): left-to-right scan with open-start tracking."""
    S = sl.shape[0]
    sa = sl.argmax(-1).numpy()
    ea = el.argmax(-1).numpy()
    spans = []
    for k in range(1, sl.shape[-1]):
        open_start = None
        for i in range(S):
            if sa[i] == k and open_start is None:
                open_start = i
            if ea[i] == k and open_start is not None:
                spans.append((open_start, i, k - 1))
                open_start = None
    return spans


# ═══════════ Features ═══════════


def mh3(k, s=0):
    h = s
    d = k.encode() if isinstance(k, str) else k
    for i in range(0, len(d) - len(d) % 4, 4):
        kk = d[i] | (d[i + 1] << 8) | (d[i + 2] << 16) | (d[i + 3] << 24)
        kk = (kk * 0xCC9E2D51) & 0xFFFFFFFF
        kk = ((kk << 15) | (kk >> 17)) & 0xFFFFFFFF
        kk = (kk * 0x1B873593) & 0xFFFFFFFF
        h ^= kk
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = ((h * 5) + 0xE6546B64) & 0xFFFFFFFF
    tl = len(d) % 4
    if tl:
        kk = 0
        for i in range(tl):
            kk |= d[len(d) - tl + i] << (8 * i)
        kk = (kk * 0xCC9E2D51) & 0xFFFFFFFF
        kk = ((kk << 15) | (kk >> 17)) & 0xFFFFFFFF
        kk = (kk * 0x1B873593) & 0xFFFFFFFF
        h ^= kk
    h ^= len(d)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def shp(t):
    o = []
    last = "\0"
    seq = 0
    for c in t:
        sc = "X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c
        seq = seq + 1 if sc == last else 0
        last = sc
        if seq < 4:
            o.append(sc)
    return "".join(o)


def keys(tokens):
    S = len(tokens)
    k = np.zeros((S, 16), dtype=np.uint64)
    for i, tok in enumerate(tokens):
        n = tok.lower()
        for j in range(4):
            k[i, 0 * 4 + j] = mh3(n, 8 * 4 + j)
        p = tok[0] if tok else " "
        for j in range(4):
            k[i, 1 * 4 + j] = mh3(p, 9 * 4 + j)
        sf = tok[-3:] if len(tok) >= 3 else tok
        for j in range(4):
            k[i, 2 * 4 + j] = mh3(sf, 10 * 4 + j)
        sh = shp(tok)
        for j in range(4):
            k[i, 3 * 4 + j] = mh3(sh, 11 * 4 + j)
    return k


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
print(f"{NT} types")

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


def i2b(tags):
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
    ex["ner_tags"] = i2b(ex["ner_tags"])
for ex in cvl:
    ex["ner_tags"] = i2b(ex["ner_tags"])
print(f"Train: {len(ctr)} ex, {sum(len(ex['tokens']) for ex in ctr):,} tokens")
print(f"Val:   {len(cvl)} ex, {sum(len(ex['tokens']) for ex in cvl):,} tokens")

# ═══════════ Training ═══════════

device = torch.device("cuda")
enc = Encoder(128).to(device)
dec = StartEndDecoder(128, NT).to(device)
print(
    f"Params: {sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters()):,}"
)

opt = torch.optim.AdamW(
    list(enc.parameters()) + list(dec.parameters()), lr=0.002, weight_decay=0.01
)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
ce = nn.CrossEntropyLoss(ignore_index=-1)

print("Precomputing features...")
tk = [keys(ex["tokens"]) for ex in tqdm(ctr, desc="train")]
vk = [keys(ex["tokens"]) for ex in tqdm(cvl, desc="val")]
tt = [ex["ner_tags"] for ex in ctr]
vb = [ex["ner_tags"] for ex in cvl]

# Pre-build start/end labels
tsl = [bilou_to_start_end(t)[0] for t in tt]
tel = [bilou_to_start_end(t)[1] for t in tt]


def batch(kl, el, ix):
    bs = len(ix)
    ml = max(len(el[i]["tokens"]) for i in ix)
    x = torch.zeros(bs, ml, 16, dtype=torch.long)
    for j, i in enumerate(ix):
        n = len(el[i]["tokens"])
        x[j, :n] = torch.from_numpy(kl[i])
    return x.to(device, non_blocking=True)


B = 64
EP = 80
best = 0
print(f"\nTraining v4 (token start/end, B={B}, {EP}ep)...")

for epoch in range(EP):
    enc.train()
    dec.train()
    tl, tn = 0, 0
    ix = np.random.permutation(len(ctr))
    for i in tqdm(range(0, len(ix), B), desc=f"E{epoch + 1}"):
        x = batch(tk, ctr, ix[i : i + B])
        h = enc(x)
        sl, el = dec(h)
        # Build label tensors
        bs_i = len(ix[i : i + B])
        s_lbl = torch.full((bs_i, x.shape[1]), -1, dtype=torch.long)
        e_lbl = torch.full((bs_i, x.shape[1]), -1, dtype=torch.long)
        for j in range(bs_i):
            n = len(ctr[ix[i + j]]["tokens"])
            s_lbl[j, :n] = torch.tensor(tsl[ix[i + j]])
            e_lbl[j, :n] = torch.tensor(tel[ix[i + j]])
        s_lbl = s_lbl.to(device)
        e_lbl = e_lbl.to(device)
        loss = ce(sl.reshape(-1, NT + 1), s_lbl.reshape(-1)) + ce(
            el.reshape(-1, NT + 1), e_lbl.reshape(-1)
        )
        opt.zero_grad()
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
            x = batch(vk, cvl, list(range(i, min(i + 64, len(cvl)))))
            h = enc(x)
            sl, el = dec(h)
            for j in range(x.shape[0]):
                preds.append(
                    logits_to_spans(
                        sl[j, : len(cvl[i + j]["tokens"])].cpu(),
                        el[j, : len(cvl[i + j]["tokens"])].cpu(),
                    )
                )
    sp, sr, sf = span_f1(vb, preds)
    print(
        f"E{epoch + 1}: loss={tl / tn:.4f} span_P={sp:.4f} span_R={sr:.4f} span_F1={sf:.4f}"
    )
    if sf > best:
        best = sf
        torch.save(
            {"enc": enc.state_dict(), "dec": dec.state_dict()},
            "/home/max/Projects/mesh-actors/models/onto18_v4_best.pt",
        )
        print(f"  ↑ best ({sf:.4f})")
print(f"\nBest: {best:.4f}")
