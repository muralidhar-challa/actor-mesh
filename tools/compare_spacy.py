"""Head-to-head: our 18-type ONNX-ready model vs spaCy en_core_web_sm."""

import numpy as np
import spacy
import torch
import torch.nn as nn


# ── Architecture (must match training) ──
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
            s = x[:, :, col * 4 : (col + 1) * 4].reshape(B * S, 4)
            embeds.append(he(s).view(B, S, -1))
        e = self.proj(torch.cat(embeds, -1))
        e = self.norm_in(e)
        e = self.rope(e)
        et = e.transpose(1, 2)
        local = self.local_conv(et).transpose(1, 2)
        local = self.local_norm(local)
        local = self.local_se(local).transpose(1, 2)
        local = torch.nn.functional.hardswish(local)
        g = et
        for res in self.global_res:
            g = res(g)
        lt = local.transpose(1, 2)
        gt = g.transpose(1, 2)
        gw = torch.sigmoid(self.gate(self.drop(torch.cat([lt, gt], -1))))
        return self.ner(self.drop(gw * lt + (1 - gw) * gt))


# ── Hash ──
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


# ── 18-type labels ──
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


def decode_spans(tokens, preds):
    ents = []
    i = 0
    while i < len(preds):
        t = preds[i]
        if t == 0:
            i += 1
        else:
            pos = (t - 1) % 4
            etype = (t - 1) // 4
            if pos == 3:
                ents.append((tokens[i], SPACY_TYPES[etype]))
                i += 1
            elif pos == 0:
                start = i
                i += 1
                while i < len(preds) and preds[i] != 0 and (preds[i] - 1) % 4 in (1, 2):
                    if (preds[i] - 1) % 4 == 2:
                        ents.append(
                            (" ".join(tokens[start : i + 1]), SPACY_TYPES[etype])
                        )
                        i += 1
                        break
                    i += 1
            else:
                i += 1
    return ents


# ── Load models ──
print("Loading models...")
nlp = spacy.load("en_core_web_sm")

model = AdvancedNER(embed_dim=128, n_classes=73)
ckpt = torch.load(
    "/home/max/Projects/mesh-actors/models/onto18_best_backup.pt", map_location="cpu"
)
model.load_state_dict(ckpt)
model.eval()

# ── Test sentences ──
sentences = [
    "Apple CEO Tim Cook met with Sundar Pichai in New York last Tuesday.",
    "John Smith works at Microsoft in Seattle and his email is john@example.com.",
    "The Eiffel Tower in Paris was built by Gustave Eiffel in 1889.",
    "Google acquired YouTube for 1.65 billion dollars in 2006.",
    "Dr. Maria Garcia from Stanford University published a paper on COVID-19.",
    "The meeting on January 15th at 3 PM was attended by French and German delegates.",
    "IBM reported $5.2 billion in revenue, a 12% increase over last quarter.",
    "Hurricane Katrina devastated New Orleans in August 2005.",
    "The 1st amendment to the U.S. Constitution was ratified in 1791.",
    "Shakespeare's Hamlet was performed at the Globe Theatre in London.",
]

for text in sentences:
    doc = nlp(text)
    tokens = [t.text for t in doc]

    # spaCy
    spacy_ents = [(e.text, e.label_) for e in doc.ents]

    # Ours
    keys = compute_hash_keys(tokens)
    x = torch.from_numpy(keys).unsqueeze(0).long()
    with torch.no_grad():
        preds = model(x).argmax(-1).squeeze(0).tolist()
    our_ents = decode_spans(tokens, preds)

    # Find matches (any overlap)
    spacy_texts = {e[0].lower() for e in spacy_ents}

    print(f"\n{'=' * 80}")
    print(f"TEXT: {text}")
    print(f"  spaCy: {spacy_ents}")
    print(f"  Ours:  {our_ents}")

    matched = [e for e in our_ents if e[0].lower() in spacy_texts]
    missed = [e for e in our_ents if e[0].lower() not in spacy_texts]
    spacy_missed = [
        e for e in spacy_ents if e[0].lower() not in {x[0].lower() for x in our_ents}
    ]
    if matched:
        print(f"    ✓ Matched: {matched}")
    if missed:
        print(f"    + Extra (ours only): {missed}")
    if spacy_missed:
        print(f"    - Missed (spaCy only): {spacy_missed}")
