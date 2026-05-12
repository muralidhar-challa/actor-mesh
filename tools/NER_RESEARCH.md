# PII NER Model — Architecture Evolution & Findings

## Overview

Building a compact, deployable Named Entity Recognition model for PII masking.
Target: pure C inference via ONNX, <10 MB model, no Python runtime dependency.

## Final Scoreboard

| Model | Encoder | Decoder | Params | Span F1 | Notes |
|-------|---------|---------|--------|---------|-------|
| v1 (8-type) | Hash embed + BILOU | Argmax | ~1M | 64.82% | Baseline |
| v2 (18-type) | Hash embed + BILOU | Argmax + focal loss | 1.7M | 71.31% | Rich features (NORM/PREFIX/SUFFIX/SHAPE) |
| **v3 (18-type)** | **Hash embed + BILOU** | **CRF** | **1.7M** | **72.78%** | **Best: hash + CRF** |
| v4 (18-type) | Hash embed | Token start/end | 1.7M | 69.18% | Greedy pairing loses BILOU structure |
| v5-old (18-type) | CharCNN (full conv) | CRF | 284K | 69.79% | 6× fewer params, 3% lower F1 |
| v5-lean (18-type) | CharCNN (sep conv) | CRF | 203K | 67.46% | Depthwise sep too aggressive for char level |
| spaCy en_core_web_sm | CNN + transition parser | Parser | ~5M | ~86% | Target to beat |

## Data

| Dataset | Tokens | % | Types |
|---------|--------|---|-------|
| Few-NERD | 3.2M | 70% | 8 |
| OntoNotes 5 | 1.1M | 24% | 18 |
| CoNLL-2003 | 204K | 4% | 4 |
| WNUT-17 | 63K | 1% | 6 |
| **Total** | **4.6M** | | **18 (spaCy-compatible)** |

### 18 Entity Types

```
PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART,
LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
```

OntoNotes adds DATE, TIME, MONEY, PERCENT etc. that Few-NERD lacks.
Rare types (LAW 1K, LANGUAGE 317 tokens) remain challenging.

## Architecture Evolution

### Encoder: Hash Embed (v2/v3 — winner)

```
Token → 4 views: NORM(lowercase), PREFIX(1st char), SUFFIX(last 3), SHAPE(XxXd pattern)
     → murmurhash3 × 4 variants per view → 16 int64 hash keys
     → 4× HashEmbed(vocab, 128) → concat 512 → Linear(512→128) → LayerNorm
     → 3-path CNN backbone (local + global d=1,3,5 + wide d=9,15)
     → 3-way softmax gate fusion → 128-dim
```

Hash embeddings are collision-prone but:
- Extremely fast (16 ints per token, no compute)
- Work well with sufficient vocab size (5000/1000/2500/2500)
- No out-of-vocabulary issues
- Total: 1.7M params, 36 sec/epoch (argmax), 2.5 min/epoch (CRF)

### Encoder: CharCNN (v5 — promising but behind)

```
Token → 4 char views (max 20 chars each) 
     → utf-8 byte encoding → char IDs 1..255
     → CharCNN per column:
         Embed(256→64) → Conv(64→96,k=3) → Conv(96→128,k=3) 
         → max_pool → Linear(128→64) → LayerNorm
     → Concat 4×64 = 256 → Linear(256→128) → LayerNorm
     → [same 3-path CNN backbone]
```

CharCNN advantages:
- No collisions — learns morphology from characters
- Smaller (can be under 300K params)
- Captures prefixes/suffixes/character patterns

CharCNN disadvantages:
- 32× more data per token (256 float32 vs 16 int64)
- Training bandwidth bottleneck
- With our data volume (4.6M tokens), can't match hash embed accuracy
- Depthwise separable convs too aggressive for 20-char sequences

**Key finding**: With limited training data, hash collisions cost less than 
CharCNN's parameter inefficiency. More data would favor CharCNN.

### Decoder: CRF (v3 — winner)

```
Linear(128→73) → CRF(73×73=5,329 transitions) → Viterbi decode
```

CRF enforces valid BILOU transitions (B-PER → I-PER allowed, B-PER → O penalized).
Adds +1.5 F1 over argmax, +3 over token start/end.

Training cost: 4× slower than argmax (forward-backward O(S·C²)).
Inference cost: Viterbi is O(S·C²) ≈ 117K ops for 22 tokens — negligible (~5 µs).

### Decoders tested and abandoned

| Decoder | F1 | Why abandoned |
|---------|-----|---------------|
| Token start/end + greedy pair | 69.18% | Loses BILOU structure, mismatched spans |
| Span-sampled classifier | — | O(S²) eval too slow, train/eval mismatch |
| Biaffine | — | O(S²) eval, over-engineered for flat NER |
| W2NER | — | S×S grid, designed for nested NER |

## Key Design Decisions

### No RoPE (removed in v3+)
CNNs are inherently position-aware through sliding windows. RoPE is redundant
for CNNs and slightly harmful (breaks translation equivariance). Removed in v3+.

### Rich features over simple hash
NORM/PREFIX/SUFFIX/SHAPE gave +6.8 F1 over simple 4× lowercased hash.
The prefix captures titles (Dr., Mr.), suffix captures morphology (ing, tion),
shape captures orthographic patterns (Xxxx, Xx.).

### 3-path multi-scale over single-scale
Local (k=3) + Global (d=1,3,5) + Wide (d=9,15) with learned gate fusion.
Receptive field extends to ±24 tokens. Essential for long entity detection.

### Focal loss didn't help
Focal loss (γ=2) matched cross-entropy at ceiling. Class imbalance (79.5% O)
wasn't severe enough to benefit. Weighted CE also didn't improve peak F1.

### CRF worth the training cost
+1.5 F1 for 4× training time is a good tradeoff. Constrained Viterbi
(hardcoded BILOU rules) is unexplored — could give most CRF benefit for free.

### Gate optimization
`Linear(384→3)` (1K params) matches `Linear(384→128)` (49K params) for 
3-way fusion. The full 128-dim gate output is redundant — softmax collapses it.

### Depthwise separable convs
Excellent for token-level backbone (35% param reduction, same accuracy).
Too aggressive for char-level (20-char sequences need full convs).

## What We Learned About Constraints

- **Batch size**: 64 balances GPU utilization (84%) with memory (1.6 GB / 12 GB)
- **Precompute everything**: Hash keys and char embeddings precomputed once, 
  eliminates CPU bottleneck. `non_blocking=True` for CPU→GPU transfers.
- **CRF dominates runtime**: Encoder is 5% of training time, CRF forward-backward is 95%
- **Hash embed is bandwidth-efficient**: 16 int64 per token vs 256 float32 for CharCNN

## Julia + Zygote Exploration

Zygote's source-to-source AD could simplify CRF implementation:
- Write CRF forward as plain Julia for-loop
- Zygote differentiates through control flow automatically
- No C++ extension, no custom backward pass needed

Potentially enables:
- Semi-CRF (span-level Viterbi) 
- Differentiable constrained decoding
- Custom transition priors injected into loss

Status: Julia installed (1.11.0-rc3), package downloads failing (network issue).

## ONNX Export

Best model: v3 (hash+CRF), 1.7M params, ~7 MB ONNX.
Input: `(batch, seq_len, 16)` int64 hash keys.
Output: `(batch, seq_len, 73)` float32 BILOU logits.
Viterbi decode runs in C as a separate step (~50 lines of C, 73×73 transition table).

## C Pipeline Integration

- `ner_features.h`: tokenizer + feature extraction (spaCy-compatible)
- `ner_infer.h`: ONNX inference via onnxruntime
- `ner_rules.h`: regex-based PII detection (email, phone, SSN, CC, IP, URL)
- Tokenizer has known gaps vs spaCy: no infix splitting, UTF-8 issues, 
  abbreviation handling differs

Feature extraction MUST match training. Our C pipeline was built for 
spaCy's NORM/PREFIX/SUFFIX/SHAPE → spacy_hash → thinc_hash chain. 
Our model uses murmurhash3(tok.lower(), seed*4+k). These are incompatible.
C feature extraction needs rewrite to match training.

## Future Directions

1. **Constrained Viterbi on v2**: Hardcoded BILOU transitions, 0 training cost,
   estimated +2-3 F1. Test pending.

2. **More training data**: Synthesize PII examples, add more OntoNotes-like data.
   Current 4.6M tokens may be insufficient for rare types.

3. **CharCNN with more data**: If we 10× the training data, CharCNN should 
   surpass hash embed. Currently data-limited.

4. **Pretrained tiny embeddings**: Distilled GloVe or BPEmb subword embeddings.
   Could add 500K params and +5-8 F1. Breaks pure-C deployment if embeddings
   are large.

5. **Julia prototype**: Zygote for cleaner CRF, potentially faster training.
   Same accuracy ceiling, better developer experience.

6. **C feature extraction rewrite**: Match our murmurhash3-based features in C
   for correct inference.
