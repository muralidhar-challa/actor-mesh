#!/usr/bin/env python3
"""
test_ner_decode.py — end-to-end NER using only ONNX model + greedy BILUO decoder.
No spaCy NER head at runtime; only spaCy's tokenizer + feature extraction.

Validates that the full C-replicable pipeline produces the same entities as
native spaCy.
"""
import numpy as np
import base64
import onnxruntime as ort
import spacy
from thinc.backends import NumpyOps

MODEL_PATH = "models/ner.onnx"

nlp  = spacy.load("en_core_web_sm")
ops  = NumpyOps()

# feature extraction metadata (same as C ner_build_feature_matrix)
ner       = nlp.get_pipe("ner")
embed     = ner.model._layers[0]._layers[0]._layers[0]
concat_he = embed._layers[2]._layers[0]
columns   = embed._layers[0].attrs["columns"]
seeds     = [concat_he._layers[c]._layers[1].attrs["seed"] for c in range(4)]

sess = ort.InferenceSession(MODEL_PATH)
meta = sess.get_modelmeta().custom_metadata_map

pa_b     = np.frombuffer(base64.b64decode(meta["pa_b"]),     dtype=np.float32).reshape(64, 2)
linear_W = np.frombuffer(base64.b64decode(meta["linear_W"]), dtype=np.float32).reshape(74, 64)
linear_b = np.frombuffer(base64.b64decode(meta["linear_b"]), dtype=np.float32)
move_names = meta["moves"].split(",")

# ── BILUO helpers ──────────────────────────────────────────────────────────────

def _action(cls):
    if cls < 18:  return 'B'
    if cls < 36:  return 'I'
    if cls < 54:  return 'L'
    if cls < 72:  return 'U'
    if cls == 73: return 'O'
    return 'MISSING'

def _label_idx(cls): return cls % 18

_VALID_OUTSIDE = np.zeros(74, dtype=bool)
_VALID_OUTSIDE[:18]   = True   # B-*
_VALID_OUTSIDE[54:72] = True   # U-*
_VALID_OUTSIDE[73]    = True   # O

def _valid_inside(lbl_idx):
    v = np.zeros(74, dtype=bool)
    v[18 + lbl_idx] = True   # I-same
    v[36 + lbl_idx] = True   # L-same
    return v

def greedy_biluo(table, n_tok):
    """
    table: (n_tok+1, 3, 64, 2) float32
    Returns list of (start, end_exclusive, label_str) spans.

    Feature slots from StateC::set_context_tokens (n==3):
      ids[0] = B0          current token
      ids[1] = E(0)        entity start if open, else -1 → row 0 (pad)
      ids[2] = B0-1        previous token if inside entity, else -1 → row 0 (pad)
    Table rows: row 0 = pad, row t+1 = token t.
    """
    state_outside = True
    ent_start = -1
    lbl_idx   = 0
    spans     = []

    for t in range(n_tok):
        b0 = t + 1                                      # row for current token
        s0 = 0 if state_outside else (ent_start + 1)   # entity start row, or pad
        b1 = 0 if state_outside else t                  # prev-token row (t-1+1=t), or pad

        feat = table[b0, 0] + table[s0, 1] + table[b1, 2]   # (64, 2)
        feat = feat + pa_b
        feat_max = feat.max(axis=1)                           # maxout → (64,)
        scores = linear_W @ feat_max + linear_b              # (74,)

        valid = _VALID_OUTSIDE if state_outside else _valid_inside(lbl_idx)
        scores[~valid] = -1e9
        cls = int(scores.argmax())
        act = _action(cls)

        if act == 'B':
            ent_start = t; lbl_idx = _label_idx(cls); state_outside = False
        elif act == 'I':
            pass
        elif act == 'L':
            spans.append((ent_start, t + 1, move_names[cls][2:]))
            state_outside = True
        elif act == 'U':
            spans.append((t, t + 1, move_names[cls][2:]))
            state_outside = True
        else:   # O
            state_outside = True

    return spans


def onnx_entities(text):
    doc = nlp.make_doc(text)
    n   = len(doc)

    raw   = doc.to_array(columns).astype(np.uint64)
    feats = np.zeros((n, 16), dtype=np.uint64)
    for i in range(4):
        feats[:, i*4:i*4+4] = ops.hash(np.ascontiguousarray(raw[:, i]), seeds[i])

    table  = sess.run(None, {"token_hashes": feats})[0]   # (n+1, 3, 64, 2)
    spans  = greedy_biluo(table, n)
    return [(doc[s:e].text, lbl) for s, e, lbl in spans]


def native_entities(text):
    return [(e.text, e.label_) for e in nlp(text).ents]


texts = [
    # original 6
    "Apple CEO Tim Cook met with Sundar Pichai in New York last Tuesday.",
    "Dr. Emily Chen joined Acme Corp on January 15th.",
    "The Federal Reserve raised rates by 25 basis points.",
    "Elon Musk founded SpaceX in Hawthorne, California in 2002.",
    "Barack Obama was born in Honolulu, Hawaii.",
    "Microsoft acquired LinkedIn for $26 billion in 2016.",
    # multi-token names / adjacency
    "Jeff Bezos stepped down as Amazon CEO in July 2021.",
    "Satya Nadella has led Microsoft since February 2014.",
    "Warren Buffett and Charlie Munger run Berkshire Hathaway.",
    "The United Nations headquarters is located in New York City.",
    "Angela Merkel served as Chancellor of Germany for 16 years.",
    "Cristiano Ronaldo plays for Al Nassr in Saudi Arabia.",
    # single-token (U-* path)
    "Tesla shares rose 5% on Friday.",
    "NASA launched the Artemis mission from Florida.",
    "Google was founded in Menlo Park.",
    # long, entity-dense
    "Kamala Harris met with Emmanuel Macron in Paris to discuss NATO and Ukraine.",
    "Amazon, Google, and Meta reported earnings on Thursday.",
    "The European Central Bank, led by Christine Lagarde, hiked rates in Frankfurt.",
    # ambiguous / tricky
    "Apple and Microsoft are both headquartered in the United States.",
    "The New York Times reported that President Biden signed the bill.",
    "Dr. Fauci advised the White House on the COVID-19 response.",
    # dates and money
    "SpaceX raised $2 billion in January 2023.",
    "The World Cup final was held in Lusail, Qatar on December 18, 2022.",
    # no entities
    "The quick brown fox jumps over the lazy dog.",
    "Interest rates remain unchanged after the committee vote.",
]

print(f"\n{'Text':<58}  match")
print("─" * 100)
all_match = True
for text in texts:
    native = native_entities(text)
    onnx   = onnx_entities(text)
    match  = native == onnx
    all_match = all_match and match
    flag = "✓" if match else "≠"
    print(f"{text[:57]:<58}  {flag}")
    print(f"  native: {native}")
    if not match:
        print(f"  onnx:   {onnx}")

print("\n" + ("ALL MATCH" if all_match else "SOME DIFFER"))
