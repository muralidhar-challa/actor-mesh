#!/usr/bin/env python3
"""
Cross-check ner_features.h hash values against spaCy + thinc.

Usage: python3 tools/test_ner_features.py
"""
import numpy as np
import spacy
from thinc.backends import NumpyOps

nlp  = spacy.load("en_core_web_sm")
ops  = NumpyOps()

ner        = nlp.get_pipe("ner").model
big_chain  = ner._layers[0]
inner      = big_chain._layers[0]
embed      = inner._layers[0]
concat_he  = embed._layers[2]._layers[0]
columns    = embed._layers[0].attrs["columns"]

seeds = [concat_he._layers[col]._layers[1].attrs["seed"] for col in range(4)]
print(f"columns: {columns}")
print(f"seeds:   {seeds}")

texts = [
    "Apple CEO Tim Cook met with Sundar Pichai in New York last Tuesday.",
    "Dr. Emily Chen joined Acme Corp on 2024-01-15.",
    "hello world",
]

for text in texts:
    doc = nlp.make_doc(text)
    raw = doc.to_array(columns).astype(np.uint64)   # (seq, 4)
    n   = raw.shape[0]
    feats = np.zeros((n, 16), dtype=np.uint64)
    for col_idx in range(4):
        keys = ops.hash(np.ascontiguousarray(raw[:, col_idx]), seeds[col_idx])
        feats[:, col_idx*4 : col_idx*4+4] = keys

    print(f"\nText: {text}")
    toks = [t.text for t in doc]
    print(f"Tokens ({len(toks)}): {toks}")
    print("Feature matrix:")
    for i in range(n):
        print(f"  tok[{i:2d}] {toks[i]:<12s}  " +
              ",".join(str(v) for v in feats[i]))

# standalone hash checks
from murmurhash.mrmr import hash_bytes as mm64
h = mm64(b"hello", 0)
print(f"\nMurmurHash64A(\"hello\", 5, 0) = {h & 0xffffffffffffffff}")

# thinc's ops.hash on a known id
import ctypes
test_id = np.array([h & 0xffffffffffffffff], dtype=np.uint64)
k4 = ops.hash(test_id, 0)
print(f"thinc_hash4(h, seed=0) = {list(k4[0])}")
