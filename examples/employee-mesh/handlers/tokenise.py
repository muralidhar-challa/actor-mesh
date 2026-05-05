#!/usr/bin/env python3
"""
handlers/tokenise.py — PII tokeniser

Masks named entities in all string fields of any message heading toward the LLM.
Emits the correct output topic via the "\n" prefix mechanism.

Env: TOKENMAP_PATH, ACTOR_CORRELATION_ID
"""

import sys, os, json
import msgpack, lmdb, spacy

NLP = spacy.load("en_core_web_sm")

TOKENMAP_PATH = os.environ.get("TOKENMAP_PATH", "/tmp/actor-mesh/tokenmap")
CORR_ID       = os.environ.get("ACTOR_CORRELATION_ID", "default")

def open_tokenmap():
    os.makedirs(TOKENMAP_PATH, exist_ok=True)
    return lmdb.open(TOKENMAP_PATH, max_dbs=1, map_size=64 * 1024 * 1024)

def get_or_create_token(txn, db, text, label):
    lookup_key = f"{CORR_ID}:text:{text}".encode()
    existing = txn.get(lookup_key, db=db)
    if existing:
        return existing.decode()
    count_key = f"{CORR_ID}:__count__".encode()
    n = int(txn.get(count_key, db=db) or b"0")
    txn.put(count_key, str(n + 1).encode(), db=db)
    tok = f"TOK_{n:04d}"
    txn.put(lookup_key, tok.encode(), db=db)
    txn.put(f"{CORR_ID}:tok:{tok}".encode(),
            json.dumps({"text": text, "label": label}).encode(), db=db)
    return tok

def mask_text(text, env, db):
    doc = NLP(text)
    if not doc.ents:
        return text
    result = []
    last = 0
    with env.begin(write=True, db=db) as txn:
        for ent in doc.ents:
            result.append(text[last:ent.start_char])
            result.append(get_or_create_token(txn, db, ent.text, ent.label_))
            last = ent.end_char
    result.append(text[last:])
    return "".join(result)

def mask_value(val, env, db):
    if isinstance(val, str):  return mask_text(val, env, db)
    if isinstance(val, list): return [mask_value(v, env, db) for v in val]
    if isinstance(val, dict): return {k: mask_value(v, env, db) for k, v in val.items()}
    return val

def main():
    raw = sys.stdin.buffer.read()
    msg = msgpack.unpackb(raw, raw=False)
    env = open_tokenmap()
    db  = env.open_db(b"tokenmap")
    out = mask_value(msg, env, db)
    out["type"] = msg.get("type", "") + "_masked"
    env.close()
    topic   = out["type"]
    payload = msgpack.packb(out, use_bin_type=True)
    sys.stdout.buffer.write(topic.encode() + b"\n" + payload)

if __name__ == "__main__":
    main()
