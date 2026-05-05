#!/usr/bin/env python3
"""
handlers/detokenise.py — PII detokeniser

One topic in, one topic out. Restores TOK_NNNN tokens in all string fields.
Actor sets ACTOR_RESULT_TOPIC=agent_response.

Env: TOKENMAP_PATH, ACTOR_CORRELATION_ID
"""

import sys, os, json, re
import msgpack, lmdb

TOKENMAP_PATH = os.environ.get("TOKENMAP_PATH", "/tmp/actor-mesh/tokenmap")
CORR_ID       = os.environ.get("ACTOR_CORRELATION_ID", "default")
TOKEN_RE      = re.compile(r'TOK_\d{4}')

def open_tokenmap():
    os.makedirs(TOKENMAP_PATH, exist_ok=True)
    return lmdb.open(TOKENMAP_PATH, max_dbs=1, map_size=64 * 1024 * 1024, readonly=True)

def restore_text(text, txn, db):
    def replace(m):
        raw = txn.get(f"{CORR_ID}:tok:{m.group(0)}".encode(), db=db)
        return json.loads(raw)["text"] if raw else m.group(0)
    return TOKEN_RE.sub(replace, text)

def restore_value(val, txn, db):
    if isinstance(val, str):  return restore_text(val, txn, db)
    if isinstance(val, list): return [restore_value(v, txn, db) for v in val]
    if isinstance(val, dict): return {k: restore_value(v, txn, db) for k, v in val.items()}
    return val

def main():
    raw = sys.stdin.buffer.read()
    msg = msgpack.unpackb(raw, raw=False)
    try:
        env = open_tokenmap()
        db  = env.open_db(b"tokenmap")
        with env.begin(db=db) as txn:
            out = restore_value(msg, txn, db)
        env.close()
    except lmdb.Error:
        out = msg  # no token map yet — pass through
    out["type"] = "agent_response"
    sys.stdout.buffer.write(msgpack.packb(out, use_bin_type=True))

if __name__ == "__main__":
    main()
