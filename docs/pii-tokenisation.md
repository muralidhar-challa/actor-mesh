# PII Tokenisation — Design Notes

## Motivation

When the actor mesh routes queries through an external LLM (cloud API), raw PII — names, organisations, locations — travels in plaintext. The tokeniser/detokeniser pipeline intercepts tuples before they leave the trusted boundary, replaces entity spans with opaque tokens, and restores them on the way back.

The current `employee-mesh` demo uses a local Ollama model so this pipeline is optional. It becomes load-bearing the moment the LLM endpoint moves off-host.

---

## Pipeline Position in the Mesh

```
[user] ──user_message──► [tokenise-actor] ──user_message_masked──► [llm-agent]
                                │                                        │
                         writes token map                        calls query_db
                         to LMDB[corr_id]                               │
                                                                  sql_query (masked)
                                                                        │
                                                               [sqlite-tool]
                                                                        │
                                                                  sql_result (masked)
                                                                        │
                                                              [llm-agent] ──agent_response_masked──► [detokenise-actor]
                                                                                                              │
                                                                                                     reads LMDB[corr_id]
                                                                                                     fills token gaps
                                                                                                              │
                                                                                               agent_response ──► [tui/client]
```

Topics introduced:
- `user_message_masked` — tokenised query from tokenise-actor to llm-agent
- `agent_response_masked` — llm-agent output before detokenisation
- `agent_response` — final restored response to client (existing topic, same as today)

The `correlation_id` in the tuple header is the shared key: both tokenise-actor and detokenise-actor use it to read/write the same LMDB token map, even though they are separate processes with no direct connection.

---

## Token Map Format

Stored in LMDB, keyed by `corr_id + ":" + token_id`:

```
KEY   <corr_id>:TOK_0001  →  VALUE  {"text": "John Smith", "label": "PERSON"}
KEY   <corr_id>:TOK_0002  →  VALUE  {"text": "Acme Corp",  "label": "ORG"}
```

Or more compactly as msgpack. Token IDs are zero-padded counters per `corr_id`, which keeps them stable across multiple turns of the same conversation.

---

## Query Tokenisation Problem

A concern: if the user asks *"what is John Smith's salary?"*, the SQL the agent generates will contain `TOK_0001` in the WHERE clause, which won't match any row in the database.

**Resolution — already solved by the current setup.** The agent never queries by name directly. It explores the schema, finds numeric IDs, and uses those in WHERE clauses. Names appear only in final prose answers, not in SQL predicates. The detokeniser restores them in the response text.

If a future query genuinely requires a name lookup, the correct fix is option 3 from the original analysis: the tokenise-actor first runs a pre-query to resolve the name to an employee_id, substitutes the ID into the masked message, and the name never reaches the LLM at all.

---

## NER Model — Architecture Findings

### Model: `en_core_web_sm` (spaCy 3.x)

Size: ~12 MB on disk. Backend: **pure numpy** (thinc `NumpyOps`). No PyTorch dependency at inference time.

Pipeline components: `tok2vec`, `tagger`, `parser`, `attribute_ruler`, `lemmatizer`, `ner`

NER labels: `PERSON`, `NORP`, `FAC`, `ORG`, `GPE`, `LOC`, `PRODUCT`, `EVENT`, `WORK_OF_ART`, `LAW`, `LANGUAGE`, `DATE`, `TIME`, `PERCENT`, `MONEY`, `QUANTITY`, `ORDINAL`, `CARDINAL`

### Layer Inventory (tok2vec + ner)

Every layer maps directly to standard ONNX ops:

| thinc layer | ONNX equivalent |
|---|---|
| `hashembed` | `Gather` (embedding lookup) |
| `expand_window` | `Concat` of neighbour slices (sliding window over token dim) |
| `maxout` | `Gemm` → `Reshape` → `ReduceMax` |
| `layernorm` | `MeanVarianceNormalization` or manual `Sub/Div/Mul/Add` |
| `residual` | `Add` (skip connection) |
| `linear` | `Gemm` |
| `precomputable_affine` | `Einsum` or explicit `Gemm` + `Reshape` |

Weight shapes (tok2vec):
- hashembed embeddings: 6 tables, shapes `(5000,96)` `(1000,96)` `(2500,96)` × 2 `(50,96)` × 2
- maxout W: `(96, 3, 576)` for first layer, `(96, 3, 288)` for residual layers
- layernorm G/b: `(96,)`

Weight shapes (ner head):
- 4 hashembed tables (same pattern, smaller set)
- linear W: `(64, 96)` → `(74, 64)` (74 = NER output classes, transition-based parser states)
- precomputable_affine W: `(3, 64, 2, 64)`

### ONNX Export Strategy

No tracing. No torch. Manual graph construction using `onnx.helper.make_node`:

1. Walk the thinc model tree (depth-first `m._layers`)
2. For each node type, emit the corresponding ONNX node(s) and register weights as initializers
3. Wire inputs/outputs by generated tensor names
4. Call `onnx.checker.check_model` and save

The `hashembed` lookup is the only non-trivial part: thinc hashes the input integer modulo the table size before the gather. This is a single `Mod` + `Gather` in ONNX.

The `expand_window` (window_size=1) concatenates `[token[i-1], token[i], token[i+1]]` — three `Gather` by index offset + `Concat`.

The `precomputable_affine` in the NER head is the transition parser's state-action scorer. It can be precomputed over the token vectors once and reused, but for export simplicity it can be materialised as a single `Einsum`.

---

## NER Model Options

### Option A — `en_core_web_sm` (spaCy, 6 MB) ← preferred long-term
Pure numpy, thinc CNN backend. No PyTorch dependency at runtime.
Manual ONNX export script written at `tools/export_ner_onnx.py`.

**Status**: export produces a valid ONNX graph (`onnx.checker` passes, inference runs).
Numerical verification still in progress — the thinc `hashembed` uses MurmurHash3
with 4 keys per token (`gather_add`), which is correctly implemented in the export script.
The remaining ~0.08 mean absolute difference in tok2vec output needs tracing before
the model can be trusted for production use.

**Input format**: `(seq_len, 16)` int64 — 4 columns × 4 MurmurHash keys, pre-computed
outside the graph using `thinc.backends.NumpyOps.hash(ids, seed) % nV`.
Column order: `NORM, PREFIX, SUFFIX, SHAPE`.
Metadata stored in `model.onnx` props: `col_N → "COLNAME:table_size:seed"`.

**To resume**: fix the numerical mismatch by comparing each residual block output
against spaCy's NER internal tok2vec (not the shared tok2vec pipe — they differ).

### Option B — `elastic/distilbert-base-uncased-finetuned-conll03-english` (253 MB) ← current
HuggingFace PyTorch model, exported via `optimum` in one line.
Saved at `models/distilbert-ner/model.onnx`.
Tokenizer (WordPiece) saved alongside at `models/distilbert-ner/`.
Labels: `PER, ORG, LOC, MISC`.
Aggregation strategy: `first` (merges WordPiece subwords correctly).

**Input**: WordPiece token IDs + attention mask — tokenizer must run before ONNX inference.
The C handler needs to implement or call a WordPiece tokenizer.

---

## Export Scripts

- `tools/export_ner_onnx.py` — manual spaCy/thinc → ONNX export (Option A)
- Option B exported via: `ORTModelForTokenClassification.from_pretrained(..., export=True)`

## C Handler Locations (to be written)

`examples/employee-mesh/handlers/tokenise.c` — ONNX Runtime C API, loads NER model,
reads mpack from stdin, writes masked mpack + token map to LMDB[corr_id].

`examples/employee-mesh/handlers/detokenise.c` — reads LMDB token map,
substitutes tokens back into response text, writes final mpack to stdout.

---

## Sequencing

1. ~~Export NER model~~ — done (Option B at `models/distilbert-ner/model.onnx`)
2. **Write** `handlers/tokenise.c` — WordPiece tokenizer in C + ONNX Runtime inference + LMDB write
3. **Write** `handlers/detokenise.c`
4. **Wire** new topics into `Makefile`
5. **End-to-end test**: verify name never appears in llm-agent log, restored in TUI output
6. *(Optional)* Fix Option A numerical mismatch and switch to 6 MB model
