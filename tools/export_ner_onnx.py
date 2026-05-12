#!/usr/bin/env python3
"""
Export spaCy en_core_web_sm full NER pipeline to ONNX.

Input:  (seq_len, 16) uint64  — raw MurmurHash3 output for 4 cols × 4 keys
                                 C handler computes MurmurHash3, feeds directly.
                                 Model does Mod(nV) internally.

Output: (seq_len+2, nF=3, nO=64, nP=2) float32
        Precomputed affine table for the NER greedy decoder.
        Row 0 = boundary pad.  Rows 1..seq_len+1 = per-token features.

Decoder weights stored in metadata_props:
  pa_b      — (nO=64, nP=2)   bias added after gather-sum, before maxout
  linear_W  — (n_moves=74, nO=64)  linear head weights
  linear_b  — (n_moves=74,)        linear head bias
  moves     — comma-separated BILUO move names (index == move id)

Full layer map (from inspecting thinc model tree):

  hashembed × 4  →  concat(384)
    → maxout(384→96, P=3) + layernorm
    → 4 × residual( expand_window(288) → maxout(288→96,P=3) + layernorm )
    → linear(96→64)

Usage: python3 tools/export_ner_onnx.py [--smoke-test]
"""

import argparse

import numpy as np
import onnx
import onnx.checker
import onnx.helper as oh
import onnx.numpy_helper as onh

# ── name counter ──────────────────────────────────────────────────────────────

_n = 0


def n(prefix="v"):
    global _n
    _n += 1
    return f"{prefix}_{_n}"


def f32(name, arr):
    return onh.from_array(arr.astype(np.float32), name=name)


def i64(name, arr):
    return onh.from_array(arr.astype(np.int64), name=name)


def u64(name, arr):
    return onh.from_array(arr.astype(np.uint64), name=name)


# ── graph builder ─────────────────────────────────────────────────────────────


class Graph:
    def __init__(self):
        self.nodes = []
        self.inits = []

    def op(self, op, inputs, outputs, **kw):
        self.nodes.append(oh.make_node(op, inputs, outputs, **kw))
        return outputs[0] if len(outputs) == 1 else outputs

    def const_f32(self, arr):
        nm = n("c")
        self.inits.append(f32(nm, arr))
        return nm

    def const_i64(self, arr):
        nm = n("c")
        self.inits.append(i64(nm, arr))
        return nm

    def const_u64(self, arr):
        nm = n("c")
        self.inits.append(u64(nm, arr))
        return nm


# ── layer builders ────────────────────────────────────────────────────────────


def build_hashembed(g, col_slice, nV, E):
    """
    col_slice: (seq_len, 4) uint64 — raw MurmurHash3 keys for one column
    Mod inside graph, Gather from embedding table, sum 4 results.
    """
    nV_c = g.const_u64(np.array([nV], dtype=np.uint64))
    E_c = g.const_f32(E)
    gs = []
    for k in range(4):
        idx_c = g.const_i64(np.array([k]))
        raw_k = g.op("Gather", [col_slice, idx_c], [n("raw")], axis=1)
        ax1 = g.const_i64(np.array([1]))
        raw_k = g.op("Squeeze", [raw_k, ax1], [n("sq")])
        modd = g.op("Mod", [raw_k, nV_c], [n("mod")], fmod=0)
        idx = g.op("Cast", [modd], [n("idx")], to=onnx.TensorProto.INT64)
        emb = g.op("Gather", [E_c, idx], [n("emb")], axis=0)
        gs.append(emb)
    a = g.op("Add", [gs[0], gs[1]], [n("a")])
    b = g.op("Add", [a, gs[2]], [n("a")])
    return g.op("Add", [b, gs[3]], [n("emb_out")])


def build_maxout(g, x, W, b):
    """
    W: (nO, nP, nI)  b: (nO, nP)
    out = max over pieces of (x @ W[piece].T + b[piece])
    Implemented as: Gemm(x, W.reshape(nO*nP, nI).T) + b → reshape → ReduceMax
    """
    nO, nP, nI = W.shape
    W_c = g.const_f32(W.reshape(nO * nP, nI))
    b_c = g.const_f32(b.reshape(nO * nP))
    shp_c = g.const_i64(np.array([-1, nO, nP]))
    lin = g.op("Gemm", [x, W_c, b_c], [n("mlin")], transB=1)
    rs = g.op("Reshape", [lin, shp_c], [n("mrs")])
    return g.op("ReduceMax", [rs], [n("mout")], axes=[2], keepdims=0)


def build_layernorm(g, x, G, b, eps=1e-8):
    G_c = g.const_f32(G)
    b_c = g.const_f32(b)
    eps_c = g.const_f32(np.array([eps], dtype=np.float32))
    mu = g.op("ReduceMean", [x], [n("mu")], axes=[-1], keepdims=1)
    d = g.op("Sub", [x, mu], [n("d")])
    sq = g.op("Mul", [d, d], [n("sq")])
    var = g.op("ReduceMean", [sq], [n("var")], axes=[-1], keepdims=1)
    ve = g.op("Add", [var, eps_c], [n("ve")])
    std = g.op("Sqrt", [ve], [n("std")])
    norm = g.op("Div", [d, std], [n("norm")])
    sc = g.op("Mul", [norm, G_c], [n("sc")])
    return g.op("Add", [sc, b_c], [n("ln")])


def build_expand_window(g, x, window_size=1):
    """
    Pad sequence and slide a window of (2w+1) tokens.
    Input: (seq_len, dim)  Output: (seq_len, (2w+1)*dim)
    """
    w = window_size
    n_ctx = 2 * w + 1
    pads_c = g.const_i64(np.array([w, 0, w, 0]))
    padded = g.op("Pad", [x, pads_c], [n("padded")], mode="constant")
    sl1d = g.op("Shape", [x], [n("sl1d")], start=0, end=1)
    slices = []
    for off in range(n_ctx):
        st_c = g.const_i64(np.array([off]))
        end_c = g.op("Add", [sl1d, g.const_i64(np.array([off]))], [n("end")])
        ax_c = g.const_i64(np.array([0]))
        slices.append(g.op("Slice", [padded, st_c, end_c, ax_c], [n("sl")]))
    return g.op("Concat", slices, [n("ew")], axis=1)


def build_linear(g, x, W, b):
    W_c = g.const_f32(W)
    b_c = g.const_f32(b)
    return g.op("Gemm", [x, W_c, b_c], [n("lin")], transB=1)


# ── main ──────────────────────────────────────────────────────────────────────


def load_model():
    import spacy

    print("Loading en_core_web_sm ...")
    nlp = spacy.load("en_core_web_sm")
    ner = nlp.get_pipe("ner").model

    # Navigate model tree (verified against live inspection)
    #
    # ner (_layers)
    #   [0] chain nO=64
    #       [0] chain nO=96
    #           [0] embed chain
    #               [0] extract_features  (columns attr)
    #               [1] list2ragged       (no-op in ONNX)
    #               [2] with_array(concat of 4 hashembeds)
    #                   [0] concat
    #                       [0..3] ints-getitem >> hashembed
    #                              [1] hashembed  (E, seed, nV)
    #               [3] with_array(maxout>>layernorm>>dropout)
    #                   [0] chain
    #                       [0] maxout>>layernorm
    #                           [0] maxout   (W, b)
    #                           [1] layernorm(G, b)
    #               [4] ragged2list       (no-op)
    #           [1] with_array(4 residuals)
    #               [0] chain of 4 residuals
    #                   [0..3] residual
    #                       [0] chain
    #                           [0] expand_window (window_size)
    #                           [1] chain
    #                               [0] maxout>>layernorm
    #                                   [0] maxout   (W, b)
    #                                   [1] layernorm(G, b)
    #       [1] list2array                (no-op)
    #       [2] linear(96→64)             (W, b)
    #   [1] precomputable_affine          (exported separately / not needed for tok2vec)
    #   [2] linear(74 classes)            (exported separately)

    big_chain = ner._layers[0]
    inner_chain = big_chain._layers[0]
    embed_chain = inner_chain._layers[0]
    res_wa = inner_chain._layers[1]
    proj_linear = big_chain._layers[2]

    columns = embed_chain._layers[0].attrs["columns"]
    concat_he = embed_chain._layers[2]._layers[0]
    proj_mo_ln = embed_chain._layers[3]._layers[0]._layers[0]
    res_chain = res_wa._layers[0]

    return dict(
        columns=columns,
        concat_he=concat_he,
        proj_mo_ln=proj_mo_ln,
        res_wa=res_wa,
        res_chain=res_chain,
        proj_linear=proj_linear,
    )


def export(output_path):
    m = load_model()
    g = Graph()

    concat_he = m["concat_he"]
    proj_mo_ln = m["proj_mo_ln"]
    res_wa = m["res_wa"]
    res_chain = m["res_chain"]
    proj_linear = m["proj_linear"]
    columns = m["columns"]

    n_cols = len(concat_he._layers)  # 4
    n_keys = n_cols * 4  # 16

    inp = "token_hashes"  # (seq_len, 16) uint64 — raw MurmurHash3, no pre-mod

    # ── 4 hashembeds → concat ─────────────────────────────────────────────────
    embeds = []
    for col_idx, chain in enumerate(concat_he._layers):
        he = chain._layers[1]
        E = he.get_param("E")
        nV = he.get_dim("nV")
        seed = he.attrs["seed"]
        # slice columns col_idx*4 : col_idx*4+4  (the 4 keys for this column)
        st_c = g.const_i64(np.array([col_idx * 4]))
        en_c = g.const_i64(np.array([col_idx * 4 + 4]))
        ax_c = g.const_i64(np.array([1]))
        col4 = g.op("Slice", [inp, st_c, en_c, ax_c], [n("col4")])
        embeds.append(build_hashembed(g, col4, nV, E))
        print(f"  hashembed col{col_idx}: E{E.shape} nV={nV} seed={seed}")

    tok_emb = g.op("Concat", embeds, [n("tok_emb")], axis=1)
    print(f"  concat: {n_cols} × 96 = {n_cols * 96}")

    # ── projection: maxout(384→96) + layernorm ────────────────────────────────
    mo = proj_mo_ln._layers[0]
    ln = proj_mo_ln._layers[1]
    proj = build_maxout(g, tok_emb, mo.get_param("W"), mo.get_param("b"))
    proj = build_layernorm(g, proj, ln.get_param("G"), ln.get_param("b"))
    print(f"  projection maxout+ln: W{mo.get_param('W').shape}")

    # ── 4 residual CNN blocks ─────────────────────────────────────────────────
    # thinc wraps residuals in with_array(pad=4): pad 4 zero rows before/after
    # so boundary tokens see zeros up to 4 hops away, matching spaCy's behaviour.
    pad = res_wa.attrs.get("pad", 4)
    pad_c = g.const_i64(np.array([pad, 0, pad, 0]))
    sl1d_orig = g.op("Shape", [proj], [n("sl1d_orig")], start=0, end=1)
    cur = g.op("Pad", [proj, pad_c], [n("res_padded")], mode="constant")

    for i, res_block in enumerate(res_chain._layers):
        inner = res_block._layers[0]
        ew = inner._layers[0]
        mo_ln = inner._layers[1]._layers[0]
        mo_b = mo_ln._layers[0]
        ln_b = mo_ln._layers[1]
        ws = ew.attrs.get("window_size", 1)
        expanded = build_expand_window(g, cur, ws)
        new = build_maxout(g, expanded, mo_b.get_param("W"), mo_b.get_param("b"))
        new = build_layernorm(g, new, ln_b.get_param("G"), ln_b.get_param("b"))
        cur = g.op("Add", [cur, new], [n("res")])
        print(f"  residual {i + 1}: window={ws} W{mo_b.get_param('W').shape}")

    # unpad: Slice rows [pad : pad+seq_len]
    pad_start = g.const_i64(np.array([pad]))
    pad_end = g.op("Add", [sl1d_orig, g.const_i64(np.array([pad]))], [n("unpad_end")])
    ax0 = g.const_i64(np.array([0]))
    cur = g.op("Slice", [cur, pad_start, pad_end, ax0], [n("unpadded")])

    # ── linear 96→64 (tok2vec final projection) ───────────────────────────────
    tok2vec = build_linear(
        g, cur, proj_linear.get_param("W"), proj_linear.get_param("b")
    )
    print(f"  linear: W{proj_linear.get_param('W').shape}")

    # ── precomputable_affine  (seq, 64) → (seq+2, nF=3, nO=64, nP=2) ─────────
    #
    # Forward (from spacy/ml/parser_model.pyx):
    #   Yf = zeros(seq+1, nF*nO*nP)
    #   Yf[1:] = tok2vec @ W_pa.reshape(nF*nO*nP, nI).T
    #   Yf = Yf.reshape(seq+1, nF, nO, nP)
    #   Yf[0] = pad
    #
    # The big_chain prepends one boundary zero row, so tok2vec is (seq+1, 64).
    # We replicate that boundary prepend here, then apply W_pa.
    import spacy as _spacy

    _nlp = _spacy.load("en_core_web_sm")
    _pa = _nlp.get_pipe("ner").model._layers[1]
    _linear = _nlp.get_pipe("ner").model._layers[2]

    W_pa = _pa.get_param("W")  # (nF=3, nO=64, nP=2, nI=64)
    pad_pa = _pa.get_param("pad")  # (1, nF=3, nO=64, nP=2)
    nF_pa, nO_pa, nP_pa, nI_pa = W_pa.shape

    # precomputable_affine forward (parser_model.pyx):
    #   Yf = zeros(n+1, nF*nO*nP)
    #   Yf[1:] = tok2vec @ W.reshape(nF*nO*nP, nI).T
    #   Yf[0]  = pad
    # tok2vec is (seq, 64) — NO extra boundary row prepended
    W_pa_flat = g.const_f32(W_pa.reshape(nF_pa * nO_pa * nP_pa, nI_pa))
    yf_body = g.op("Gemm", [tok2vec, W_pa_flat], [n("yf_body")], transB=1)

    # prepend pad row: (seq, nF*nO*nP) → (seq+1, nF*nO*nP)
    pad_flat = g.const_f32(pad_pa.reshape(1, nF_pa * nO_pa * nP_pa))
    yf_flat = g.op("Concat", [pad_flat, yf_body], [n("yf_flat")], axis=0)

    # reshape to (seq+1, nF, nO, nP)
    seq_plus1 = g.op("Shape", [yf_flat], [n("sp1")], start=0, end=1)
    shape_4d = g.op(
        "Concat",
        [
            seq_plus1,
            g.const_i64(np.array([nF_pa])),
            g.const_i64(np.array([nO_pa])),
            g.const_i64(np.array([nP_pa])),
        ],
        [n("shape4d")],
        axis=0,
    )
    out = g.op("Reshape", [yf_flat, shape_4d], [n("pa_table")])
    print(
        f"  precomputable_affine: W{W_pa.shape} → table (seq+1, {nF_pa}, {nO_pa}, {nP_pa})"
    )

    # ── build and save ────────────────────────────────────────────────────────
    import base64 as _b64

    graph = oh.make_graph(
        g.nodes,
        "spacy_ner_full",
        [oh.make_tensor_value_info(inp, onnx.TensorProto.UINT64, ["seq_len", n_keys])],
        [
            oh.make_tensor_value_info(
                out, onnx.TensorProto.FLOAT, ["seq_plus1", nF_pa, nO_pa, nP_pa]
            )
        ],
        initializer=g.inits,
    )
    model_proto = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model_proto.ir_version = 8

    def _meta(key, value):
        e = model_proto.metadata_props.add()
        e.key, e.value = key, value

    # hashembed column metadata (for C feature extraction)
    for col_idx, chain in enumerate(load_model()["concat_he"]._layers):
        he = chain._layers[1]
        _meta(
            f"col_{col_idx}",
            f"{columns[col_idx]}:{he.get_dim('nV')}:{he.attrs['seed']}",
        )

    # decoder weights as base64-encoded little-endian float32 bytes
    _meta(
        "pa_b", _b64.b64encode(_pa.get_param("b").astype(np.float32).tobytes()).decode()
    )
    _meta(
        "linear_W",
        _b64.b64encode(_linear.get_param("W").astype(np.float32).tobytes()).decode(),
    )
    _meta(
        "linear_b",
        _b64.b64encode(_linear.get_param("b").astype(np.float32).tobytes()).decode(),
    )
    _meta("pa_nF", str(nF_pa))
    _meta("pa_nO", str(nO_pa))
    _meta("pa_nP", str(nP_pa))

    # BILUO move names via oracle sequences — order is B×18, I×18, L×18, U×18, O, MISSING
    from spacy.training import Example as _Example

    _nlp2 = _spacy.load("en_core_web_sm")
    _moves = _nlp2.get_pipe("ner").moves
    _ner2 = _nlp2.get_pipe("ner")
    _labels = list(_ner2.labels)
    _n = _moves.n_moves
    move_names = ["MISSING"] * _n

    for _lbl in _labels:
        _ex = _Example.from_dict(_nlp2.make_doc("hello"), {"entities": [(0, 5, _lbl)]})
        try:
            _seq = _moves.get_oracle_sequence(_ex)
            if _seq:
                move_names[_seq[0]] = f"U-{_lbl}"
        except:
            pass
        _ex3 = _Example.from_dict(_nlp2.make_doc("a b c"), {"entities": [(0, 5, _lbl)]})
        try:
            _seq3 = _moves.get_oracle_sequence(_ex3)
            if len(_seq3) >= 3:
                move_names[_seq3[0]] = f"B-{_lbl}"
                move_names[_seq3[1]] = f"I-{_lbl}"
                move_names[_seq3[2]] = f"L-{_lbl}"
        except:
            pass
    _ex_o = _Example.from_dict(_nlp2.make_doc("hi"), {"entities": []})
    _seq_o = _moves.get_oracle_sequence(_ex_o)
    if _seq_o:
        move_names[_seq_o[0]] = "O"

    _meta("moves", ",".join(move_names))
    _meta("n_moves", str(_n))
    print(f"  moves: {_n} (sample: {move_names[:4]} ...)")

    print("\nChecking ONNX model ...")
    onnx.checker.check_model(model_proto)
    print("  check passed")
    onnx.save(model_proto, output_path)
    size_kb = model_proto.ByteSize() // 1024
    print(f"Saved: {output_path}  ({size_kb} KB)")


def smoke_test(model_path):
    import onnxruntime as ort
    import spacy
    from thinc.backends import NumpyOps

    nlp = spacy.load("en_core_web_sm")
    text = "Apple CEO Tim Cook met with Sundar Pichai in New York last Tuesday."
    doc = nlp(text)
    print(f"\nspaCy entities: {[(e.text, e.label_) for e in doc.ents]}")

    ner = nlp.get_pipe("ner").model
    big_chain = ner._layers[0]
    inner_chain = big_chain._layers[0]
    embed_chain = inner_chain._layers[0]
    concat_he = embed_chain._layers[2]._layers[0]
    columns = embed_chain._layers[0].attrs["columns"]

    # build uint64 hash matrix — raw MurmurHash3, no mod
    ops = NumpyOps()
    raw = doc.to_array(columns).astype(np.uint64)
    seq = raw.shape[0]
    feats = np.zeros((seq, len(concat_he._layers) * 4), dtype=np.uint64)
    for col_idx, chain in enumerate(concat_he._layers):
        seed = chain._layers[1].attrs["seed"]
        keys = ops.hash(np.ascontiguousarray(raw[:, col_idx]), seed)  # (seq, 4) uint64
        feats[:, col_idx * 4 : col_idx * 4 + 4] = keys
    print(f"\nFeature matrix: {feats.shape} {feats.dtype}")

    # run ONNX
    sess = ort.InferenceSession(model_path)
    onnx_out = sess.run(None, {"token_hashes": feats})[0]  # (seq, 64)
    print(f"ONNX output: {onnx_out.shape}")

    # get spaCy tok2vec output for comparison
    # hook list2array (converts ragged list → 2D array) which sits after residuals
    list2array_m = big_chain._layers[1]
    captured = []
    orig_fn = list2array_m._func

    def hook(m, X, is_train):
        res, bp = orig_fn(m, X, is_train)
        captured.append(np.array(res))
        return res, bp

    list2array_m._func = hook
    nlp.get_pipe("ner").predict([doc])
    list2array_m._func = orig_fn

    spacy_96 = captured[0]  # (seq, 96) — before the final linear
    print(f"spaCy tok2vec (pre-linear) shape: {spacy_96.shape}")

    # apply the final linear manually to get (seq, 64)
    lin_W = big_chain._layers[2].get_param("W")  # (64, 96)
    lin_b = big_chain._layers[2].get_param("b")  # (64,)
    spacy_64 = spacy_96 @ lin_W.T + lin_b

    diff = np.abs(spacy_64 - onnx_out)
    print(f"\nComparison (seq, 64):")
    print(f"  spaCy[0,:4] = {spacy_64[0, :4]}")
    print(f"  ONNX [0,:4] = {onnx_out[0, :4]}")
    print(f"  max abs diff:  {diff.max():.6f}")
    print(f"  mean abs diff: {diff.mean():.6f}")

    tol = 1e-3
    if diff.max() < tol:
        print(f"\n  PASS — max diff {diff.max():.2e} < {tol}")
    else:
        print(f"\n  FAIL — max diff {diff.max():.2e} >= {tol}")
        # find worst token
        tok_idx = diff.max(axis=1).argmax()
        print(f"  worst token {tok_idx} ({doc[tok_idx].text!r}):")
        print(f"    spaCy: {spacy_64[tok_idx, :8]}")
        print(f"    ONNX:  {onnx_out[tok_idx, :8]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="models/ner.onnx")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    export(args.output)
    if args.smoke_test:
        smoke_test(args.output)
