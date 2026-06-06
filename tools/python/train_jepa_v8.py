"""v8 CharCNN + JEPA self-supervised pre-training + NER fine-tuning.

Two-phase pipeline:
  Phase 1 (JEPA): Asymmetric masking — context encoder sees 85% of tokens,
                   target encoder (EMA) sees 15%. Smooth L1 loss aligns a
                   depthwise predictor's output with the target encoder's
                   latent representations at masked positions. No labels needed.
  Phase 2 (NER):   Fine-tune on labeled NER data with CRF + token-CE loss.

Architecture mirrors Julia ner_lux_v8.jl:
  SharedCharCNN (1x, all 4 views) -> Encoder (3-path CNN, Conv1x1 gate) -> Decoder.

Speed optimizations: precomputed char IDs, AMP mixed precision.

Data: ner_data.json (Few-NERD + OntoNotes + CoNLL + WNUT, 18 types, BILOU tags)
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from tqdm import tqdm


# ============================================================
# Building blocks (matching Julia ner_lux_v8.jl)
# ============================================================

class SepConv(nn.Module):
    def __init__(self, d, out=None, dl=1):
        super().__init__()
        out = out or d
        self.dw = nn.Conv1d(d, d, 3, padding=dl, dilation=dl, groups=d)
        self.pw = nn.Conv1d(d, out, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class SEBlock(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.Conv1d(d, d // r, 1), nn.Hardswish(),
            nn.Conv1d(d // r, d, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x.mean(2, keepdim=True))


class SepResBlock(nn.Module):
    def __init__(self, d, dl=1):
        super().__init__()
        self.conv = SepConv(d, dl=dl)
        self.norm = nn.LayerNorm(d)
        self.se = SEBlock(d)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        r = x
        x = self.conv(x)
        x = x.transpose(1, 2); x = self.norm(x); x = x.transpose(1, 2)
        x = self.se(x)
        x = F.hardswish(x)
        return self.drop(x) + r


# ============================================================
# SharedCharCNN (v8: 1 CNN for all 4 token views)
# ============================================================

class SharedCharCNN(nn.Module):
    def __init__(self, char_vocab=257, char_dim=64, out_dim=64):
        super().__init__()
        self.embed = nn.Embedding(char_vocab, char_dim, padding_idx=0)
        self.conv1 = SepConv(char_dim, out=96)
        self.conv2 = SepConv(96, out=128)
        self.proj = nn.Conv1d(128, out_dim, 1)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        e = self.embed(x)
        e = e.transpose(1, 2)
        e = F.relu(self.conv1(e))
        e = F.relu(self.conv2(e))
        e = e.max(-1)[0]
        e = e.unsqueeze(-1)
        y = self.proj(e).squeeze(-1)
        return self.ln(y)


# ============================================================
# Encoder (3-path CNN backbone with Conv1x1 gate)
# ============================================================

class Encoder(nn.Module):
    def __init__(self, d=128, in_dim=256):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, d, 1)
        self.norm = nn.LayerNorm(d)
        self.lc = SepConv(d)
        self.ln = nn.LayerNorm(d)
        self.ls = SEBlock(d)
        self.gr = nn.ModuleList([
            SepResBlock(d, 1), SepResBlock(d, 3), SepResBlock(d, 5)])
        self.wr = nn.ModuleList([SepResBlock(d, 9)])
        self.gate = nn.Conv1d(d * 3, 3, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, e):
        e = self.proj(e)
        e = e.transpose(1, 2); e = self.norm(e); e = e.transpose(1, 2)

        l = self.lc(e)
        l = l.transpose(1, 2); l = self.ln(l); l = l.transpose(1, 2)
        l = self.ls(l); l = F.hardswish(l)

        g = e
        for block in self.gr: g = block(g)
        w = e
        for block in self.wr: w = block(w)

        gw = torch.cat([l, g, w], dim=1)
        gw = self.gate(self.drop(gw))
        gw = F.softmax(gw, dim=1)
        return gw[:, 0:1] * l + gw[:, 1:2] * g + gw[:, 2:3] * w


# ============================================================
# Backbone + Predictor + NERHead
# ============================================================

class CharCNNBackbone(nn.Module):
    def __init__(self, char_vocab=257, char_dim=64, cnn_out=64, enc_d=128,
                 n_views=4, char_len=20):
        super().__init__()
        self.char_cnn = SharedCharCNN(char_vocab, char_dim, cnn_out)
        self.encoder = Encoder(d=enc_d, in_dim=cnn_out * n_views)
        self.n_views = n_views
        self.cnn_out = cnn_out

    def forward(self, x):
        B, V, S, L = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * S * V, L)
        e = self.char_cnn(x_flat)
        e = e.view(B, S, V * self.cnn_out).transpose(1, 2)
        return self.encoder(e)


class Predictor(nn.Module):
    def __init__(self, d=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, hidden, 1), nn.BatchNorm1d(hidden),
            nn.GELU(), nn.Conv1d(hidden, d, 1))

    def forward(self, ctx):
        return self.net(ctx)


class NERHead(nn.Module):
    def __init__(self, d=128, n_tags=73):
        super().__init__()
        self.proj = nn.Conv1d(d, n_tags, 1)
        self.crf = CRF(n_tags, batch_first=True)

    def forward(self, h):
        return self.proj(h).transpose(1, 2)


# ============================================================
# Loss functions
# ============================================================

def jepa_loss(z1, z2, loss_type="smooth_l1"):
    if z1.numel() == 0:
        return z1.new_tensor(0.0)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(z1, z2)
    elif loss_type == "cosine":
        z1_n = F.normalize(z1, dim=-1)
        z2_n = F.normalize(z2, dim=-1)
        return (1.0 - (z1_n * z2_n).sum(-1)).mean()
    elif loss_type == "vicreg":
        N, D = z1.shape
        if N < 2:
            return z1.new_tensor(0.0)
        inv = F.mse_loss(z1, z2)
        eps = 1e-4
        std1 = torch.sqrt(z1.var(0) + eps)
        std2 = torch.sqrt(z2.var(0) + eps)
        var = F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()
        z1_c = z1 - z1.mean(0); z2_c = z2 - z2.mean(0)
        cov1 = (z1_c.T @ z1_c) / (N - 1)
        cov2 = (z2_c.T @ z2_c) / (N - 1)
        cov = ((cov1 - torch.diag(torch.diag(cov1))) ** 2).sum() / D + \
              ((cov2 - torch.diag(torch.diag(cov2))) ** 2).sum() / D
        return 25.0 * inv + 25.0 * var + cov
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ============================================================
# EMA update
# ============================================================

@torch.no_grad()
def ema_update(online, target, momentum=0.996):
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.data = momentum * pt.data + (1.0 - momentum) * po.data


# ============================================================
# Token features + precomputed batching
# ============================================================

def ner_shape(tok):
    o = []; last = "\0"; seq = 0
    for c in tok:
        sc = "X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c
        seq = seq + 1 if sc == last else 0; last = sc
        if seq < 4: o.append(sc)
    return "".join(o)


def token_views(tok):
    return (tok.lower(), tok[0] if tok else " ",
            tok[-3:] if len(tok) >= 3 else tok, ner_shape(tok))


def token_to_char_ids(token, max_len=20):
    ids = [0] * max_len
    raw = token.encode("utf-8", errors="replace")[:max_len]
    for i, c in enumerate(raw):
        ids[i] = c + 1
    return ids


def build_char_tensor(tokens, char_len=20):
    S = len(tokens)
    x = torch.zeros(4, S, char_len, dtype=torch.long)
    for s, tok in enumerate(tokens):
        views = token_views(tok)
        for col in range(4):
            x[col, s] = torch.tensor(token_to_char_ids(views[col], char_len))
    return x


def precompute_char_ids(examples, char_len=20, verbose=True):
    if isinstance(examples[0], dict):
        tokens_list = [ex["tokens"] for ex in examples]
        tags_list = [torch.tensor(ex.get("ner_tags", [0] * len(ex["tokens"])))
                     for ex in examples]
        has_tags = True
    else:
        tokens_list = examples
        tags_list = None
        has_tags = False

    xs = []
    iterator = tqdm(tokens_list, desc="precompute chars") if verbose else tokens_list
    for tokens in iterator:
        xs.append(build_char_tensor(tokens, char_len))

    if has_tags:
        return list(zip(xs, tags_list))
    return xs


def collate_batch(items, device=None):
    has_tags = isinstance(items[0], tuple)
    if has_tags:
        xs, ys = zip(*items)
    else:
        xs = items

    B = len(xs)
    Sm = max(x.shape[1] for x in xs)
    x = xs[0].new_zeros(B, 4, Sm, xs[0].shape[2])
    y = xs[0].new_zeros(B, Sm, dtype=torch.long) if has_tags else None
    m = xs[0].new_zeros(B, Sm, dtype=torch.bool)

    for b, xb in enumerate(xs):
        S = xb.shape[1]
        x[b, :, :S] = xb
        if has_tags:
            y[b, :S] = ys[b][:S]
        m[b, :S] = True

    if device is not None:
        x = x.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        if has_tags:
            y = y.to(device, non_blocking=True)

    return (x, y, m) if has_tags else (x, m)


# ============================================================
# JEPA masking
# ============================================================

def jepa_masks(x, mask, mask_ratio=0.15):
    B, _, S, _ = x.shape
    device = x.device
    pos_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    for b in range(B):
        valid = mask[b].nonzero(as_tuple=True)[0]
        n_valid = len(valid)
        n_mask = max(1, int(n_valid * mask_ratio))
        idx = torch.randperm(n_valid, device=device)[:n_mask]
        pos_mask[b, valid[idx]] = True

    ctx_x = x.clone()
    ctx_x[pos_mask.unsqueeze(1).unsqueeze(-1).expand_as(x)] = 0
    tgt_x = x.clone()
    keep_mask = ~pos_mask
    tgt_x[keep_mask.unsqueeze(1).unsqueeze(-1).expand_as(x)] = 0
    return ctx_x, tgt_x, pos_mask


# ============================================================
# Data loading + Span F1
# ============================================================

def load_data(json_path):
    data = json.load(open(json_path))
    return data["train"], data["val"]


def load_unlabeled(json_path):
    data = json.load(open(json_path))
    return [ex["tokens"] for ex in data["train"]] + \
           [ex["tokens"] for ex in data["val"]]


def extract_spans(tags):
    spans = set()
    i, n = 0, len(tags)
    while i < n:
        t = tags[i]
        if t == 0:
            i += 1
        else:
            t0 = t - 1; pos = t0 % 4; et = t0 // 4
            if pos == 3:
                spans.add((i, i, et)); i += 1
            elif pos == 0:
                s = i; i += 1
                while i < n and tags[i] != 0:
                    p = (tags[i] - 1) % 4
                    if p == 2: spans.add((s, i, et)); i += 1; break
                    elif p == 1: i += 1
                    else: break
            else:
                i += 1
    return spans


def span_f1(gold, pred):
    tg = tp = tc = 0
    for g, p in zip(gold, pred):
        gs = extract_spans(g); ps = extract_spans(p)
        tg += len(gs); tp += len(ps); tc += len(gs & ps)
    prec = tc / tp if tp > 0 else 0.0
    rec = tc / tg if tg > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


# ============================================================
# Phase 1: JEPA self-supervised pre-training
# ============================================================

def phase1_pretrain(ctx_encoder, tgt_encoder, predictor,
                    all_tokens, device,
                    epochs=20, batch_size=128, lr=0.001,
                    mask_ratio=0.15, ema_momentum=0.996,
                    ckpt_path="models/jepa_backbone.pt",
                    loss_type="smooth_l1", use_compile=False,
                    resume=False):

    print("  Precomputing character IDs for all examples...")
    char_tensors = precompute_char_ids(all_tokens, verbose=True)
    idx_by_len = sorted(range(len(char_tensors)),
                        key=lambda i: char_tensors[i].shape[1])

    fwd_backbone = torch.compile(ctx_encoder, mode="reduce-overhead") \
                   if use_compile else ctx_encoder
    fwd_predictor = torch.compile(predictor, mode="reduce-overhead") \
                    if use_compile else predictor

    opt = torch.optim.AdamW(
        list(ctx_encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=0.01)
    total_steps = epochs * (len(idx_by_len) // batch_size)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_ctx = lambda: torch.amp.autocast("cuda" if use_amp else "cpu")

    start_epoch = 0
    best_loss = float("inf")

    if resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path}...")
        ck = torch.load(ckpt_path, map_location=device)
        ctx_encoder.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "sch" in ck:
            sch.load_state_dict(ck["sch"])
        start_epoch = ck.get("epoch", 0)
        best_loss = ck.get("best_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best loss so far: {best_loss:.6f}")
        if start_epoch >= epochs:
            print(f"  Already completed {start_epoch}/{epochs}. Done.")
            return ctx_encoder

    tgt_encoder.load_state_dict(ctx_encoder.state_dict())
    for p in tgt_encoder.parameters():
        p.requires_grad = False

    ctag = " +compile" if use_compile else ""
    atag = " +AMP" if use_amp else ""
    print(f"\nPhase 1: JEPA pre-training (epochs {start_epoch+1}-{epochs}, B={batch_size})")
    print(f"  Unlabeled: {len(char_tensors):,}  Mask: {mask_ratio}  EMA: {ema_momentum}")
    print(f"  Loss: {loss_type}  LR: {lr}{atag}{ctag}")

    for epoch in range(start_epoch, epochs):
        ctx_encoder.train(); predictor.train(); tgt_encoder.eval()

        total_loss, n_batches = 0, 0
        perm = torch.randperm(len(idx_by_len))

        pbar = tqdm(range(0, len(perm), batch_size), desc=f"E{epoch+1}/{epochs}")
        for i in pbar:
            idx = [idx_by_len[perm[j]]
                   for j in range(i, min(i + batch_size, len(perm)))]
            x, m = collate_batch([char_tensors[j] for j in idx], device=device)
            ctx_x, tgt_x, pos_mask = jepa_masks(x, m, mask_ratio)

            with amp_ctx():
                ctx_h = fwd_backbone(ctx_x)
                pred = fwd_predictor(ctx_h)
                with torch.no_grad():
                    tgt_h = tgt_encoder(tgt_x)

                pos_flat = pos_mask.nonzero(as_tuple=False)
                if pos_flat.numel() == 0:
                    continue
                pred_vecs = pred[pos_flat[:, 0], :, pos_flat[:, 1]]
                tgt_vecs = tgt_h[pos_flat[:, 0], :, pos_flat[:, 1]]
                loss = jepa_loss(pred_vecs.float(), tgt_vecs.float(), loss_type)

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    list(ctx_encoder.parameters()) + list(predictor.parameters()), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(ctx_encoder.parameters()) + list(predictor.parameters()), 1.0)
                opt.step()
            sch.step()

            ema_update(ctx_encoder, tgt_encoder, ema_momentum)
            total_loss += loss.item(); n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_loss = total_loss / n_batches
        print(f"  loss={avg_loss:.4f}  lr={sch.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ctx_encoder.state_dict(), ckpt_path)
            print(f"  ↑ best -> {ckpt_path}")

        # Full resume checkpoint
        torch.save(
            {"model": ctx_encoder.state_dict(),
             "opt": opt.state_dict(), "sch": sch.state_dict(),
             "epoch": epoch + 1, "best_loss": best_loss},
            ckpt_path.replace(".pt", "_resume.pt"))

    print(f"\nPhase 1 done. Best loss: {best_loss:.4f}")
    ctx_encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    return ctx_encoder


# ============================================================
# Phase 2: NER fine-tuning
# ============================================================

def phase2_finetune(backbone, head, train_ex, val_ex, device,
                    epochs=80, batch_size=128, lr=0.001,
                    ce_weight=0.5, ce_decay=True,
                    ckpt_best="models/jepa_ner_best.pt",
                    ckpt_last="models/jepa_ner_last.pt",
                    use_compile=False):
    n_tags = head.proj.out_channels

    print("  Precomputing character IDs...")
    train_items = precompute_char_ids(train_ex)
    val_items = precompute_char_ids(val_ex)
    train_idx_by_len = sorted(range(len(train_items)),
                              key=lambda i: train_items[i][0].shape[1])

    train_tags = [ex["ner_tags"] for ex in train_ex]
    val_tags = [ex["ner_tags"] for ex in val_ex]
    all_tags = set()
    for tags in train_tags + val_tags:
        all_tags.update(tags)
    print(f"  Tag range: {min(all_tags)}-{max(all_tags)}  (expected 0-{n_tags-1})")

    fwd_backbone = torch.compile(backbone, mode="reduce-overhead") \
                   if use_compile else backbone

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_ctx = lambda: torch.amp.autocast("cuda" if use_amp else "cpu")

    opt = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_f1 = 0.0
    ctag = " +compile" if use_compile else ""
    atag = " +AMP" if use_amp else ""
    print(f"\nPhase 2: NER fine-tuning ({epochs} epochs, B={batch_size})")
    print(f"  Train: {len(train_ex):,}  Val: {len(val_ex):,}{atag}{ctag}")

    for epoch in range(epochs):
        backbone.train(); head.train()
        w = ce_weight * (1.0 - epoch / max(epochs - 1, 1)) if ce_decay else ce_weight

        total_loss, n_batches = 0, 0
        perm = torch.randperm(len(train_idx_by_len))

        pbar = tqdm(range(0, len(perm), batch_size), desc=f"E{epoch+1}/{epochs}")
        for i in pbar:
            idx = [train_idx_by_len[perm[j]]
                   for j in range(i, min(i + batch_size, len(perm)))]
            x, y, m = collate_batch([train_items[j] for j in idx], device=device)

            with amp_ctx():
                h = fwd_backbone(x)
                logits = head(h)
                crf_loss = -head.crf(logits.float(), y, mask=m, reduction="mean")
                lsm = F.log_softmax(logits.float(), dim=-1)
                token_ce = F.nll_loss(
                    lsm[m].view(-1, n_tags), y[m].view(-1), reduction="mean")
                loss = crf_loss + w * token_ce

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()), 1.0)
                opt.step()

            total_loss += loss.item(); n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.2f}")

        sch.step()

        # Validation
        backbone.eval(); head.eval()
        preds = []
        with torch.no_grad():
            for vi in range(0, len(val_items), batch_size):
                vend = min(vi + batch_size, len(val_items))
                xv, _, mv = collate_batch(
                    [val_items[j] for j in range(vi, vend)], device=device)
                with amp_ctx():
                    hv = fwd_backbone(xv)
                    logits_v = head(hv)
                p = head.crf.decode(logits_v.float(), mask=mv)
                preds.extend(p)

        trimmed_preds = [p[:len(vt)] for p, vt in zip(preds, val_tags)]
        sp, sr, sf = span_f1(val_tags, trimmed_preds)
        print(f"  loss={total_loss/n_batches:.4f}  P={sp:.4f}  R={sr:.4f}  "
              f"F1={sf:.4f}  ce_w={w:.3f}  lr={sch.get_last_lr()[0]:.6f}")

        if sf > best_f1:
            best_f1 = sf
            torch.save(
                {"backbone": backbone.state_dict(), "head": head.state_dict()},
                ckpt_best)
            print(f"  ^ best ({sf:.4f}) -> {ckpt_best}")

        torch.save(
            {"backbone": backbone.state_dict(), "head": head.state_dict(),
             "epoch": epoch + 1, "best_f1": best_f1,
             "opt": opt.state_dict(), "sch": sch.state_dict()},
            ckpt_last)

    print(f"\nPhase 2 done. Best span F1: {best_f1:.4f}")
    return best_f1


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ner_data.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase1-epochs", type=int, default=20)
    ap.add_argument("--phase2-epochs", type=int, default=80)
    ap.add_argument("--batch-p1", type=int, default=128)
    ap.add_argument("--batch-p2", type=int, default=128)
    ap.add_argument("--lr-p1", type=float, default=0.001)
    ap.add_argument("--lr-p2", type=float, default=0.001)
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--ce-weight", type=float, default=0.5)
    ap.add_argument("--loss", default="smooth_l1",
                    choices=["smooth_l1", "cosine", "vicreg"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--skip-phase1", action="store_true")
    ap.add_argument("--resume-phase1", action="store_true")
    ap.add_argument("--init-from", default="",
                    help="Load backbone weights before Phase 1 (e.g. prior best.pt)")
    ap.add_argument("--backbone-ckpt", default="models/jepa_backbone.pt")
    ap.add_argument("--ner-best", default="models/jepa_ner_best.pt")
    ap.add_argument("--ner-last", default="models/jepa_ner_last.pt")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_types = 18
    n_tags = n_types * 4 + 1
    os.makedirs("models", exist_ok=True)

    ctx_encoder = CharCNNBackbone().to(device)
    tgt_encoder = CharCNNBackbone().to(device)
    predictor = Predictor(d=128, hidden=256).to(device)

    n_backbone = sum(p.numel() for p in ctx_encoder.parameters())
    n_predictor = sum(p.numel() for p in predictor.parameters())
    print(f"Backbone params: {n_backbone:,}")
    print(f"Predictor params: {n_predictor:,}")

    if not args.skip_phase1:
        if args.init_from and os.path.exists(args.init_from):
            print(f"Initializing backbone from: {args.init_from}")
            ctx_encoder.load_state_dict(
                torch.load(args.init_from, map_location=device))
        all_tokens = load_unlabeled(args.data)
        ctx_encoder = phase1_pretrain(
            ctx_encoder, tgt_encoder, predictor, all_tokens, device,
            epochs=args.phase1_epochs, batch_size=args.batch_p1,
            lr=args.lr_p1, mask_ratio=args.mask_ratio,
            ema_momentum=args.ema, ckpt_path=args.backbone_ckpt,
            loss_type=args.loss, use_compile=args.compile,
            resume=args.resume_phase1)
    elif os.path.exists(args.backbone_ckpt):
        print(f"\nLoading pre-trained backbone: {args.backbone_ckpt}")
        ctx_encoder.load_state_dict(
            torch.load(args.backbone_ckpt, map_location=device))
    else:
        print("\nWarning: --skip-phase1 but no checkpoint found. Random init.")

    train_ex, val_ex = load_data(args.data)
    head = NERHead(d=128, n_tags=n_tags).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"NER head params: {n_head:,}")
    print(f"Total (Phase 2): {n_backbone + n_head:,}")

    phase2_finetune(
        ctx_encoder, head, train_ex, val_ex, device,
        epochs=args.phase2_epochs, batch_size=args.batch_p2,
        lr=args.lr_p2, ce_weight=args.ce_weight,
        ckpt_best=args.ner_best, ckpt_last=args.ner_last,
        use_compile=args.compile)


if __name__ == "__main__":
    main()
