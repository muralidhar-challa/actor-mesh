"""Export AdvancedNER to ONNX — standalone, no data loading."""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# ── Model architecture (must match train_advanced.py exactly) ──


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
        gate = self.se(pooled).unsqueeze(1)
        return x * gate


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
        residual = x
        out = self.conv(x)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.se(out)
        out = out.transpose(1, 2)
        out = torch.nn.functional.hardswish(out)
        out = self.drop(out)
        return out + residual


class HashEmbed(nn.Module):
    def __init__(self, nV, embed_dim):
        super().__init__()
        self.nV = nV
        self.embed = nn.Embedding(nV, embed_dim)

    def forward(self, col_slice):
        S = col_slice.shape[0]
        out = torch.zeros(S, self.embed.weight.shape[1], device=col_slice.device)
        for k in range(4):
            idx = (col_slice[:, k] % self.nV).long()
            out += self.embed(idx)
        return out


class AdvancedNER(nn.Module):
    def __init__(self, embed_dim=128, n_classes=33):
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
            col_slice = x[:, :, col * 4 : (col + 1) * 4].reshape(B * S, 4)
            emb = he(col_slice)
            embeds.append(emb.view(B, S, -1))
        emb = self.proj(torch.cat(embeds, dim=-1))
        emb = self.norm_in(emb)
        emb = self.rope(emb)
        emb_t = emb.transpose(1, 2)
        local = self.local_conv(emb_t).transpose(1, 2)
        local = self.local_norm(local)
        local = self.local_se(local).transpose(1, 2)
        local = torch.nn.functional.hardswish(local)
        g = emb_t
        for res in self.global_res:
            g = res(g)
        local_t = local.transpose(1, 2)
        g_t = g.transpose(1, 2)
        gate_w = torch.sigmoid(self.gate(self.drop(torch.cat([local_t, g_t], dim=-1))))
        fused = gate_w * local_t + (1 - gate_w) * g_t
        return self.ner(self.drop(fused))


# ── Export ──
DIM = 128
CKPT = "/home/max/Projects/mesh-actors/models/advanced_128_best.pt"
OUT = "/home/max/Projects/mesh-actors/models/ner_advanced_128.onnx"

model = AdvancedNER(embed_dim=DIM, n_classes=33)
ckpt = torch.load(CKPT, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded: {n_params:,} params")

dummy = torch.zeros(1, 10, 16, dtype=torch.long)
torch.onnx.export(
    model,
    dummy,
    OUT,
    input_names=["token_hashes"],
    output_names=["logits"],
    dynamic_axes={
        "token_hashes": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    },
    opset_version=14,
    do_constant_folding=True,
)

size_mb = os.path.getsize(OUT) / (1024 * 1024)
print(f"Exported: {OUT} ({size_mb:.1f} MB)")

# ── Verify roundtrip ──
import onnxruntime as ort

session = ort.InferenceSession(OUT)
test = torch.randint(0, 2**31, (1, 7, 16), dtype=torch.long).numpy()
with torch.no_grad():
    torch_out = model(torch.from_numpy(test)).numpy()
onnx_out = session.run(None, {"token_hashes": test})[0]
diff = np.abs(torch_out - onnx_out).max()
print(f"Max diff: {diff:.8f} {'✓ OK' if diff < 1e-4 else '✗ MISMATCH'}")
