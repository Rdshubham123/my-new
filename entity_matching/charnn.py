"""
Deep ensemble member: character-level cross-attention matcher (ESIM-style)
fused with the tabular features (wide & deep).

Each record -> 96 bytes: normalised name (32) + normalised address (64); both
are ASCII after transliteration.  A shared char-CNN encodes each side, soft
cross-attention aligns every character position of one side with the other
side (Chen et al. 2017, ESIM; Mudgal et al. 2018, DeepMatcher), the aligned
comparisons [h - a, h * a] are mean/max pooled, and an MLP fuses them with
the standardised tabular features.  It learns typo / transliteration /
reordering tolerance end-to-end, which gives the gradient-boosting members
a genuinely different error profile to average with.

Only used in phase C (after every fork), trained fold-wise with the same
GroupKFold split so its out-of-fold predictions are comparable, and chosen
into the blend only if OOF macro F0.5 improves.  A wall-clock budget cuts
epochs or skips the member instead of risking the deadline.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from .utils import log

NAME_LEN, ADDR_LEN = 32, 64
SEQ = NAME_LEN + ADDR_LEN


def _encode_unique(values, width):
    """Unique strings -> (n, width) uint8 matrix (ASCII, zero padded)."""
    buf = b"".join(str(v).encode("ascii", "replace")[:width].ljust(width, b"\0") for v in values)
    return np.frombuffer(buf, dtype=np.uint8).reshape(len(values), width).copy() if len(values) \
        else np.zeros((0, width), np.uint8)


class TextIndex:
    """Per-record byte text without materialising 10M strings: unique-value
    matrices + per-record codes (from the categorical columns)."""

    def __init__(self, df: pd.DataFrame, name_col="n_core", addr_col="a_atext"):
        parts = []
        for col, width in ((name_col, NAME_LEN), (addr_col, ADDR_LEN)):
            s = df[col]
            if not isinstance(s.dtype, pd.CategoricalDtype):
                s = s.astype("category")
            parts.append((_encode_unique(list(s.cat.categories), width),
                          s.cat.codes.to_numpy().astype(np.int32)))
        (self.nm, self.nc), (self.am, self.ac) = parts

    def to_arrays(self, prefix):
        return {prefix + "nm": self.nm, prefix + "nc": self.nc, prefix + "am": self.am, prefix + "ac": self.ac}

    @classmethod
    def from_arrays(cls, z, prefix):
        obj = cls.__new__(cls)
        obj.nm, obj.nc, obj.am, obj.ac = (z[prefix + k] for k in ("nm", "nc", "am", "ac"))
        return obj

    def rows(self, idx):
        idx = np.asarray(idx)
        n = self.nm[np.maximum(self.nc[idx], 0)]
        n[self.nc[idx] < 0] = 0
        a = self.am[np.maximum(self.ac[idx], 0)]
        a[self.ac[idx] < 0] = 0
        return np.concatenate([n, a], axis=1)


def _build_model(n_tab):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CharMatcher(nn.Module):
        def __init__(self, d=96, e=48):
            super().__init__()
            self.d = d
            self.emb = nn.Embedding(256, e, padding_idx=0)
            self.pos = nn.Parameter(torch.zeros(1, SEQ, e))
            self.c1 = nn.Conv1d(e, d, 5, padding=2)
            self.c2 = nn.Conv1d(d, d, 3, padding=1)
            self.c3 = nn.Conv1d(d, d, 3, padding=2, dilation=2)
            self.tab = nn.Sequential(nn.Linear(n_tab, 192), nn.GELU(), nn.Linear(192, 96), nn.GELU())
            self.head = nn.Sequential(nn.Linear(8 * d + 96, 256), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

        def enc(self, x):
            m = (x > 0)
            h = self.emb(x.long()) + self.pos
            h = F.gelu(self.c1(h.transpose(1, 2)))
            h = F.gelu(self.c2(h)) + h
            h = F.gelu(self.c3(h)) + h
            return h.transpose(1, 2), m

        @staticmethod
        def _pool(f, m):
            mf = m.unsqueeze(-1).to(f.dtype)
            mean = (f * mf).sum(1) / mf.sum(1).clamp_min(1.0)
            mx = f.masked_fill(~m.unsqueeze(-1), -1e4).max(1).values
            return torch.cat([mean, mx], -1)

        def forward(self, a, b, t):
            ha, ma = self.enc(a)
            hb, mb = self.enc(b)
            att = torch.bmm(ha, hb.transpose(1, 2)) / math.sqrt(self.d)
            wa = att.masked_fill(~mb.unsqueeze(1), -1e4).softmax(-1)
            wb = att.transpose(1, 2).masked_fill(~ma.unsqueeze(1), -1e4).softmax(-1)
            al_a = torch.bmm(wa, hb)
            al_b = torch.bmm(wb, ha)
            fa = torch.cat([ha - al_a, ha * al_a], -1)
            fb = torch.cat([hb - al_b, hb * al_b], -1)
            z = torch.cat([self._pool(fa, ma), self._pool(fb, mb), self.tab(t)], -1)
            return self.head(z).squeeze(-1)

    return CharMatcher()


def nn_oof(text_tr: TextIndex, s_tr, t_tr_text: TextIndex, t_tr, X, y, folds, ok,
           text_te: TextIndex, s_te, t_te_text: TextIndex, t_te, Xt,
           gpu: bool, n_jobs: int, seed: int = 42, budget_s: float = 1200.0,
           max_epochs: int = 4, batch: int = 512):
    """Fold-wise training -> (oof probs, test probs = mean of fold models)."""
    import torch
    import torch.nn.functional as F

    t_start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, n_jobs))
    dev = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"

    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0) + 1e-6

    def tab(Z):
        return np.clip(np.nan_to_num((Z - mu) / sd), -5, 5).astype(np.float32)

    A = torch.from_numpy(text_tr.rows(s_tr)).to(dev)
    B = torch.from_numpy(t_tr_text.rows(t_tr)).to(dev)
    T = torch.from_numpy(tab(X)).to(dev)
    Y = torch.from_numpy(y.astype(np.float32)).to(dev)

    def predict(model, a, b, t):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
            for i in range(0, len(t), 4096):
                out.append(torch.sigmoid(model(a[i:i + 4096], b[i:i + 4096], t[i:i + 4096]).float()))
        return torch.cat(out).cpu().numpy() if out else np.zeros(0, np.float32)

    oof = np.zeros(len(y), np.float32)
    models = []
    n_folds = len(folds)
    per_fold_budget = budget_s / (n_folds + 1)  # keep ~1 share for test inference
    for k, (tr, va) in enumerate(folds):
        tr = tr[ok[tr]]
        model = _build_model(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        steps_per_epoch = max(1, math.ceil(len(tr) / batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps_per_epoch * max_epochs,
                                                    pct_start=0.15)
        scaler = torch.amp.GradScaler(dev.type, enabled=amp)
        tr_t = torch.from_numpy(tr).to(dev)
        va_t = torch.from_numpy(va).to(dev)
        best, best_state, f0 = np.inf, None, time.time()
        for ep in range(max_epochs):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), device=dev)]
            for i in range(0, len(perm), batch):
                bi = perm[i:i + batch]
                with torch.autocast(dev.type, enabled=amp):
                    loss = F.binary_cross_entropy_with_logits(model(A[bi], B[bi], T[bi]).float(), Y[bi])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            pv = predict(model, A[va_t], B[va_t], T[va_t])
            yv = y[va]
            ll = float(-np.mean(yv * np.log(np.clip(pv, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - pv, 1e-6, 1))))
            if ll < best:
                best, best_state = ll, {kk: v.detach().clone() for kk, v in model.state_dict().items()}
                oof[va] = pv
            el = time.time() - f0
            log(f"    [charnn-{dev.type}] fold {k} epoch {ep + 1}: val logloss {ll:.5f} ({el:.0f}s)")
            if el / (ep + 1) * (ep + 2) > per_fold_budget:  # next epoch would exceed the budget
                break
        model.load_state_dict(best_state)
        models.append(model)
        if time.time() - t_start > budget_s * (k + 2) / (n_folds + 1) and k < n_folds - 1:
            raise TimeoutError("charnn over budget - member skipped")
    del A, B, T, Y
    # test: mean of the fold models (no refit -> bounded time), streamed in
    # chunks so millions of test pairs never sit in memory as tensors
    pt = np.zeros(len(s_te), np.float32)
    for i in range(0, len(s_te), 262144):
        sl = slice(i, i + 262144)
        At = torch.from_numpy(text_te.rows(s_te[sl])).to(dev)
        Bt = torch.from_numpy(t_te_text.rows(t_te[sl])).to(dev)
        Tt = torch.from_numpy(tab(Xt[sl])).to(dev)
        pt[sl] = np.mean([predict(m, At, Bt, Tt) for m in models], axis=0)
        del At, Bt, Tt
    del models
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    log(f"    [charnn] done in {time.time() - t_start:.0f}s")
    return oof, pt
