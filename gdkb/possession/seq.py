"""Sequence models over per-frame possession features (P5.seqmodel): bi-GRU, TCN, transformer encoder.

Everything here is pure PyTorch and vendor-agnostic; the feature tables come from ``features.py`` (24 base
columns) optionally extended with ``model.windowed`` (lags + rolling means, 145 columns = the P2/P3 GBM set).

* :class:`SeqConfig`        : architecture + training hyper-parameters (one dataclass, json-able, pickled with
                              the trained model so a pipeline can reproduce it).
* :func:`make_windows`      : (match, period) blocks -> fixed-length training windows with a random per-epoch
                              offset; frames within ±``boundary_pad`` of a label change get weight
                              ``boundary_weight`` (hand-over emphasis), everything else 1.
* :class:`SeqNet`           : ``kind`` in {gru, tcn, tfm}; input projection -> body -> 3-class head
                              (+ optional change-point head, a per-frame "label changes within ±3 frames" logit).
* :func:`train_seq`         : weighted cross-entropy (+ BCE on the change-point head), AdamW, early stopping
                              on a validation block's plain cross-entropy, returns the best state.
* :func:`predict_seq`       : overlapping inference windows (triangular weights) -> p_home/p_away/p_none
                              (+ p_change) per frame, as a KEY-indexed table; CPU or GPU.
* :func:`hmm_change_smooth` : Viterbi whose per-frame transition probability is driven by ``p_change``
                              (the "CRF/HMM decoding on top" of the change-point head).

Relative positions in the transformer: T5-style bucketed relative-position bias (one learned scalar per head
per bucket, buckets log-spaced beyond 16 frames, clipped at ±512 frames = ±20 s).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .features import FEATURE_COLS, KEY

CLASSES = ["home", "away", "none"]
_C2I = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class SeqConfig:
    kind: str = "gru"  # gru | tcn | tfm
    feats: str = "base"  # base (24 cols) | win (24 + lags/rolls = 145 cols)
    hidden: int = 128
    layers: int = 2  # gru layers / tfm layers; tcn uses len(dilations)
    heads: int = 4
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)  # tcn, kernel 7 -> receptive field 1 + 6*63 = 379 fr
    kernel: int = 7
    dropout: float = 0.1
    window: int = 500  # training window (frames) = 20 s
    stride: int = 250
    infer_window: int = 1500
    infer_stride: int = 750
    boundary_weight: float = 1.0  # 1 = no hand-over weighting
    boundary_pad: int = 25
    change_head: bool = False  # change-point head + BCE term
    change_lambda: float = 1.0
    change_pad: int = 3
    # PIPE2 (all default off = shipped model): spell-recall variants
    short_weight: float = 1.0  # loss weight of frames inside GT team spells shorter than short_len frames
    short_len: int = 50
    ms_conv: bool = False  # residual dilated-conv front-end (multi-scale local context) before the GRU/TCN/TFM body
    ms_dilations: tuple[int, ...] = (1, 2, 4, 8)
    ms_kernel: int = 5
    start_head: bool = False  # auxiliary spell-start head (BCE, change_pad frames before to change_pad-1 after a team spell start)
    batch: int = 32
    epochs: int = 40
    patience: int = 6
    lr: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 0
    cols: list[str] = field(default_factory=list)  # filled at fit time

    def name(self) -> str:
        return f"{self.kind}_{self.feats}_w{self.boundary_weight:g}_cp{int(self.change_head)}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dilations"] = list(self.dilations)
        d["ms_dilations"] = list(self.ms_dilations)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SeqConfig:
        d = dict(d)
        d["dilations"] = tuple(d.get("dilations", (1, 2, 4, 8, 16, 32)))
        d["ms_dilations"] = tuple(d.get("ms_dilations", (1, 2, 4, 8)))
        return cls(**d)


# ------------------------------------------------------------------------------------------------ data
def feature_table(feat: pd.DataFrame, feats: str) -> tuple[pd.DataFrame, list[str]]:
    """Sorted feature table + the column list for ``feats`` (``win`` adds model.windowed lags/rolls)."""
    if feats == "win":
        from .model import windowed

        wf = windowed(feat)
        cols = FEATURE_COLS + [c for c in wf.columns if "_lag" in c or "_roll" in c]
        return wf, cols
    return feat.sort_values(KEY, kind="stable").reset_index(drop=True), list(FEATURE_COLS)


def standardise(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Z = (X - mu) / sd
    return np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -10, 10).astype(np.float32)


def blocks_of(feat: pd.DataFrame) -> list[np.ndarray]:
    """Row indices of every (match_id, period) block, in frame order (``feat`` must be KEY-sorted)."""
    return [np.sort(v) for v in feat.groupby(["match_id", "period"], sort=False).indices.values()]


def labels_codes(feat: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    y = feat[KEY].merge(labels, on=KEY, how="left").possession.map(_C2I)
    return y.fillna(-1).to_numpy(int)


def boundary_mask(y: np.ndarray, blocks: list[np.ndarray], pad: int) -> np.ndarray:
    """True within a window around a label change (per block): frames c-pad .. c+pad-1 for the first frame c of the new run."""
    out = np.zeros(len(y), bool)
    for idx in blocks:
        yb = y[idx]
        ch = np.flatnonzero(yb[1:] != yb[:-1]) + 1  # first frame of the new run
        m = np.zeros(len(idx), bool)
        for c in ch:
            m[max(0, c - pad):c + pad] = True
        out[idx] = m
    return out


def short_spell_mask(y: np.ndarray, blocks: list[np.ndarray], max_len: int) -> np.ndarray:
    """True for frames inside a constant home/away run shorter than ``max_len`` frames (per block)."""
    out = np.zeros(len(y), bool)
    for idx in blocks:
        yb = y[idx]
        cut = np.flatnonzero(np.diff(yb) != 0) + 1
        starts = np.concatenate(([0], cut))
        stops = np.concatenate((cut, [len(yb)]))
        for a, b in zip(starts, stops, strict=True):
            if yb[a] in (0, 1) and b - a < max_len:
                out[idx[a:b]] = True
    return out


def start_mask(y: np.ndarray, blocks: list[np.ndarray], pad: int) -> np.ndarray:
    """True within a window around the first frame c of a home/away run that follows a different label (per block).

    The window is frames c-pad .. c+pad-1 (``pad`` before, ``pad - 1`` after; i.e. 3 before to 2 after for pad=3).
    A transition from an unlabeled (-1) frame into home/away counts as a spell start; the first frame of a block does not.
    """
    out = np.zeros(len(y), bool)
    for idx in blocks:
        yb = y[idx]
        ch = np.flatnonzero((yb[1:] != yb[:-1]) & np.isin(yb[1:], (0, 1))) + 1
        m = np.zeros(len(idx), bool)
        for c in ch:
            m[max(0, c - pad):c + pad] = True
        out[idx] = m
    return out


def make_windows(blocks: list[np.ndarray], window: int, stride: int, offset: int = 0,
                 min_len: int = 50) -> list[np.ndarray]:
    """Fixed-length index windows per block starting at ``offset`` plus one flush-right window."""
    out = []
    for idx in blocks:
        n = len(idx)
        if n < min_len:
            continue
        if n <= window:
            out.append(idx)
            continue
        starts = list(range(offset, n - window + 1, stride))
        if not starts or starts[-1] != n - window:
            starts.append(n - window)
        out += [idx[s:s + window] for s in starts]
    return out


# ------------------------------------------------------------------------------------------------ model
def _relative_buckets(n: int, num_buckets: int = 32, max_distance: int = 512):
    """T5 relative-position buckets for a length-n sequence: (n, n) int64 in [0, 2*num_buckets)."""
    import torch

    pos = torch.arange(n)
    rel = pos[None, :] - pos[:, None]  # key - query
    sign = (rel > 0).long() * num_buckets
    a = rel.abs()
    exact = num_buckets // 2
    is_small = a < exact
    large = exact + (torch.log(a.float().clamp(min=1) / exact) / math.log(max_distance / exact)
                     * (num_buckets - exact)).long()
    large = torch.minimum(large, torch.full_like(large, num_buckets - 1))
    return sign + torch.where(is_small, a, large)


def build_net(cfg: SeqConfig, n_in: int):
    import torch
    from torch import nn

    n_out = 3 + (1 if (cfg.change_head or cfg.start_head) else 0)
    H = cfg.hidden

    class TCNBlock(nn.Module):
        def __init__(self, d: int, k: int | None = None):
            super().__init__()
            k = k or cfg.kernel
            pad = (k - 1) // 2 * d
            self.conv = nn.Conv1d(H, H, k, dilation=d, padding=pad)
            self.pw = nn.Conv1d(H, H, 1)
            self.norm = nn.GroupNorm(1, H)
            self.drop = nn.Dropout(cfg.dropout)

        def forward(self, x):  # (B, H, T)
            h = self.drop(torch.nn.functional.gelu(self.conv(self.norm(x))))
            return x + self.pw(h)

    class RelAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(H, 3 * H)
            self.out = nn.Linear(H, H)
            self.bias = nn.Embedding(64, cfg.heads)
            self.drop = nn.Dropout(cfg.dropout)
            self._cache: dict[int, torch.Tensor] = {}

        def forward(self, x):  # (B, T, H)
            B, T, _ = x.shape
            h = cfg.heads
            q, k, v = self.qkv(x).view(B, T, 3, h, H // h).unbind(2)
            if T not in self._cache or self._cache[T].device != x.device:
                self._cache[T] = _relative_buckets(T).to(x.device)
            bias = self.bias(self._cache[T]).permute(2, 0, 1)[None]  # (1, h, T, T)
            a = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias,
                dropout_p=cfg.dropout if self.training else 0.0)
            return self.out(a.transpose(1, 2).reshape(B, T, H))

    class TfmLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(H), nn.LayerNorm(H)
            self.att = RelAttention()
            self.ff = nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(2 * H, H))
            self.drop = nn.Dropout(cfg.dropout)

        def forward(self, x):
            x = x + self.drop(self.att(self.n1(x)))
            return x + self.drop(self.ff(self.n2(x)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, H), nn.GELU(), nn.Dropout(cfg.dropout))
            if cfg.ms_conv:
                self.ms = nn.Sequential(*[TCNBlock(d, cfg.ms_kernel) for d in cfg.ms_dilations])
            if cfg.kind == "gru":
                self.body = nn.GRU(H, H, num_layers=cfg.layers, batch_first=True, bidirectional=True,
                                   dropout=cfg.dropout if cfg.layers > 1 else 0.0)
                d_body = 2 * H
            elif cfg.kind == "tcn":
                self.body = nn.Sequential(*[TCNBlock(d) for d in cfg.dilations])
                d_body = H
            elif cfg.kind == "tfm":
                self.body = nn.Sequential(*[TfmLayer() for _ in range(cfg.layers)], nn.LayerNorm(H))
                d_body = H
            else:
                raise ValueError(cfg.kind)
            self.head = nn.Linear(d_body, n_out)

        def forward(self, x):  # (B, T, n_in) -> (B, T, n_out)
            h = self.inp(x)
            if cfg.ms_conv:
                h = self.ms(h.transpose(1, 2)).transpose(1, 2)
            if cfg.kind == "gru":
                h = self.body(h)[0]
            elif cfg.kind == "tcn":
                h = self.body(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.body(h)
            return self.head(h)

    return Net()


# ---------------------------------------------------------------------------------------------- training
@dataclass
class SeqModel:
    cfg: SeqConfig
    state: dict  # torch state_dict (CPU tensors)
    mu: np.ndarray
    sd: np.ndarray
    info: dict = field(default_factory=dict)

    def net(self, device: str = "cpu"):

        net = build_net(self.cfg, len(self.cfg.cols))
        net.load_state_dict(self.state)
        return net.to(device).eval()

    def to_pickle(self) -> dict:
        return {"cfg": self.cfg.to_dict(), "state": {k: v.cpu().numpy() for k, v in self.state.items()},
                "mu": self.mu, "sd": self.sd, "info": self.info}

    @classmethod
    def from_pickle(cls, d: dict) -> SeqModel:
        import torch

        return cls(SeqConfig.from_dict(d["cfg"]), {k: torch.from_numpy(np.asarray(v)) for k, v in d["state"].items()},
                   np.asarray(d["mu"]), np.asarray(d["sd"]), d.get("info", {}))

    # pickle-free export (P6.robust): weights + standardisation as a plain ``.npz``, config as json
    def to_npz(self, path) -> dict:
        """Write ``path`` (npz: one array per state_dict entry under ``state.<name>``, plus ``mu`` / ``sd``);
        returns the json-able config to store next to it (``SeqConfig`` + ``info`` without the history)."""
        arrays = {f"state.{k}": v.cpu().numpy() for k, v in self.state.items()}
        np.savez(path, mu=np.asarray(self.mu, np.float64), sd=np.asarray(self.sd, np.float64), **arrays)
        return {"cfg": self.cfg.to_dict(), "info": {k: v for k, v in self.info.items() if k != "history"}}

    @classmethod
    def from_npz(cls, path, spec: dict) -> SeqModel:
        import torch

        with np.load(path) as z:
            state = {k[len("state."):]: torch.from_numpy(np.array(z[k])) for k in z.files if k.startswith("state.")}
            mu, sd = np.array(z["mu"]), np.array(z["sd"])
        return cls(SeqConfig.from_dict(spec["cfg"]), state, mu, sd, dict(spec.get("info", {})))


def _loss_terms(out, yb, wb, cb, cfg: SeqConfig):
    import torch
    from torch.nn import functional as F

    logits = out[..., :3]
    ce = F.cross_entropy(logits.reshape(-1, 3), yb.reshape(-1), reduction="none", ignore_index=-1)
    valid = (yb.reshape(-1) >= 0).float()
    ce = (ce * wb.reshape(-1) * valid).sum() / (wb.reshape(-1) * valid).sum().clamp(min=1.0)
    bce = torch.zeros((), device=out.device)
    if cfg.change_head or cfg.start_head:
        pos_w = torch.tensor(10.0, device=out.device)
        bce = F.binary_cross_entropy_with_logits(out[..., 3].reshape(-1), cb.reshape(-1), pos_weight=pos_w,
                                                 reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)
    return ce, bce


def train_seq(cfg: SeqConfig, train_feat: pd.DataFrame, train_labels: pd.DataFrame,
              val_feat: pd.DataFrame | None, val_labels: pd.DataFrame | None, device: str = "cuda",
              log=print) -> SeqModel:
    """Fit on ``train_feat`` (any number of matches/periods); early-stop on ``val_feat``'s cross-entropy."""
    import torch

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    t0 = time.time()
    tf, cols = feature_table(train_feat, cfg.feats)
    cfg.cols = cols
    M = tf[cols].to_numpy(np.float32)
    mu, sd = np.nanmean(M, 0), np.nanstd(M, 0) + 1e-6
    X = standardise(M, mu, sd)
    del M
    y = labels_codes(tf, train_labels)
    blocks = blocks_of(tf)
    w = np.ones(len(y), np.float32)
    if cfg.boundary_weight != 1.0:
        w[boundary_mask(y, blocks, cfg.boundary_pad)] = cfg.boundary_weight
    if cfg.short_weight != 1.0:
        w[short_spell_mask(y, blocks, cfg.short_len)] *= cfg.short_weight
    c = (start_mask(y, blocks, cfg.change_pad) if cfg.start_head
         else boundary_mask(y, blocks, cfg.change_pad)).astype(np.float32)
    Xv = yv = None
    if val_feat is not None:
        vf, _ = feature_table(val_feat, cfg.feats)
        Xv = standardise(vf[cols].to_numpy(np.float32), mu, sd)
        yv = labels_codes(vf, val_labels)
        val_windows = make_windows(blocks_of(vf), cfg.infer_window, cfg.infer_window)
    net = build_net(cfg, len(cols)).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    rng = np.random.default_rng(cfg.seed)
    best, best_state, best_ep, bad = math.inf, None, -1, 0
    hist = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    Xt = torch.from_numpy(X)
    yt, wt, ct = torch.from_numpy(y), torch.from_numpy(w), torch.from_numpy(c)
    for ep in range(cfg.epochs):
        net.train()
        wins = make_windows(blocks, cfg.window, cfg.stride, offset=int(rng.integers(0, cfg.stride)))
        rng.shuffle(wins)
        tot_ce = tot_bce = 0.0
        nb = 0
        for i in range(0, len(wins), cfg.batch):
            batch = [b for b in wins[i:i + cfg.batch] if len(b) == cfg.window] or wins[i:i + 1]
            idx = torch.from_numpy(np.stack(batch))
            xb, yb, wb, cb = (Xt[idx].to(device), yt[idx].to(device), wt[idx].to(device), ct[idx].to(device))
            out = net(xb)
            ce, bce = _loss_terms(out, yb, wb, cb, cfg)
            loss = ce + cfg.change_lambda * bce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot_ce += float(ce)
            tot_bce += float(bce)
            nb += 1
        rec = {"epoch": ep, "train_ce": tot_ce / max(nb, 1), "train_bce": tot_bce / max(nb, 1),
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0)}
        if Xv is not None:
            net.eval()
            vs, vn = 0.0, 0
            with torch.no_grad():
                for vw in val_windows:
                    xb = torch.from_numpy(Xv[vw])[None].to(device)
                    yb = torch.from_numpy(yv[vw])[None].to(device)
                    out = net(xb)
                    ce = torch.nn.functional.cross_entropy(out[..., :3].reshape(-1, 3), yb.reshape(-1),
                                                           reduction="sum", ignore_index=-1)
                    vs += float(ce)
                    vn += int((yb >= 0).sum())
            rec["val_ce"] = vs / max(vn, 1)
            sched.step(rec["val_ce"])
            if rec["val_ce"] < best - 1e-4:
                best, best_ep, bad = rec["val_ce"], ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
        hist.append(rec)
        log(json.dumps(rec))
        if Xv is not None and bad >= cfg.patience:
            break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        best_ep = len(hist) - 1
    info = {"params": n_par, "best_epoch": best_ep, "epochs_run": len(hist), "best_val_ce": best,
            "train_seconds": round(time.time() - t0), "history": hist, "n_train_frames": int(len(y)),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else None}
    return SeqModel(cfg, best_state, mu, sd, info)


def predict_seq(model: SeqModel, feat: pd.DataFrame, device: str = "cpu", batch: int = 8) -> pd.DataFrame:
    """Per-frame probabilities (overlapping inference windows, triangular blend) as a KEY table."""
    import torch

    cfg = model.cfg
    tf, _ = feature_table(feat, cfg.feats)
    X = standardise(tf[cfg.cols].to_numpy(np.float32), model.mu, model.sd)
    net = model.net(device)
    n_out = 3 + (1 if cfg.change_head else 0)  # the auxiliary start head (if any) is not read at inference
    acc = np.zeros((len(X), n_out), np.float64)
    wsum = np.zeros(len(X), np.float64)
    wins = make_windows(blocks_of(tf), cfg.infer_window, cfg.infer_stride, min_len=1)
    with torch.no_grad():
        for i in range(0, len(wins), batch):
            group = wins[i:i + batch]
            for L in sorted({len(g) for g in group}):
                sub = [g for g in group if len(g) == L]
                xb = torch.from_numpy(X[np.stack(sub)]).to(device)
                out = net(xb)
                P = torch.softmax(out[..., :3], -1)
                if cfg.change_head:
                    P = torch.cat([P, torch.sigmoid(out[..., 3:4])], -1)
                P = P.cpu().numpy()
                tri = np.minimum(np.arange(1, L + 1), np.arange(L, 0, -1)).astype(np.float64)
                tri = tri / tri.max()
                for g, p in zip(sub, P, strict=True):
                    acc[g] += p * tri[:, None]
                    wsum[g] += tri
    P = acc / np.maximum(wsum, 1e-9)[:, None]
    out = tf[KEY].copy()
    out["possession"] = np.array(CLASSES, dtype=object)[P[:, :3].argmax(1)]
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = P[:, i].astype(np.float32)
    if cfg.change_head:
        out["p_change"] = P[:, 3].astype(np.float32)
    return out


def predict_seq_ensemble(models: list[SeqModel], feat: pd.DataFrame, device: str = "cpu", batch: int = 8) -> pd.DataFrame:
    """Seed ensemble (PIPE6): the arithmetic mean of the class probabilities of N models (same features, same windows) as a KEY
    table, before the unchanged decode. One model gives exactly ``predict_seq``; N models cost N forward passes."""
    if not models:
        raise ValueError("no sequence model")
    outs = [predict_seq(m, feat, device=device, batch=batch) for m in models]
    base = outs[0]
    if len(outs) == 1:
        return base
    for o in outs[1:]:
        if len(o) != len(base) or not (o[KEY].to_numpy() == base[KEY].to_numpy()).all():
            raise ValueError("ensemble members disagree on the frame keys (different feature sets?)")
    cols = [f"p_{c}" for c in CLASSES] + (["p_change"] if "p_change" in base else [])
    out = base.copy()
    for c in cols:
        out[c] = np.mean([o[c].to_numpy(np.float64) for o in outs], axis=0).astype(np.float32)
    out["possession"] = np.array(CLASSES, dtype=object)[out[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1)]
    return out


# ---------------------------------------------------------------------------------------------- decoding
def hmm_change_smooth(pred: pd.DataFrame, stay: float = 0.995, gain: float = 1.0, floor: float = 1e-3) -> pd.DataFrame:
    """Viterbi with a per-frame switch probability q_t = clip((1-stay) + gain * p_change_t, 0, 0.5).

    ``gain = 0`` reduces to model.hmm_smooth. With the change-point head, transitions become cheap exactly
    where the head sees a hand-over and stay sticky elsewhere.
    """
    pred = pred.sort_values(KEY, kind="stable").reset_index(drop=True)
    out = pred.copy()
    pc = pred["p_change"].to_numpy(float) if "p_change" in pred else np.zeros(len(pred))
    for _, g in pred.groupby(["match_id", "period"], sort=False):
        E = np.log(np.clip(g[[f"p_{c}" for c in CLASSES]].to_numpy(float), floor, 1.0))
        q = np.clip((1 - stay) + gain * pc[g.index.to_numpy()], 1e-6, 0.5)
        n = len(E)
        delta = np.zeros((n, 3))
        back = np.zeros((n, 3), dtype=np.int8)
        delta[0] = np.log(1 / 3) + E[0]
        for t in range(1, n):
            logA = np.full((3, 3), np.log(q[t] / 2))
            np.fill_diagonal(logA, np.log(1 - q[t]))
            cand = delta[t - 1][:, None] + logA
            back[t] = cand.argmax(0)
            delta[t] = cand.max(0) + E[t]
        path = np.zeros(n, dtype=int)
        path[-1] = delta[-1].argmax()
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        out.loc[g.index, "possession"] = np.array(CLASSES, dtype=object)[path]
    return out


__all__ = ["SeqConfig", "SeqModel", "train_seq", "predict_seq", "hmm_change_smooth", "make_windows",
           "boundary_mask", "feature_table", "standardise", "CLASSES"]
