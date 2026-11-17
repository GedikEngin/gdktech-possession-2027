"""Learned possession models over per-frame features (features.py), plus smoothing.

* `windowed(feat)`     : adds lagged/lead values and rolling means of the key features (offline evaluation ->
                         look-ahead is allowed; a leaderboard scores whole matches).
* `GBM`                : LightGBM multiclass on the windowed features -> p_home/p_away/p_none.
* `GRU`                : small bidirectional GRU (PyTorch, CPU) on standardised per-frame features, trained on
                         fixed-length chunks. Optional: only if torch is installed.
* smoothing            : `mode_filter` (sliding-window majority on labels) and `hmm_smooth` (Viterbi with a
                         sticky transition matrix over the model's class probabilities).
All functions take/return tables keyed by (match_id, period, frame) (docs/frame-format.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLS, KEY

CLASSES = ["home", "away", "none"]
_C2I = {c: i for i, c in enumerate(CLASSES)}
WINDOW_BASE = ["d_home", "d_away", "d_diff", "ball_speed", "ball_missing", "contact_home", "contact_away",
               "nearest_team", "ball_out_of_pitch", "n_home_r2", "n_away_r2"]


def windowed(feat: pd.DataFrame, lags: tuple[int, ...] = (-50, -25, -10, -3, 3, 10, 25, 50),
             rolls: tuple[int, ...] = (13, 51, 251)) -> pd.DataFrame:
    """Append shifted (negative = past) and centred-rolling-mean versions of WINDOW_BASE within each period."""
    feat = feat.sort_values(KEY, kind="stable").reset_index(drop=True)
    g = feat.groupby(["match_id", "period"], sort=False)[WINDOW_BASE]
    new = {}
    for lag in lags:
        sh = g.shift(lag)
        for c in WINDOW_BASE:
            new[f"{c}_lag{lag}"] = sh[c].to_numpy()
    for w in rolls:
        rm = g.rolling(w, center=True, min_periods=1).mean().reset_index(level=[0, 1], drop=True).sort_index()
        for c in WINDOW_BASE:
            new[f"{c}_roll{w}"] = rm[c].to_numpy()
    return pd.concat([feat, pd.DataFrame(new, index=feat.index)], axis=1)


def feature_matrix(wf: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = FEATURE_COLS + [c for c in wf.columns if "_lag" in c or "_roll" in c]
    return wf[cols].to_numpy(np.float32), cols


def _probs_table(wf: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    out = wf[KEY].copy()
    out["possession"] = np.array(CLASSES, dtype=object)[P.argmax(1)]
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = P[:, i]
    return out


class GBM:
    def __init__(self, **params):
        self.params = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 31,
                       "min_data_in_leaf": 200, "feature_fraction": 0.7, "bagging_fraction": 0.7,
                       "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": 2, "verbose": -1, "seed": 0}
        self.params.update(params)
        self.model, self.cols = None, None

    def fit(self, wf: pd.DataFrame, labels: pd.DataFrame, num_rounds: int = 300,
            weight: np.ndarray | None = None) -> GBM:
        """``weight``: optional per-row sample weights aligned with ``wf`` (P4: up-weighting pseudo-tracking rows)."""
        import lightgbm as lgb

        X, self.cols = feature_matrix(wf)
        y = wf[KEY].merge(labels, on=KEY, how="left").possession.map(_C2I).to_numpy()
        ok = ~np.isnan(y.astype(float))
        w = None if weight is None else np.asarray(weight, dtype=np.float32)[ok]
        self.model = lgb.train(self.params, lgb.Dataset(X[ok], y[ok].astype(int), weight=w), num_boost_round=num_rounds)
        return self

    def predict(self, wf: pd.DataFrame) -> pd.DataFrame:
        return _probs_table(wf, self.model.predict(wf[self.cols].to_numpy(np.float32)))

    def importance(self, k: int = 15) -> list[tuple[str, int]]:
        imp = self.model.feature_importance("gain")
        return sorted(zip(self.cols, imp.astype(int)), key=lambda t: -t[1])[:k]


# ----------------------------------------------------------------------------------------------- smoothing
def mode_filter(pred: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """Sliding-window majority vote on the label column, per (match, period)."""
    pred = pred.sort_values(KEY, kind="stable").reset_index(drop=True)
    out = pred.copy()
    for _, g in pred.groupby(["match_id", "period"], sort=False):
        oh = np.stack([(g.possession == c).to_numpy(float) for c in CLASSES], 1)
        sm = pd.DataFrame(oh).rolling(window, center=True, min_periods=1).sum().to_numpy()
        out.loc[g.index, "possession"] = np.array(CLASSES, dtype=object)[sm.argmax(1)]
    return out


def hmm_smooth(pred: pd.DataFrame, stay: float = 0.995, floor: float = 1e-3) -> pd.DataFrame:
    """Viterbi decoding with a symmetric sticky transition matrix over the p_* columns (treated as emissions)."""
    pred = pred.sort_values(KEY, kind="stable").reset_index(drop=True)
    out = pred.copy()
    A = np.full((3, 3), (1 - stay) / 2)
    np.fill_diagonal(A, stay)
    logA = np.log(A)
    for _, g in pred.groupby(["match_id", "period"], sort=False):
        E = np.log(np.clip(g[[f"p_{c}" for c in CLASSES]].to_numpy(float), floor, 1.0))
        n = len(E)
        delta = np.zeros((n, 3))
        back = np.zeros((n, 3), dtype=np.int8)
        delta[0] = np.log(1 / 3) + E[0]
        for t in range(1, n):
            cand = delta[t - 1][:, None] + logA  # (from, to)
            back[t] = cand.argmax(0)
            delta[t] = cand.max(0) + E[t]
        path = np.zeros(n, dtype=int)
        path[-1] = delta[-1].argmax()
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        out.loc[g.index, "possession"] = np.array(CLASSES, dtype=object)[path]
    return out


# ----------------------------------------------------------------------------------------------- GRU
class GRU:
    """Bidirectional GRU over standardised per-frame features; CPU only, chunked training."""

    def __init__(self, hidden: int = 32, chunk: int = 1500, epochs: int = 8, lr: float = 3e-3, seed: int = 0):
        self.hidden, self.chunk, self.epochs, self.lr, self.seed = hidden, chunk, epochs, lr, seed
        self.net, self.mu, self.sd, self.cols = None, None, None, None

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _xy(self, feat: pd.DataFrame, labels: pd.DataFrame | None):
        feat = feat.sort_values(KEY, kind="stable").reset_index(drop=True)
        X = np.nan_to_num(((feat[self.cols].to_numpy(np.float32) - self.mu) / self.sd), nan=0.0)
        y = None
        if labels is not None:
            y = feat[KEY].merge(labels, on=KEY, how="left").possession.map(_C2I).fillna(2).to_numpy(int)
        bounds = feat.groupby(["match_id", "period"], sort=False).indices
        return feat, X, y, [np.sort(v) for v in bounds.values()]

    def fit(self, feat: pd.DataFrame, labels: pd.DataFrame) -> GRU:
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        self.cols = FEATURE_COLS
        M = feat[self.cols].to_numpy(np.float32)
        self.mu, self.sd = np.nanmean(M, 0), np.nanstd(M, 0) + 1e-6
        _, X, y, blocks = self._xy(feat, labels)
        chunks = [idx[i:i + self.chunk] for idx in blocks for i in range(0, len(idx), self.chunk)]
        chunks = [c for c in chunks if len(c) >= 50]
        self.net = nn.ModuleDict({"gru": nn.GRU(X.shape[1], self.hidden, batch_first=True, bidirectional=True),
                                  "head": nn.Linear(2 * self.hidden, 3)})
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.CrossEntropyLoss()
        rng = np.random.default_rng(self.seed)
        for ep in range(self.epochs):
            rng.shuffle(chunks)
            tot = 0.0
            for i in range(0, len(chunks), 8):
                batch = [c for c in chunks[i:i + 8] if len(c) == self.chunk] or chunks[i:i + 1]
                xb = torch.tensor(np.stack([X[c] for c in batch]))
                yb = torch.tensor(np.stack([y[c] for c in batch]))
                out = self.net["head"](self.net["gru"](xb)[0])
                loss = lossf(out.reshape(-1, 3), yb.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
        return self

    def predict(self, feat: pd.DataFrame) -> pd.DataFrame:
        import torch

        feat, X, _, blocks = self._xy(feat, None)
        P = np.zeros((len(X), 3), np.float32)
        with torch.no_grad():
            for idx in blocks:
                out = self.net["head"](self.net["gru"](torch.tensor(X[idx])[None])[0])[0]
                P[idx] = torch.softmax(out, -1).numpy()
        return _probs_table(feat, P)
