"""Segment-then-refine post-processing for the possession segmenter (P3.possession).

Everything here works on a *predictions* table (``match_id, period, frame, possession`` with ``possession in
{home, away, none}``) aligned frame-for-frame with the per-frame *features* table of
:mod:`gdkb.possession.features`; all tables are keyed by ``KEY`` and processed per ``(match_id, period)`` block.

* :func:`refine_boundaries` — around every predicted hand-over into a team (home<->away, and optionally
  none->team) the boundary is re-estimated inside ``±window`` frames from ball-centric cues and shifted:
  ``control_onset`` = the first frame of a run where the receiving team's nearest player is within
  ``control_r`` of the (visible) ball and nearer than the other team; ``speed_min`` = the frame of minimum
  ball speed; strategies pick the nearest / first onset, the speed minimum, or an onset with the speed
  minimum as fallback. Shifts never cross the neighbouring boundaries (``min_keep`` frames stay on each side).
* :class:`DeadBallRules` — rule dead-ball (D) detector: ball out of the pitch, ball static with nobody near it
  for ``static_s``, long ball-missing stretches that end in a restart, Smart-Tagging-style set-piece windows
  (D from the window start until the restart moment).
* :class:`DeadBallModel` — binary LightGBM on :func:`dead_ball_features` (the rule cues as features, with
  run-length context), decoded with a threshold and a minimum run length.
* :func:`merge_dead` — combine the segmenter's labels with a D flag (``union`` / ``replace`` / ``segmenter``),
  carrying the last team label where a D run is removed (never ``X``).
* :func:`fragmentation_control` — merge team spells shorter than ``min_spell_frames`` into their neighbour;
  the HMM stickiness and this minimum are tuned on training folds in scripts/p3_postprocess.py.
* :func:`simulate_set_pieces` — Smart-Tagging-like set-piece windows from a labels table (every dead->team
  restart becomes a ~5 s window opening ~3.5 s early with jitter; type from the ball position; throw-ins
  carry no team), for training/evaluating on our converted matches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .features import KEY

CLASSES = ["home", "away", "none"]
_C2I = {c: i for i, c in enumerate(CLASSES)}
HOME, AWAY, NONE = 0, 1, 2


# ----------------------------------------------------------------------------------------------- helpers
def runs(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run-length encoding: (starts, ends, values) with half-open [start, end)."""
    codes = np.asarray(codes)
    n = len(codes)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, codes.dtype)
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]
    return starts, ends, codes[starts]


def codes_of(pred: pd.DataFrame) -> np.ndarray:
    return pred["possession"].map(_C2I).to_numpy(np.int8)


def labels_of(codes: np.ndarray) -> np.ndarray:
    return np.array(CLASSES, dtype=object)[codes]


def _aligned(pred: pd.DataFrame, feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort both tables by KEY and check they cover the same frames."""
    p = pred.sort_values(KEY, kind="stable").reset_index(drop=True)
    f = feat.sort_values(KEY, kind="stable").reset_index(drop=True)
    if len(p) != len(f) or not (p["frame"].to_numpy() == f["frame"].to_numpy()).all() or not (
            p["period"].to_numpy() == f["period"].to_numpy()).all():
        f = p[KEY].merge(f, on=KEY, how="left")
        if f["ball_missing"].isna().any():
            raise ValueError("features do not cover every predicted frame")
    return p, f


def _blocks(df: pd.DataFrame) -> list[np.ndarray]:
    return [np.sort(idx) for _, idx in df.groupby(["match_id", "period"], sort=False).indices.items()]


def _true_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) of the True runs of a boolean array, half-open."""
    m = np.asarray(mask, dtype=bool)
    if len(m) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _run_length_so_far(mask: np.ndarray) -> np.ndarray:
    """Frames since the start of the current True run (0 outside runs, 1 on the first frame)."""
    m = np.asarray(mask, dtype=bool)
    out = np.zeros(len(m), dtype=np.int32)
    s, e = _true_runs(m)
    for a, b in zip(s, e, strict=True):
        out[a:b] = np.arange(1, b - a + 1)
    return out


def _run_length_total(mask: np.ndarray) -> np.ndarray:
    """Total length of the True run each frame belongs to (0 outside runs); offline look-ahead."""
    m = np.asarray(mask, dtype=bool)
    out = np.zeros(len(m), dtype=np.int32)
    s, e = _true_runs(m)
    for a, b in zip(s, e, strict=True):
        out[a:b] = b - a
    return out


# --------------------------------------------------------------------------------- boundary re-estimation
@dataclass
class RefineConfig:
    window: int = 25  # frames either side of the predicted boundary
    control_r: float = 2.0  # m: receiving team's nearest player this close = control
    strategy: str = "nearest_onset"  # nearest_onset | first_onset | speed_min | onset_then_speed | none
    min_keep: int = 3  # frames that must remain on each side of a shifted boundary
    refine_restarts: bool = False  # also shift none->team boundaries
    max_shift: int | None = None  # cap |shift| (None = window)

    def to_dict(self) -> dict:
        return asdict(self)


def _control_onsets(d_recv: np.ndarray, d_other: np.ndarray, missing: np.ndarray, r: float) -> np.ndarray:
    """Boolean: first frame of each run where the receiving team controls the visible ball."""
    ctrl = (d_recv <= r) & (d_recv < d_other) & ~missing
    ctrl = np.nan_to_num(ctrl.astype(float), nan=0).astype(bool)
    onset = ctrl.copy()
    onset[1:] &= ~ctrl[:-1]
    return onset


def refine_block(codes: np.ndarray, d_home: np.ndarray, d_away: np.ndarray, speed: np.ndarray,
                 missing: np.ndarray, cfg: RefineConfig) -> tuple[np.ndarray, list[int]]:
    """Refine one block; returns the new codes and the list of applied shifts (pred_new - pred_old)."""
    out = codes.copy()
    if cfg.strategy == "none":
        return out, []
    starts, _, vals = runs(codes)
    bounds = starts[1:]  # boundary b: codes[b] != codes[b-1]
    shifts: list[int] = []
    max_shift = cfg.window if cfg.max_shift is None else cfg.max_shift
    # control onsets on the whole block, so a run that started before a window is not an onset at its edge
    onsets = {HOME: _control_onsets(d_home, d_away, missing, cfg.control_r),
              AWAY: _control_onsets(d_away, d_home, missing, cfg.control_r)}
    # process boundaries left to right on the *current* array so shifted neighbours are respected
    for k, b in enumerate(bounds):
        new, old = int(vals[k + 1]), int(vals[k])
        if new == NONE:
            continue
        if old == NONE and not cfg.refine_restarts:
            continue
        prev_start = int(starts[k])
        next_start = int(starts[k + 2]) if k + 2 < len(starts) else len(codes)
        lo = max(b - cfg.window, prev_start + cfg.min_keep)
        hi = min(b + cfg.window, next_start - cfg.min_keep)  # inclusive candidate range [lo, hi]
        if hi < lo:
            continue
        sl = slice(lo, hi + 1)
        t = None
        if cfg.strategy in ("nearest_onset", "first_onset", "onset_then_speed"):
            cand = np.flatnonzero(onsets[new][sl]) + lo
            if len(cand):
                t = int(cand[0]) if cfg.strategy == "first_onset" else int(cand[np.argmin(np.abs(cand - b))])
        if t is None and cfg.strategy in ("speed_min", "onset_then_speed"):
            sp = np.where(missing[sl], np.inf, speed[sl])
            if np.isfinite(sp).any():
                t = int(np.argmin(sp)) + lo
        if t is None or t == b or abs(t - b) > max_shift:
            continue
        # apply: frames between the old and the new boundary take the label on the other side
        if t < b:
            out[t:b] = new
        else:
            out[b:t] = old
        shifts.append(t - b)
    return out, shifts


def refine_boundaries(pred: pd.DataFrame, feat: pd.DataFrame, cfg: RefineConfig | None = None,
                      return_shifts: bool = False):
    """Re-estimate every predicted hand-over from ball cues (see :class:`RefineConfig`)."""
    cfg = cfg or RefineConfig()
    p, f = _aligned(pred, feat)
    codes = codes_of(p)
    d_home, d_away = f["d_home"].to_numpy(float), f["d_away"].to_numpy(float)
    speed, missing = f["ball_speed"].to_numpy(float), f["ball_missing"].to_numpy() == 1
    out = codes.copy()
    all_shifts: list[int] = []
    for idx in _blocks(p):
        new, sh = refine_block(codes[idx], d_home[idx], d_away[idx], speed[idx], missing[idx], cfg)
        out[idx] = new
        all_shifts.extend(sh)
    res = p[KEY].copy()
    res["possession"] = labels_of(out)
    return (res, all_shifts) if return_shifts else res


# ------------------------------------------------------------------------------------------- dead ball
@dataclass
class DeadBallRules:
    """Rule D detector. Every component can be switched off; the output is a per-frame boolean."""

    fps: float = 25.0
    use_out_of_pitch: bool = True
    out_margin_frames: int = 3  # frames out of pitch before the rule fires
    use_static: bool = True
    static_speed: float = 0.5  # m/s
    static_s: float = 2.0  # seconds the ball must be static with nobody within ``alone_r``
    alone_r: float = 1.5  # m
    use_missing: bool = True
    missing_s: float = 8.0  # a ball-missing stretch at least this long ...
    missing_restart_s: float = 4.0  # ... that ends within this many seconds before a restart cue is dead
    use_set_pieces: bool = True
    sp_default_restart_s: float = 3.5  # restart moment inside a window when no ball cue is found
    sp_restart_speed: float = 1.5  # ball speed (m/s) that marks the restart inside a window
    sp_restart_r: float = 2.5  # ... with a player this close
    sp_back_s: float = 2.0  # extend D backwards from a window start while no team contact is seen, up to this
    restart_cue: str = "contact"  # how a cue-driven D run ends: first team contact after the run

    def to_dict(self) -> dict:
        return asdict(self)

    def detect_block(self, f: pd.DataFrame, sp: pd.DataFrame | None) -> np.ndarray:
        n = len(f)
        missing = f["ball_missing"].to_numpy() == 1
        out_pitch = (f["ball_out_of_pitch"].to_numpy() == 1) & ~missing
        speed = np.nan_to_num(f["ball_speed"].to_numpy(float), nan=0.0)
        dmin = np.fmin(f["d_home"].to_numpy(float), f["d_away"].to_numpy(float))
        dmin = np.where(np.isnan(dmin), np.inf, dmin)
        fr = f["frame"].to_numpy(np.int64)
        contact = (dmin <= self.alone_r) & ~missing
        dead = np.zeros(n, dtype=bool)
        if self.use_out_of_pitch:
            dead |= _run_length_so_far(out_pitch) >= self.out_margin_frames
        if self.use_static:
            static = (speed < self.static_speed) & (dmin > self.alone_r) & ~missing
            dead |= _run_length_total(static) >= int(self.static_s * self.fps)
        windows: list[tuple[int, int]] = []
        if self.use_set_pieces and sp is not None and len(sp):
            s0 = np.searchsorted(fr, sp["start_frame"].to_numpy(np.int64))
            e0 = np.searchsorted(fr, sp["end_frame"].to_numpy(np.int64))
            windows = [(int(a), int(min(b, n))) for a, b in zip(s0, e0, strict=True) if a < n]
        if self.use_missing:
            ms, me = _true_runs(missing)
            min_len = int(self.missing_s * self.fps)
            horizon = int(self.missing_restart_s * self.fps)
            restart_cue = out_pitch.copy()
            for a, _b in windows:
                restart_cue[a:min(n, a + int(self.sp_default_restart_s * self.fps) + 1)] = True
            for a, b in zip(ms, me, strict=True):
                if b - a >= min_len and restart_cue[b:min(n, b + horizon)].any():
                    dead[a:b] = True
        # a cue-driven D run lasts until the first team contact after it (the restart)
        if self.restart_cue == "contact":
            ds, de = _true_runs(dead)
            for b in de:
                nxt = np.flatnonzero(contact[b:])
                dead[b:(b + int(nxt[0])) if len(nxt) else n] = True
        for a, b in windows:
            # restart moment: first frame in the window where the visible ball moves with a player at it
            moving = (speed[a:b] >= self.sp_restart_speed) & (dmin[a:b] <= self.sp_restart_r) & ~missing[a:b]
            hit = np.flatnonzero(moving)
            r = a + int(hit[0]) if len(hit) else min(n, a + int(self.sp_default_restart_s * self.fps))
            dead[a:r] = True
            # extend backwards while nobody touches the ball
            back = a
            lim = max(0, a - int(self.sp_back_s * self.fps))
            while back > lim and not contact[back - 1]:
                back -= 1
            dead[back:a] = True
        return dead

    def detect(self, feat: pd.DataFrame, set_pieces: pd.DataFrame | None = None) -> pd.DataFrame:
        """Per-frame ``dead`` flag table (KEY + dead) for a features table and optional set-piece windows
        (``match_id, period, start_frame, end_frame`` in the features' frame units, half-open)."""
        f = feat.sort_values(KEY, kind="stable").reset_index(drop=True)
        dead = np.zeros(len(f), dtype=bool)
        for (m, per), idx in f.groupby(["match_id", "period"], sort=False).indices.items():
            idx = np.sort(idx)
            sp = None
            if set_pieces is not None and len(set_pieces):
                sp = set_pieces[(set_pieces["match_id"] == m) & (set_pieces["period"] == per)]
            dead[idx] = self.detect_block(f.iloc[idx], sp)
        out = f[KEY].copy()
        out["dead"] = dead
        return out


DEAD_FEATURE_COLS = ["ball_missing", "ball_out_of_pitch", "ball_speed", "d_min", "n_r2", "n_r5", "miss_len_sofar",
                     "miss_len_total", "out_len_sofar", "static_len_sofar", "static_len_total", "since_contact",
                     "until_contact", "in_sp", "sp_since_start", "sp_until_end", "sp_since_end", "sp_until_start",
                     "cx_gap", "ball_vs_cx_mean"]


def dead_ball_features(feat: pd.DataFrame, set_pieces: pd.DataFrame | None = None, fps: float = 25.0,
                       alone_r: float = 1.5, static_speed: float = 0.5) -> pd.DataFrame:
    """Per-frame features for the learned D detector (KEY + DEAD_FEATURE_COLS); offline (uses look-ahead)."""
    f = feat.sort_values(KEY, kind="stable").reset_index(drop=True)
    n = len(f)
    cols = {c: np.zeros(n, dtype=np.float32) for c in DEAD_FEATURE_COLS}
    for (m, per), idx in f.groupby(["match_id", "period"], sort=False).indices.items():
        idx = np.sort(idx)
        g = f.iloc[idx]
        k = len(idx)
        missing = g["ball_missing"].to_numpy() == 1
        out_pitch = (g["ball_out_of_pitch"].to_numpy() == 1) & ~missing
        speed = np.nan_to_num(g["ball_speed"].to_numpy(float), nan=0.0)
        dmin = np.fmin(g["d_home"].to_numpy(float), g["d_away"].to_numpy(float))
        dmin = np.where(np.isnan(dmin), 60.0, dmin)
        contact = (dmin <= alone_r) & ~missing
        static = (speed < static_speed) & (dmin > alone_r) & ~missing
        fr = g["frame"].to_numpy(np.int64)
        cols["ball_missing"][idx] = missing
        cols["ball_out_of_pitch"][idx] = out_pitch
        cols["ball_speed"][idx] = speed
        cols["d_min"][idx] = dmin
        cols["n_r2"][idx] = g["n_home_r2"].to_numpy() + g["n_away_r2"].to_numpy()
        cols["n_r5"][idx] = g["n_home_r5"].to_numpy() + g["n_away_r5"].to_numpy()
        cols["miss_len_sofar"][idx] = _run_length_so_far(missing) / fps
        cols["miss_len_total"][idx] = _run_length_total(missing) / fps
        cols["out_len_sofar"][idx] = _run_length_so_far(out_pitch) / fps
        cols["static_len_sofar"][idx] = _run_length_so_far(static) / fps
        cols["static_len_total"][idx] = _run_length_total(static) / fps
        ar = np.arange(k)
        last = np.maximum.accumulate(np.where(contact, ar, -1))
        cols["since_contact"][idx] = np.where(last < 0, 60.0, (ar - last) / fps)
        nxt = np.minimum.accumulate(np.where(contact, ar, k + 1500)[::-1])[::-1]
        cols["until_contact"][idx] = np.minimum((nxt - ar) / fps, 60.0)
        cx_gap = g["home_cx"].to_numpy(float) - g["away_cx"].to_numpy(float)
        cols["cx_gap"][idx] = np.nan_to_num(cx_gap, nan=0.0)
        cols["ball_vs_cx_mean"][idx] = np.nan_to_num(
            g["ball_x"].to_numpy(float) - 0.5 * (g["home_cx"].to_numpy(float) + g["away_cx"].to_numpy(float)), nan=0.0)
        in_sp = np.zeros(k, dtype=bool)
        since_start = np.full(k, 60.0)
        until_end = np.full(k, 60.0)
        since_end = np.full(k, 60.0)
        until_start = np.full(k, 60.0)
        if set_pieces is not None and len(set_pieces):
            sp = set_pieces[(set_pieces["match_id"] == m) & (set_pieces["period"] == per)]
            s0 = np.searchsorted(fr, sp["start_frame"].to_numpy(np.int64))
            e0 = np.searchsorted(fr, sp["end_frame"].to_numpy(np.int64))
            for a, b in zip(s0, e0, strict=True):
                a, b = int(a), int(min(b, k))
                if a >= k:
                    continue
                in_sp[a:b] = True
                since_start[a:b] = np.minimum(since_start[a:b], (np.arange(a, b) - a) / fps)
                until_end[a:b] = np.minimum(until_end[a:b], (b - np.arange(a, b)) / fps)
            if len(s0):
                starts_sorted = np.sort(s0)
                ends_sorted = np.sort(e0)
                nxt_start = starts_sorted[np.minimum(np.searchsorted(starts_sorted, ar, side="right"),
                                                     len(starts_sorted) - 1)]
                until_start = np.where(nxt_start > ar, (nxt_start - ar) / fps, 60.0)
                prev_end_i = np.searchsorted(ends_sorted, ar, side="right") - 1
                prev_end = np.where(prev_end_i >= 0, ends_sorted[np.maximum(prev_end_i, 0)], -10**9)
                since_end = np.where(prev_end >= 0, (ar - prev_end) / fps, 60.0)
        cols["in_sp"][idx] = in_sp
        cols["sp_since_start"][idx] = np.minimum(since_start, 60.0)
        cols["sp_until_end"][idx] = np.minimum(until_end, 60.0)
        cols["sp_since_end"][idx] = np.minimum(since_end, 60.0)
        cols["sp_until_start"][idx] = np.minimum(until_start, 60.0)
    out = f[KEY].copy()
    for c in DEAD_FEATURE_COLS:
        out[c] = cols[c]
    return out


def windowed_dead(df: pd.DataFrame, lags: tuple[int, ...] = (-50, -12, 12, 50),
                  rolls: tuple[int, ...] = (25, 125, 625)) -> pd.DataFrame:
    """Lags / centred rolling means of the dead-ball cue columns, per block."""
    base = ["ball_missing", "ball_out_of_pitch", "ball_speed", "d_min", "n_r5", "in_sp"]
    g = df.groupby(["match_id", "period"], sort=False)[base]
    new = {}
    for lag in lags:
        sh = g.shift(lag)
        for c in base:
            new[f"{c}_dlag{lag}"] = sh[c].to_numpy()
    for w in rolls:
        rm = g.rolling(w, center=True, min_periods=1).mean().reset_index(level=[0, 1], drop=True).sort_index()
        for c in base:
            new[f"{c}_droll{w}"] = rm[c].to_numpy()
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


class DeadBallModel:
    """Binary LightGBM D detector on :func:`dead_ball_features` (+ :func:`windowed_dead`)."""

    def __init__(self, threshold: float = 0.5, min_run_frames: int = 25, **params):
        self.params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
                       "feature_fraction": 0.8, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 1.0,
                       "num_threads": 2, "verbose": -1, "seed": 0, "deterministic": True, "force_row_wise": True}
        self.params.update(params)
        self.threshold, self.min_run_frames = threshold, min_run_frames
        self.model, self.cols = None, None

    @staticmethod
    def design(feat: pd.DataFrame, set_pieces: pd.DataFrame | None, fps: float = 25.0) -> pd.DataFrame:
        return windowed_dead(dead_ball_features(feat, set_pieces, fps))

    def fit(self, design: pd.DataFrame, labels: pd.DataFrame, num_rounds: int = 200) -> DeadBallModel:
        import lightgbm as lgb

        self.cols = [c for c in design.columns if c not in KEY]
        y = design[KEY].merge(labels[KEY + ["possession"]], on=KEY, how="left")["possession"]
        ok = y.notna().to_numpy()
        X = design.loc[ok, self.cols].to_numpy(np.float32)
        self.model = lgb.train(self.params, lgb.Dataset(X, (y[ok] == "none").to_numpy(int)), num_boost_round=num_rounds)
        return self

    def predict_proba(self, design: pd.DataFrame) -> np.ndarray:
        return self.model.predict(design[self.cols].to_numpy(np.float32))

    def predict(self, design: pd.DataFrame, threshold: float | None = None, min_run_frames: int | None = None,
                stride: int = 1) -> pd.DataFrame:
        thr = self.threshold if threshold is None else threshold
        mr = self.min_run_frames if min_run_frames is None else min_run_frames
        p = self.predict_proba(design)
        out = design[KEY].copy()
        out["p_dead"] = p
        out["dead"] = decode_dead(out, thr, mr)
        return out


def decode_dead(tbl: pd.DataFrame, threshold: float = 0.5, min_run_frames: int = 25) -> np.ndarray:
    """Threshold ``p_dead`` and drop D runs shorter than ``min_run_frames`` (per block)."""
    p = tbl["p_dead"].to_numpy(float)
    dead = p >= threshold
    out = dead.copy()
    for idx in _blocks(tbl):
        d = dead[idx]
        s, e = _true_runs(d)
        for a, b in zip(s, e, strict=True):
            if b - a < min_run_frames:
                d[a:b] = False
        out[idx] = d
    return out


def merge_dead(pred: pd.DataFrame, dead: pd.DataFrame, mode: str = "union", min_dead_frames: int = 0) -> pd.DataFrame:
    """Combine segmenter labels with a D flag.

    ``union``: D where the segmenter or the detector says so; ``replace``: D only where the detector says so,
    the segmenter's ``none`` becomes the carried team label; ``segmenter``: ignore the detector. Removed D runs
    take the last team label (back-filled at a block head). D runs shorter than ``min_dead_frames`` are dropped.
    """
    p = pred.sort_values(KEY, kind="stable").reset_index(drop=True)
    codes = codes_of(p)
    d = p[KEY].merge(dead[KEY + ["dead"]], on=KEY, how="left")["dead"].fillna(False).to_numpy(bool)
    seg_dead = codes == NONE
    if mode == "union":
        new_dead = seg_dead | d
    elif mode == "replace":
        new_dead = d
    elif mode == "segmenter":
        new_dead = seg_dead
    else:
        raise ValueError("mode must be union | replace | segmenter")
    out = codes.copy()
    for idx in _blocks(p):
        nd = new_dead[idx].copy()
        if min_dead_frames > 0:
            s, e = _true_runs(nd)
            for a, b in zip(s, e, strict=True):
                if b - a < min_dead_frames:
                    nd[a:b] = False
        team = codes[idx].astype(float)
        team[team == NONE] = np.nan
        team = pd.Series(team).ffill().bfill().to_numpy()
        team = np.where(np.isnan(team), NONE, team).astype(np.int8)
        out[idx] = np.where(nd, NONE, team)
    res = p[KEY].copy()
    res["possession"] = labels_of(out)
    return res


# ------------------------------------------------------------------------------------ long-gap guard
@dataclass
class GapGuardConfig:
    """P6.robust: inside a ball-missing stretch the pipeline may only say D on explicit dead-ball evidence.

    A stretch at least ``min_gap_s`` long is guarded. D frames inside it are kept only where (a) the players are
    standing (share of players slower than ``static_speed`` >= ``static_frac`` for >= ``static_s``), (b) a
    Smart-Tagging set-piece window covers the frame, or (c) the frame lies within ``evidence_carry_s`` of the
    stretch start and the visible ball was out of the pitch / static with nobody near it during the ``pre_s``
    before the stretch (the ball went out, then the tracker lost it). Every other D frame takes the carried team
    label (the challenge's own rule for untracked stretches). ``static_frac`` needs the player-only feature
    ``static_frac`` of :mod:`features`; without it only (b) and (c) apply.
    """

    min_gap_s: float = 3.0
    pre_s: float = 2.0
    evidence_carry_s: float = 10.0
    static_frac: float = 0.7
    static_speed: float = 0.5  # ball static rule (m/s) for the pre-window evidence
    static_s: float = 2.0
    alone_r: float = 1.5
    fps: float = 25.0

    def to_dict(self) -> dict:
        return asdict(self)


def gap_guard(pred: pd.DataFrame, feat: pd.DataFrame, cfg: GapGuardConfig | None = None,
              set_pieces: pd.DataFrame | None = None, return_stats: bool = False):
    """Apply :class:`GapGuardConfig` to a predictions table; returns the table (+ ``{"frames_changed", "gaps"}``)."""
    cfg = cfg or GapGuardConfig()
    p, f = _aligned(pred, feat)
    codes = codes_of(p)
    out = codes.copy()
    changed = n_gaps = 0
    has_static = "static_frac" in f.columns
    for idx in _blocks(p):
        g = f.iloc[idx]
        n = len(idx)
        missing = g["ball_missing"].to_numpy() == 1
        out_pitch = (g["ball_out_of_pitch"].to_numpy() == 1) & ~missing
        speed = np.nan_to_num(g["ball_speed"].to_numpy(float), nan=0.0)
        dmin = np.fmin(g["d_home"].to_numpy(float), g["d_away"].to_numpy(float))
        dmin = np.where(np.isnan(dmin), np.inf, dmin)
        ball_static = (speed < cfg.static_speed) & (dmin > cfg.alone_r) & ~missing
        allowed = np.zeros(n, dtype=bool)
        if has_static:
            st = g["static_frac"].to_numpy(float) >= cfg.static_frac
            allowed |= _run_length_total(st) >= int(cfg.static_s * cfg.fps)
        if set_pieces is not None and len(set_pieces):
            m, per = p["match_id"].iloc[idx[0]], p["period"].iloc[idx[0]]
            sp = set_pieces[(set_pieces["match_id"] == m) & (set_pieces["period"] == per)]
            fr = g["frame"].to_numpy(np.int64)
            for a, b in zip(np.searchsorted(fr, sp["start_frame"].to_numpy(np.int64)),
                            np.searchsorted(fr, sp["end_frame"].to_numpy(np.int64)), strict=True):
                allowed[a:min(b, n)] = True
        c = codes[idx].copy()
        team = c.astype(float)
        team[team == NONE] = np.nan
        team = pd.Series(team).ffill().bfill().to_numpy()
        min_len, pre = int(cfg.min_gap_s * cfg.fps), int(cfg.pre_s * cfg.fps)
        carry = int(cfg.evidence_carry_s * cfg.fps)
        for a, b in zip(*_true_runs(missing), strict=True):
            if b - a < min_len:
                continue
            n_gaps += 1
            keep = allowed[a:b].copy()
            if (out_pitch[max(0, a - pre):a] | ball_static[max(0, a - pre):a]).any():
                keep[:carry] = True
            seg = c[a:b]
            flip = (seg == NONE) & ~keep & ~np.isnan(team[a:b])
            seg[flip] = team[a:b][flip].astype(np.int8)
            changed += int(flip.sum())
            c[a:b] = seg
        out[idx] = c
    res = p[KEY].copy()
    res["possession"] = labels_of(out)
    return (res, {"frames_changed": int(changed), "gaps": int(n_gaps)}) if return_stats else res


# -------------------------------------------------------------------------------- fragmentation control
def merge_short_team_runs(codes: np.ndarray, min_frames: int) -> np.ndarray:
    """Merge team runs (home/away) shorter than ``min_frames`` into the previous run (next run at a block
    head), shortest first, until none is left. ``none`` runs are never merged away here."""
    c = np.asarray(codes).copy()
    if min_frames <= 1 or len(c) == 0:
        return c
    while True:
        starts, ends, vals = runs(c)
        lengths = ends - starts
        short = np.flatnonzero((lengths < min_frames) & (vals != NONE))
        if len(short) == 0 or len(starts) == 1:
            return c
        k = short[np.argmin(lengths[short])]
        if k > 0:
            target = vals[k - 1]
        else:
            target = vals[k + 1]
        c[starts[k]:ends[k]] = target


def fragmentation_control(pred: pd.DataFrame, min_spell_frames: int = 15) -> pd.DataFrame:
    """Merge every predicted team spell shorter than ``min_spell_frames`` into its neighbour (per block)."""
    p = pred.sort_values(KEY, kind="stable").reset_index(drop=True)
    codes = codes_of(p)
    out = codes.copy()
    for idx in _blocks(p):
        out[idx] = merge_short_team_runs(codes[idx], min_spell_frames)
    res = p[KEY].copy()
    res["possession"] = labels_of(out)
    for c in ("p_home", "p_away", "p_none"):
        if c in p.columns:
            res[c] = p[c].to_numpy()
    return res


def spell_rate(pred: pd.DataFrame, fps: float = 25.0) -> float:
    """Team spells per minute of predicted frames."""
    p = pred.sort_values(KEY, kind="stable")
    codes = codes_of(p)
    n_sp = 0
    for idx in _blocks(p.reset_index(drop=True)):
        _, _, vals = runs(codes[idx])
        n_sp += int((vals != NONE).sum())
    minutes = len(p) / fps / 60
    return n_sp / minutes if minutes else float("nan")


# ------------------------------------------------------------------------------ Smart-Tagging simulation
SP_TYPES = ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off")


def restart_type(bx: float, by: float, L: float, W: float) -> str:
    """Set-piece type from where the ball restarts (metres, centre origin) — same rule as the fixture."""
    if np.isnan(bx):
        return "Free Kick"
    if abs(by) > W / 2 - 2.0:
        return "Corner Kick" if abs(bx) > L / 2 - 3.0 else "Throw-in"
    if abs(bx) > L / 2 - 6.0 and abs(by) < 10.0:
        return "Goal Kick"
    if abs(bx) < 1.5 and abs(by) < 1.5:
        return "Kick Off"
    return "Free Kick"


def simulate_set_pieces(labels: pd.DataFrame, feat: pd.DataFrame, pitch: tuple[float, float] = (105.0, 68.0),
                        fps: float = 25.0, seed: int = 0, early_s: float = 3.5, length_s: float = 5.0,
                        jitter_s: float = 0.5, miss_frac: float = 0.05, false_per_min: float = 0.0) -> pd.DataFrame:
    """Smart-Tagging-like SET PIECES windows from a labels table: one window per dead->team restart, opening
    ``early_s`` before the restart (± ``jitter_s`` gaussian) and lasting ``length_s``; type from the first visible
    ball position after the restart; throw-ins carry no team; ``miss_frac`` of restarts are dropped and
    ``false_per_min`` spurious windows are added (the real tagging is automatic). Columns: match_id, period,
    start_frame, end_frame (half-open, feature frame units), type, team, restart_frame."""
    rng = np.random.default_rng(seed)
    L, W = pitch
    lab = labels.sort_values(KEY, kind="stable").reset_index(drop=True)
    f = feat.sort_values(KEY, kind="stable").reset_index(drop=True)
    bx_all = lab[KEY].merge(f[KEY + ["ball_x", "ball_y", "ball_missing"]], on=KEY, how="left")
    rows = []
    for (m, per), idx in lab.groupby(["match_id", "period"], sort=False).indices.items():
        idx = np.sort(idx)
        codes = lab["possession"].iloc[idx].map(_C2I).to_numpy(np.int8)
        fr = lab["frame"].iloc[idx].to_numpy(np.int64)
        bx = bx_all["ball_x"].iloc[idx].to_numpy(float)
        by = bx_all["ball_y"].iloc[idx].to_numpy(float)
        miss = bx_all["ball_missing"].iloc[idx].to_numpy(float) == 1
        starts, _, vals = runs(codes)
        for k in range(1, len(starts)):
            if vals[k] == NONE or vals[k - 1] != NONE:
                continue
            if rng.random() < miss_frac:
                continue
            r = int(starts[k])
            px, py = np.nan, np.nan
            for t in range(r, min(len(fr), r + 25)):
                if not miss[t] and not np.isnan(bx[t]):
                    px, py = bx[t], by[t]
                    break
            typ = "Kick Off" if k == 1 else restart_type(px, py, L, W)
            s = r - int(round((early_s + rng.normal(0, jitter_s)) * fps))
            s = max(0, s)
            e = s + int(round(length_s * fps))
            team = None if typ == "Throw-in" else CLASSES[int(vals[k])]
            rows.append({"match_id": m, "period": per, "start_frame": int(fr[min(max(s, 0), len(fr) - 1)]),
                         "end_frame": int(fr[min(e, len(fr) - 1)]) + (1 if e >= len(fr) else 0), "type": typ,
                         "team": team, "restart_frame": int(fr[r]), "false_positive": False})
        if false_per_min > 0:
            n_false = rng.poisson(false_per_min * len(idx) / fps / 60)
            for t in rng.integers(0, len(idx), size=n_false):
                e = min(int(t) + int(length_s * fps), len(fr) - 1)
                rows.append({"match_id": m, "period": per, "start_frame": int(fr[t]), "end_frame": int(fr[e]),
                             "type": "Free Kick", "team": None, "restart_frame": -1, "false_positive": True})
    cols = ["match_id", "period", "start_frame", "end_frame", "type", "team", "restart_frame", "false_positive"]
    df = pd.DataFrame(rows, columns=cols).sort_values(["match_id", "period", "start_frame"]).reset_index(drop=True)
    df["team"] = df["team"].astype(object).where(df["team"].notna(), None)
    return df


# ------------------------------------------------------------------------------------------- pipeline
@dataclass
class PostprocessConfig:
    hmm_stay: float = 0.995
    refine: RefineConfig | None = field(default_factory=RefineConfig)
    dead_mode: str = "union"  # union | replace | segmenter
    dead_rules: DeadBallRules | None = field(default_factory=DeadBallRules)
    min_dead_frames: int = 0
    min_spell_frames: int = 0  # 0 = no fragmentation control
    gap_guard: GapGuardConfig | None = None  # P6.robust long-gap guard (None = off)

    def to_dict(self) -> dict:
        return {"hmm_stay": self.hmm_stay, "refine": self.refine.to_dict() if self.refine else None,
                "dead_mode": self.dead_mode, "dead_rules": self.dead_rules.to_dict() if self.dead_rules else None,
                "min_dead_frames": self.min_dead_frames, "min_spell_frames": self.min_spell_frames,
                "gap_guard": self.gap_guard.to_dict() if self.gap_guard else None}

    @classmethod
    def from_dict(cls, d: dict) -> PostprocessConfig:
        return cls(hmm_stay=d["hmm_stay"], refine=RefineConfig(**d["refine"]) if d.get("refine") else None,
                   dead_mode=d.get("dead_mode", "union"),
                   dead_rules=DeadBallRules(**d["dead_rules"]) if d.get("dead_rules") else None,
                   min_dead_frames=d.get("min_dead_frames", 0), min_spell_frames=d.get("min_spell_frames", 0),
                   gap_guard=GapGuardConfig(**d["gap_guard"]) if d.get("gap_guard") else None)


def postprocess(probs: pd.DataFrame, feat: pd.DataFrame, cfg: PostprocessConfig,
                set_pieces: pd.DataFrame | None = None, dead_model: DeadBallModel | None = None,
                dead_design: pd.DataFrame | None = None) -> pd.DataFrame:
    """probs (KEY + p_home/p_away/p_none) -> HMM -> D merge -> boundary refine -> fragmentation control."""
    from .model import hmm_smooth

    pred = hmm_smooth(probs, cfg.hmm_stay)
    if cfg.dead_mode != "segmenter":
        if dead_model is not None:
            design = dead_design if dead_design is not None else DeadBallModel.design(feat, set_pieces)
            dead = dead_model.predict(design)
        elif cfg.dead_rules is not None:
            dead = cfg.dead_rules.detect(feat, set_pieces)
        else:
            dead = None
        if dead is not None:
            pred = merge_dead(pred, dead, cfg.dead_mode, cfg.min_dead_frames)
    elif cfg.min_dead_frames > 0:
        pred = merge_dead(pred, pred[KEY].assign(dead=False), "segmenter", cfg.min_dead_frames)
    if cfg.gap_guard is not None:
        pred = gap_guard(pred, feat, cfg.gap_guard, set_pieces)
    if cfg.refine is not None and cfg.refine.strategy != "none":
        pred = refine_boundaries(pred, feat, cfg.refine)
    if cfg.min_spell_frames > 0:
        pred = fragmentation_control(pred, cfg.min_spell_frames)
    return pred


__all__ = ["RefineConfig", "refine_boundaries", "refine_block", "DeadBallRules", "DeadBallModel",
           "dead_ball_features", "windowed_dead", "decode_dead", "merge_dead", "fragmentation_control",
           "merge_short_team_runs", "spell_rate", "simulate_set_pieces", "restart_type", "PostprocessConfig",
           "postprocess", "runs", "codes_of", "labels_of", "DEAD_FEATURE_COLS", "CLASSES", "GapGuardConfig",
           "gap_guard", "carry_over_missing"]


def carry_over_missing(pred: pd.DataFrame, feat: pd.DataFrame, min_missing_frac: float = 0.8
                       ) -> tuple[pd.DataFrame, int]:
    """M5.ball-fix possession guard: a D run whose frames are (almost) all *ball missing* is not evidence of a
    dead ball — the tracker lost the ball — so it takes the last team label (back-filled at a block head), the
    challenge's own rule for untracked stretches. D runs with a visible ball (out of the pitch, static, set
    piece) are kept. Returns (predictions, frames changed)."""
    p = pred.sort_values(KEY, kind="stable").reset_index(drop=True)
    miss = p[KEY].merge(feat[KEY + ["ball_missing"]], on=KEY, how="left")["ball_missing"].fillna(0).to_numpy() == 1
    lab = p["possession"].astype(str).to_numpy(object).copy()
    changed = 0
    for idx in _blocks(p):
        L = lab[idx]
        team = pd.Series(np.where(np.isin(L, ("home", "away")), L, None), dtype=object).ffill().bfill().to_numpy()
        is_none = L == "none"
        if not is_none.any() or team[0] is None:
            continue
        for a, b in zip(*_true_runs(is_none), strict=True):
            if miss[idx[a:b]].mean() >= min_missing_frac:
                L[a:b] = team[a:b]
                changed += b - a
        lab[idx] = L
    out = p.copy()
    out["possession"] = lab
    return out, int(changed)
