"""Rule baseline: nearest-player / ball-control heuristic with hysteresis.

State machine over the per-frame features (features.py):
  * a team is a *candidate* for a frame when its nearest player is within `control_r` of the ball and at least
    `margin` metres closer than the other team's nearest player;
  * the current holder keeps possession until the other team is the candidate for `k_switch` consecutive
    frames (hysteresis);
  * `none` when the ball is out of the pitch for `k_dead` consecutive frames (dead-ball proxy) — the next
    candidate restarts play;
  * ball-missing frames carry the last state unchanged (candidate evidence is not updated).
Output: predictions table (match_id, period, frame, possession).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .features import KEY


@dataclass
class RuleParams:
    control_r: float = 2.5   # m
    margin: float = 0.0      # m, how much closer the candidate team must be
    k_switch: int = 3        # frames of consistent evidence before possession flips
    k_dead: int = 5          # frames out of pitch before `none`
    k_missing: int = 50      # frames of missing ball before `none` (0 = never; Metrica drops the ball when dead)

    def to_dict(self) -> dict:
        return asdict(self)


def _candidates(feat: pd.DataFrame, p: RuleParams) -> np.ndarray:
    dh, da = feat.d_home.to_numpy(float), feat.d_away.to_numpy(float)
    missing = feat.ball_missing.to_numpy() == 1
    cand = np.zeros(len(feat), dtype=np.int8)
    home_ok = (dh <= p.control_r) & ((da - dh) >= p.margin)
    away_ok = (da <= p.control_r) & ((dh - da) >= p.margin)
    cand[home_ok & ~away_ok] = 1
    cand[away_ok & ~home_ok] = -1
    both = home_ok & away_ok
    cand[both] = np.where(dh[both] <= da[both], 1, -1) if p.margin <= 0 else 0
    cand[missing | np.isnan(dh) | np.isnan(da)] = 0
    return cand


def _run_period(cand: np.ndarray, out_of_pitch: np.ndarray, missing: np.ndarray, p: RuleParams) -> np.ndarray:
    state = 0  # +1 home, -1 away, 0 none
    streak_team, streak, dead_streak, miss_streak = 0, 0, 0, 0
    out = np.zeros(len(cand), dtype=np.int8)
    for i in range(len(cand)):
        if missing[i]:
            miss_streak += 1
            if p.k_missing and miss_streak >= p.k_missing:
                state, streak = 0, 0
            out[i] = state
            continue
        miss_streak = 0
        if out_of_pitch[i]:
            dead_streak += 1
            if dead_streak >= p.k_dead:
                state, streak = 0, 0
                out[i] = state
                continue  # no one can take the ball while it is out of play
        else:
            dead_streak = 0
        c = cand[i]
        if c != 0 and c != state:
            if c == streak_team:
                streak += 1
            else:
                streak_team, streak = c, 1
            need = 1 if state == 0 else p.k_switch
            if streak >= need:
                state, streak = c, 0
        elif c == state:
            streak_team, streak = 0, 0
        out[i] = state
    return out


def predict_rules(feat: pd.DataFrame, params: RuleParams | None = None) -> pd.DataFrame:
    p = params or RuleParams()
    feat = feat.sort_values(KEY, kind="stable")
    labels = np.empty(len(feat), dtype=object)
    pos = 0
    for _, g in feat.groupby(["match_id", "period"], sort=False):
        s = _run_period(_candidates(g, p), g.ball_out_of_pitch.to_numpy() == 1, g.ball_missing.to_numpy() == 1, p)
        labels[pos:pos + len(g)] = np.where(s == 1, "home", np.where(s == -1, "away", "none"))
        pos += len(g)
    return pd.DataFrame({**{k: feat[k].to_numpy() for k in KEY}, "possession": labels})


def grid_search(feat: pd.DataFrame, labels: pd.DataFrame, score_fn, grid: dict | None = None) -> tuple[RuleParams, dict]:
    """Tiny grid search on one match; returns the best params by `score_fn(pred, labels)['selection']`."""
    import itertools

    grid = grid or {"control_r": [1.5, 2.0, 2.5, 3.0], "margin": [0.0, 0.5], "k_switch": [1, 3, 6, 12],
                    "k_dead": [5], "k_missing": [0, 25, 50, 100]}
    best, best_s = None, -1.0
    for vals in itertools.product(*grid.values()):
        p = RuleParams(**dict(zip(grid.keys(), vals)))
        s = score_fn(predict_rules(feat, p), labels)
        if s["selection"] > best_s:
            best, best_s = p, s["selection"]
    return best, {"selection": best_s}
