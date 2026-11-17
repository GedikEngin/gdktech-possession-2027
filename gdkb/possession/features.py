"""Per-frame possession features from the frames table (docs/frame-format.md). Vendor-agnostic.

`frame_features(frames, meta)` -> one row per (match_id, period, frame), sorted, with:
  ball_x, ball_y, ball_missing, ball_speed, ball_accel, ball_dir_change, ball_out_of_pitch,
  d_home, d_away (nearest-player distance to the ball per team, NaN when the ball is missing),
  d_home2, d_away2 (second nearest), d_diff = d_home - d_away, nearest_team (+1 home, -1 away, 0 unknown),
  n_home_r2, n_away_r2, n_home_r5, n_away_r5 (players within 2 / 5 m of the ball),
  home_cx, away_cx (team centroid x in metres), ball_vs_home_cx, ball_vs_away_cx,
  contact_home, contact_away (ball within CONTACT_R of a player and slow-ish), t_since_contact_home/away (s),
  last_contact_team (+1 / -1 / 0), nearest_home_id, nearest_away_id,
  player-only cues (P6.robust, no ball needed — what carries the label through a long ball-missing stretch):
  home_vx, away_vx (centroid velocity along x, m/s, 1 s smoothing), home_spread, away_spread (rms distance of
  the players from their centroid, m), home_speed, away_speed (median player speed, m/s), static_frac (share of
  players slower than STATIC_SPEED), cx_gap (home_cx - away_cx, m).
Missing-ball frames keep ball_missing=1 and carry the last valid ball/distance values forward (ffill), so the
rules and models see a continuous signal plus an explicit flag.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

CONTACT_R = 1.5  # m: ball this close to a player counts as a possible touch
STATIC_SPEED = 0.7  # m/s: a player slower than this is standing (walking is ~1.4 m/s)
MAX_PLAYER_SPEED = 12.0  # m/s: faster = a tracklet jump (identity swap / break), not a player
KEY = ["match_id", "period", "frame"]


def _wide(frames: pd.DataFrame, team: str) -> tuple[np.ndarray, np.ndarray, list[str], pd.Index]:
    """(X, Y) arrays of shape (n_frames, n_players) for `team`, NaN where absent; plus ids and frame index."""
    sub = frames[frames.team == team]
    px = sub.pivot_table(index=["period", "frame"], columns="player_id", values="x", aggfunc="first")
    py = sub.pivot_table(index=["period", "frame"], columns="player_id", values="y", aggfunc="first")
    return px.to_numpy(float), py.reindex(columns=px.columns).to_numpy(float), list(px.columns), px.index


def _team_block(bx, by, X, Y, ids) -> dict:
    d = np.sqrt((X - bx[:, None]) ** 2 + (Y - by[:, None]) ** 2)
    all_nan = np.all(np.isnan(d), axis=1)
    d_safe = np.where(np.isnan(d), np.inf, d)
    order = np.argsort(d_safe, axis=1)
    d1 = np.take_along_axis(d_safe, order[:, :1], 1)[:, 0]
    d2 = np.take_along_axis(d_safe, order[:, 1:2], 1)[:, 0] if d.shape[1] > 1 else np.full(len(d), np.inf)
    d1 = np.where(np.isinf(d1), np.nan, d1)
    d2 = np.where(np.isinf(d2), np.nan, d2)
    nearest = np.array(ids, dtype=object)[order[:, 0]]
    nearest[all_nan] = None
    return {"d": d1, "d2": d2, "n_r2": np.sum(d_safe < 2.0, axis=1), "n_r5": np.sum(d_safe < 5.0, axis=1),
            "cx": np.nanmean(X, axis=1) if X.shape[1] else np.full(len(d), np.nan), "nearest": nearest}


def _player_block(X: np.ndarray, Y: np.ndarray, fps: float) -> dict:
    """Player-only cues of one team: centroid x velocity (1 s smoothed), spread, median speed, static count /
    present count. Robust to anonymous tracklets: speeds above MAX_PLAYER_SPEED (slot jumps) are ignored."""
    n = len(X)
    if X.shape[1] == 0 or n < 2:
        z = np.zeros(n)
        return {"vx": z, "spread": z, "speed": z, "n_static": z, "n_present": z}
    with np.errstate(invalid="ignore"):
        cx, cy = np.nanmean(X, 1), np.nanmean(Y, 1)
        spread = np.sqrt(np.nanmean((X - cx[:, None]) ** 2 + (Y - cy[:, None]) ** 2, 1))
        cx_s = pd.Series(cx).interpolate(limit_direction="both").rolling(int(fps), center=True, min_periods=1).mean()
        vx = np.gradient(cx_s.to_numpy(float), 1.0 / fps)
        vx_s = pd.Series(vx).rolling(int(fps), center=True, min_periods=1).mean().to_numpy(float)
        sp = np.hypot(np.gradient(X, 1.0 / fps, axis=0), np.gradient(Y, 1.0 / fps, axis=0))  # NaN where absent
        sp = np.where(sp > MAX_PLAYER_SPEED, np.nan, sp)
        sp = pd.DataFrame(sp).rolling(int(fps // 2) or 1, center=True, min_periods=1).mean().to_numpy(float)
        speed = np.nanmedian(sp, 1)
        n_present = np.sum(~np.isnan(sp), 1)
        n_static = np.sum(sp < STATIC_SPEED, 1)
    return {"vx": np.nan_to_num(vx_s), "spread": np.nan_to_num(spread), "speed": np.nan_to_num(speed),
            "n_static": n_static, "n_present": n_present}


def frame_features(frames: pd.DataFrame, meta: dict, interp_gaps: bool | None = None,
                   interp_max_s: float = 20.0) -> pd.DataFrame:
    """``interp_gaps`` (P4.pseudo-train quirk fix, default off; env ``GDKB_BALL_INTERP=1`` turns it on): inside a
    ball-missing stretch of at most ``interp_max_s`` the ball position is linearly interpolated between the last and
    the next visible position (offline, look-ahead allowed) instead of carrying the last value; ``ball_missing`` and the
    contact flags still see the gap. Longer gaps and gaps at a period edge keep the carry-forward behaviour."""
    if interp_gaps is None:
        interp_gaps = os.environ.get("GDKB_BALL_INTERP", "0") == "1"
    fps = float(meta.get("fps", 25))
    out = []
    for (mid, period), g in frames.groupby(["match_id", "period"], sort=True):
        ball = g[g.team == "ball"].sort_values("frame")
        fr = ball.frame.to_numpy()
        bx, by = ball.x.to_numpy(float), ball.y.to_numpy(float)
        missing = np.isnan(bx) | np.isnan(by)
        if interp_gaps and missing.any():
            bx, by = _interpolate_gaps(bx, by, missing, int(interp_max_s * fps))
        carry_mask = np.isnan(bx) | np.isnan(by)  # == missing unless gaps were interpolated
        # ball kinematics on the raw (possibly gappy) signal; gaps -> NaN then ffilled below
        vx, vy = np.gradient(bx, 1.0 / fps), np.gradient(by, 1.0 / fps)
        speed = np.hypot(vx, vy)
        accel = np.gradient(speed, 1.0 / fps)
        ang = np.arctan2(vy, vx)
        dang = np.abs(np.angle(np.exp(1j * np.diff(ang, prepend=ang[:1]))))
        L, W = float(meta.get("pitch_length_m", 105)), float(meta.get("pitch_width_m", 68))
        out_of_pitch = (np.abs(bx) > L / 2 + 0.5) | (np.abs(by) > W / 2 + 0.5)

        blocks = {}
        for team in ("home", "away"):
            X, Y, ids, idx = _wide(g, team)
            # align the team's frame index to the ball's frame index
            pos = pd.Index(idx.get_level_values("frame")).get_indexer(fr)
            Xa = np.full((len(fr), X.shape[1]), np.nan)
            Ya = Xa.copy()
            Xa[pos >= 0] = X[pos[pos >= 0]]
            Ya[pos >= 0] = Y[pos[pos >= 0]]
            blocks[team] = _team_block(bx, by, Xa, Ya, ids)
            blocks[team].update(_player_block(Xa, Ya, fps))

        df = pd.DataFrame({"match_id": mid, "period": period, "frame": fr, "time_s": ball.time_s.to_numpy(),
                           "ball_x": bx, "ball_y": by, "ball_missing": missing.astype(np.int8),
                           "ball_speed": speed, "ball_accel": accel, "ball_dir_change": dang,
                           "ball_out_of_pitch": out_of_pitch.astype(np.int8)})
        for t in ("home", "away"):
            b = blocks[t]
            df[f"d_{t}"], df[f"d_{t}2"] = b["d"], b["d2"]
            df[f"n_{t}_r2"], df[f"n_{t}_r5"], df[f"{t}_cx"] = b["n_r2"], b["n_r5"], b["cx"]
            df[f"ball_vs_{t}_cx"] = bx - b["cx"]
            df[f"nearest_{t}_id"] = b["nearest"]
            df[f"{t}_vx"], df[f"{t}_spread"], df[f"{t}_speed"] = b["vx"], b["spread"], b["speed"]
        n_present = blocks["home"]["n_present"] + blocks["away"]["n_present"]
        df["static_frac"] = np.where(n_present > 0, (blocks["home"]["n_static"] + blocks["away"]["n_static"])
                                     / np.maximum(n_present, 1), 0.0)
        df["cx_gap"] = np.nan_to_num(df.home_cx.to_numpy(float) - df.away_cx.to_numpy(float))
        df["d_diff"] = df.d_home - df.d_away
        df["nearest_team"] = np.sign(-df.d_diff.fillna(0)).astype(np.int8)  # +1 home nearer, -1 away nearer
        # contacts: ball within CONTACT_R of a player of that team
        for t in ("home", "away"):
            df[f"contact_{t}"] = ((df[f"d_{t}"] < CONTACT_R) & ~missing).astype(np.int8)
        df["t_since_contact_home"] = _time_since(df.contact_home.to_numpy(), fps)
        df["t_since_contact_away"] = _time_since(df.contact_away.to_numpy(), fps)
        lc = np.where(df.contact_home == 1, 1, np.where(df.contact_away == 1, -1, 0))
        df["last_contact_team"] = pd.Series(lc).replace(0, np.nan).ffill().fillna(0).astype(np.int8).to_numpy()
        # carry ball-dependent values over missing frames
        carry = ["ball_x", "ball_y", "ball_speed", "ball_accel", "ball_dir_change", "d_home", "d_away",
                 "d_home2", "d_away2", "d_diff", "ball_vs_home_cx", "ball_vs_away_cx"]
        df.loc[carry_mask, carry] = np.nan
        df[carry] = df[carry].ffill().bfill()
        df.loc[carry_mask, "nearest_team"] = pd.Series(df.nearest_team.where(~carry_mask)).ffill().fillna(0).astype(
            np.int8).to_numpy()[carry_mask]
        out.append(df)
    return pd.concat(out, ignore_index=True)


def _interpolate_gaps(bx: np.ndarray, by: np.ndarray, missing: np.ndarray,
                      max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of x/y over interior missing runs of length <= ``max_len`` (others stay NaN)."""
    x = pd.Series(bx).interpolate(limit_area="inside").to_numpy(copy=True)
    y = pd.Series(by).interpolate(limit_area="inside").to_numpy(copy=True)
    m = missing.astype(np.int8)
    edges = np.flatnonzero(np.diff(np.r_[0, m, 0]))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        if b - a > max_len:
            x[a:b] = np.nan
            y[a:b] = np.nan
    x = np.where(missing & np.isnan(x), bx, x)
    y = np.where(missing & np.isnan(y), by, y)
    return x, y


def _time_since(flag: np.ndarray, fps: float) -> np.ndarray:
    """Seconds since the last frame where flag==1 (large value before the first one)."""
    idx = np.arange(len(flag))
    last = np.where(flag == 1, idx, -1)
    last = np.maximum.accumulate(last)
    t = (idx - last) / fps
    t[last < 0] = 60.0
    return t


FEATURE_COLS = ["ball_x", "ball_y", "ball_missing", "ball_speed", "ball_accel", "ball_dir_change",
                "ball_out_of_pitch", "d_home", "d_away", "d_home2", "d_away2", "d_diff", "nearest_team",
                "n_home_r2", "n_away_r2", "n_home_r5", "n_away_r5", "ball_vs_home_cx", "ball_vs_away_cx",
                "contact_home", "contact_away", "t_since_contact_home", "t_since_contact_away",
                "last_contact_team",
                # P6.robust player-only cues (no ball needed)
                "home_vx", "away_vx", "home_spread", "away_spread", "home_speed", "away_speed", "static_frac",
                "cx_gap"]
BALL_FREE_COLS = FEATURE_COLS[-8:]  # the columns that stay informative while the ball is missing
