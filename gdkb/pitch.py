"""Pitch utilities (contract in docs/frame-format.md, section "Pitch utils").

Coordinates: metres, origin at the pitch centre, x along the long axis, y across (y grows towards the
"top" touchline when looking at the pitch with the home team's first-period goal on the left, i.e. the
same handedness as Sportec/SkillCorner: ``VerticalOrientation.BOTTOM_TO_TOP``; Metrica's top-left
0–1 frame is flipped in the loader).

Homographies use the OpenCV convention: ``H`` is 3x3 and maps *pixel* homogeneous points to *pitch*
points, ``[x', y', w]^T = H @ [u, v, 1]^T``, ``x = x'/w``, ``y = y'/w`` (same as ``cv2.perspectiveTransform``
with the matrix returned by ``cv2.findHomography(px, pitch)``). No OpenCV dependency here: numpy only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gdkb.formats import MatchMeta


def pitch_bounds(meta: MatchMeta) -> tuple[float, float, float, float]:
    """``(x_min, x_max, y_min, y_max)`` of the marked pitch in metres."""
    hl, hw = meta.pitch_length_m / 2.0, meta.pitch_width_m / 2.0
    return (-hl, hl, -hw, hw)


def to_attacking_right(frames: pd.DataFrame, meta: MatchMeta, team: str) -> pd.DataFrame:
    """Return a copy of ``frames`` with x/y rotated by 180° in every period where ``team`` attacks −x,
    so that ``team`` ('home' or 'away') always attacks +x. Ball rows rotate with the rest; z untouched."""
    if team not in ("home", "away"):
        raise ValueError("team must be 'home' or 'away'")
    out = frames.copy()
    flip_periods = [p for p in out["period"].unique() if not meta.attacks_right(team, int(p))]
    if flip_periods:
        m = out["period"].isin(flip_periods)
        out.loc[m, "x"] = -out.loc[m, "x"]
        out.loc[m, "y"] = -out.loc[m, "y"]
    return out


def infer_home_attacks_right(frames: pd.DataFrame) -> dict[int, bool]:
    """Data-driven direction of play per period: the team whose players sit at lower mean x defends the
    left goal, i.e. attacks right. Uses all frames of the period (robust to a few seconds of noise)."""
    pl = frames[frames["team"].isin(["home", "away"])]
    mx = pl.groupby(["period", "team"], observed=True)["x"].mean().unstack("team")
    return {int(p): bool(row["home"] < row["away"]) for p, row in mx.iterrows()}


def _as_points(points) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    if p.shape[-1] != 2:
        raise ValueError("points must have shape (N, 2)")
    return p


def apply_homography(points, H) -> np.ndarray:
    """Apply a 3x3 projective transform to (N, 2) points (OpenCV ``perspectiveTransform`` semantics)."""
    p = _as_points(points)
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3):
        raise ValueError("H must be 3x3")
    hom = np.c_[p, np.ones(len(p))] @ H.T
    w = hom[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return hom[:, :2] / w


def pixel_to_pitch(points_px, H) -> np.ndarray:
    """Image pixels (u, v) → pitch metres (x, y) with ``H`` = pixel→pitch homography."""
    return apply_homography(points_px, H)


def pitch_to_pixel(points_m, H) -> np.ndarray:
    """Pitch metres (x, y) → image pixels (u, v) using the inverse of the pixel→pitch homography ``H``."""
    return apply_homography(points_m, np.linalg.inv(np.asarray(H, dtype=float)))


def homography_from_points(src, dst) -> np.ndarray:
    """Least-squares DLT homography mapping ``src`` → ``dst`` (>= 4 point pairs), normalised so H[2,2] = 1.
    Equivalent to ``cv2.findHomography(src, dst, 0)`` up to numerical noise; used for tests and for
    video-pipeline calibration from pitch-line correspondences."""
    s, d = _as_points(src), _as_points(dst)
    if len(s) < 4 or len(s) != len(d):
        raise ValueError("need >= 4 matching point pairs")
    # Hartley normalisation for conditioning
    Ts, sn = _normalise(s)
    Td, dn = _normalise(d)
    rows = []
    for (x, y), (u, v) in zip(sn, dn, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def _normalise(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = p.mean(axis=0)
    s = np.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return T, apply_homography(p, T)
