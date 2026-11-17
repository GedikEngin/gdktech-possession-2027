"""Per-frame possession diagnostics + transition timing error (module 4 of PLAN.md).

Predictions and labels are tables with (match_id, period, frame, possession); evaluation joins on the key
(docs/frame-format.md). Frames present in the labels but missing from the predictions count as wrong
(`coverage` reports the fraction predicted).

The official ranking metric is `spellF1` (possession spells matched by temporal IoU) with `macroF1` as the
tie-break -- see gdkb/scoring/official.py (vendored from the challenge repo) and docs/spec/scorer.md.
`gdkb.scoring.score` (the package-level wrapper) adds those two to the dict below; this module's own numbers are
diagnostics. `transition_error` is ours: median |pred - true| frames between each true possession change and
the nearest predicted change (any class) within `max_lag` frames; unmatched true transitions count as misses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["home", "away", "none"]
KEY = ["match_id", "period", "frame"]


def _align(pred: pd.DataFrame, labels: pd.DataFrame, ball_missing: pd.DataFrame | None) -> pd.DataFrame:
    df = labels[KEY + ["possession"]].rename(columns={"possession": "y"}).merge(
        pred[KEY + ["possession"]].rename(columns={"possession": "p"}), on=KEY, how="left")
    df["p"] = df.p.fillna("__missing__")
    if ball_missing is not None:
        df = df.merge(ball_missing[KEY + ["ball_missing"]], on=KEY, how="left")
        df["ball_missing"] = df.ball_missing.fillna(0).astype(int)
    else:
        df["ball_missing"] = 0
    return df.sort_values(KEY, kind="stable").reset_index(drop=True)


def _f1(y: np.ndarray, p: np.ndarray, c: str) -> float:
    tp = np.sum((y == c) & (p == c))
    fp = np.sum((y != c) & (p == c))
    fn = np.sum((y == c) & (p != c))
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan")


def transitions(df: pd.DataFrame, col: str) -> list[tuple[str, int, int]]:
    """(match_id, period, frame) of every change of `col` within a (match, period) block, frame = first new."""
    out = []
    for (m, per), g in df.groupby(["match_id", "period"], sort=False):
        v = g[col].to_numpy()
        fr = g.frame.to_numpy()
        ch = np.flatnonzero(v[1:] != v[:-1]) + 1
        out.extend((m, per, int(fr[i])) for i in ch)
    return out


def transition_error(df: pd.DataFrame, max_lag: int = 125) -> dict:
    t_true = transitions(df, "y")
    t_pred = transitions(df, "p")
    by_block: dict = {}
    for m, per, f in t_pred:
        by_block.setdefault((m, per), []).append(f)
    errs, matched = [], 0
    for m, per, f in t_true:
        cand = np.asarray(by_block.get((m, per), []))
        if len(cand) == 0:
            continue
        d = np.min(np.abs(cand - f))
        if d <= max_lag:
            errs.append(int(d))
            matched += 1
    errs = np.asarray(errs)
    return {"n_true": len(t_true), "n_pred": len(t_pred),
            "matched_frac": matched / len(t_true) if t_true else float("nan"),
            "median_abs_frames": float(np.median(errs)) if len(errs) else float("nan"),
            "mean_abs_frames": float(np.mean(errs)) if len(errs) else float("nan"),
            "max_lag": max_lag}


def score(pred: pd.DataFrame, labels: pd.DataFrame, ball_missing: pd.DataFrame | None = None,
          max_lag: int = 125) -> dict:
    df = _align(pred, labels, ball_missing)
    y, p = df.y.to_numpy(object), df.p.to_numpy(object)
    acc = float(np.mean(y == p))
    per_class = {c: _f1(y, p, c) for c in CLASSES}
    present = [c for c in CLASSES if np.any(y == c)]
    macro = float(np.nanmean([per_class[c] for c in present])) if present else float("nan")
    bm = df.ball_missing.to_numpy() == 1
    not_none = y != "none"
    cm = {t: {q: int(np.sum((y == t) & (p == q))) for q in CLASSES + ["__missing__"]} for t in CLASSES}
    res = {
        "n_frames": int(len(df)),
        "coverage": float(np.mean(p != "__missing__")),
        "accuracy": acc,
        "macro_f1": macro,
        "f1_per_class": per_class,
        "accuracy_ball_missing": float(np.mean((y == p)[bm])) if bm.any() else float("nan"),
        "n_ball_missing": int(bm.sum()),
        "accuracy_excl_none": float(np.mean((y == p)[not_none])) if not_none.any() else float("nan"),
        "accuracy_home_away_only": float(np.mean((y == p)[not_none & (p != "none")]))
        if (not_none & (p != "none")).any() else float("nan"),
        "confusion": cm,
        "label_fractions": {c: float(np.mean(y == c)) for c in CLASSES},
        "transition_error": transition_error(df, max_lag),
    }
    res["selection"] = acc
    return res


def results_markdown(rows: dict[str, dict]) -> str:
    """rows: {name: score dict} -> markdown table (spellF1 = official ranking metric, macroF1 = tie-break)."""
    head = ("| model | spellF1 | macroF1 | spells pred/true | acc | F1 home | F1 away | F1 none | acc ball-missing "
            "| acc excl none | transitions pred/true | matched | median lag (frames) |")
    lines = [head, "|" + "---|" * 13]
    for name, r in rows.items():
        f = r["f1_per_class"]
        te = r["transition_error"]
        sp = (f"{r['spellF1']:.4f} | {r['macroF1']:.4f} | {r['n_pred_spells']:.0f}/{r['n_gt_spells']:.0f}"
              if "spellF1" in r else "-- | -- | --")
        lines.append(f"| {name} | {sp} | {r['accuracy']:.4f} | {f['home']:.3f} | {f['away']:.3f} "
                     f"| {f['none']:.3f} | {r['accuracy_ball_missing']:.3f} | {r['accuracy_excl_none']:.4f} "
                     f"| {te['n_pred']}/{te['n_true']} | {te['matched_frac']:.2f} | {te['median_abs_frames']:.0f} |")
    return "\n".join(lines)


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
