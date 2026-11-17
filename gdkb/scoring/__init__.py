"""Possession scoring (module 4 of PLAN.md).

Two layers:

* :mod:`gdkb.scoring.official` — the challenge's own ``possession_metrics.py`` (MIT, vendored unmodified from
  metrica-sports/more-than-a-hack-2027). ``spell_f1_mean`` is the ranking metric, ``macro_f1`` the tie-break.
* :mod:`gdkb.scoring.frame` — our older per-frame scorer (accuracy, F1s, transition timing). Kept for
  diagnostics; its ``score()`` is re-exported here and now also carries ``spellF1`` / ``macroF1`` from the
  official code.

:func:`spell_f1` is the thin wrapper the task card asked for: it takes our predictions/labels tables
(``match_id, period, frame, possession`` with ``possession in {home, away, none}`` and optionally ``abstain``
= ``X``) and returns the official ``evaluate()`` dict (json-safe), evaluated exactly as the organisers would:
each ``(match_id, period)`` block becomes a contiguous run separated by an ``X`` gap so no spell or
transition ever straddles a half boundary (``official.valid_runs``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import official
from .frame import KEY, results_markdown, save_json, transition_error, transitions
from .frame import score as frame_score

# our label vocabulary -> official codes (official.H/A/D/X == 0/1/2/3)
TO_CODE = {"home": official.H, "away": official.A, "none": official.D,
           "H": official.H, "A": official.A, "D": official.D, "X": official.X,
           "abstain": official.X, "x": official.X, "__missing__": official.X}
FROM_CODE = {official.H: "home", official.A: "away", official.D: "none", official.X: "abstain"}


def codes_from_labels(values) -> np.ndarray:
    """Map our possession strings (home/away/none/abstain, or H/A/D/X) to official int8 codes."""
    arr = np.asarray(pd.Series(values, dtype=object).fillna("X").astype(str).str.strip())
    out = np.full(len(arr), -1, dtype=np.int8)
    for name, code in TO_CODE.items():
        out[arr == name] = code
    low = np.char.upper(arr.astype(str))
    for name in ("H", "A", "D", "X"):
        out[(out < 0) & (low == name)] = TO_CODE[name]
    if (out < 0).any():
        bad = sorted(set(arr[out < 0]))
        raise ValueError(f"labels outside {sorted(TO_CODE)}: {bad[:5]}")
    return out


def to_code_arrays(pred: pd.DataFrame, gt: pd.DataFrame, gap: int = 1) -> tuple[np.ndarray, np.ndarray, list]:
    """Dense (gt_codes, pred_codes) over every (match_id, period) block of the GT table.

    Within a block the frame axis is ``arange(min_frame, max_frame + 1)``; GT frames absent from the table are
    X (never scored), predicted frames absent from ``pred`` are X (= a miss for macroF1, forward-filled for
    spellF1 — exactly how the official scorer treats a row our model left blank). Blocks are joined with
    ``gap`` X frames in both arrays so the official run detection splits them like the halftime hole.
    Returns the two arrays and ``[(match_id, period, min_frame, offset)]`` to map positions back to frames.
    """
    g = gt[KEY + ["possession"]]
    p = pred[KEY + ["possession"]].drop_duplicates(KEY)
    gts, preds, index = [], [], []
    offset = 0
    for (m, per), block in g.groupby(["match_id", "period"], sort=True):
        fr = block["frame"].to_numpy(np.int64)
        lo, hi = int(fr.min()), int(fr.max())
        n = hi - lo + 1
        gcodes = np.full(n, official.X, dtype=np.int8)
        gcodes[fr - lo] = codes_from_labels(block["possession"])
        pcodes = np.full(n, official.X, dtype=np.int8)
        pb = p[(p["match_id"] == m) & (p["period"] == per)]
        pf = pb["frame"].to_numpy(np.int64)
        keep = (pf >= lo) & (pf <= hi)
        pcodes[pf[keep] - lo] = codes_from_labels(pb["possession"])[keep]
        if gts:
            gts.append(np.full(gap, official.X, dtype=np.int8))
            preds.append(np.full(gap, official.X, dtype=np.int8))
            offset += gap
        index.append((m, int(per), lo, offset))
        gts.append(gcodes)
        preds.append(pcodes)
        offset += n
    return np.concatenate(gts), np.concatenate(preds), index


def json_safe(result: dict) -> dict:
    """``official.evaluate`` output without the per-transition tuples (same as run_evaluation.json_safe)."""
    out = {k: v for k, v in result.items() if k not in ("transitions", "spell_f1")}
    out["spell_f1"] = {str(k): v for k, v in result["spell_f1"].items()}
    out["transitions"] = {str(tol): {k: v for k, v in tr.items() if k not in ("pairs", "missed", "spurious")}
                          for tol, tr in result["transitions"].items()}
    return out


def spell_f1(pred: pd.DataFrame, gt: pd.DataFrame, bootstrap_n: int = 0, full: bool = False) -> dict:
    """Official metrics of ``pred`` against ``gt`` (both tables keyed by match_id, period, frame).

    Returns ``official.evaluate``'s dict (json-safe unless ``full``); the two headline keys are
    ``spell_f1_mean`` (ranking) and ``macro_f1`` (tie-break). ``abstain`` / ``X`` in ``pred`` is honoured.
    """
    gcodes, pcodes, _ = to_code_arrays(pred, gt)
    res = official.evaluate(gcodes, pcodes, bootstrap_n=bootstrap_n)
    return res if full else json_safe(res)


def score(pred: pd.DataFrame, labels: pd.DataFrame, ball_missing: pd.DataFrame | None = None,
          max_lag: int = 125) -> dict:
    """Our per-frame diagnostics (frame.score) + the official ``spellF1`` / ``macroF1`` and spell counts.

    ``selection`` (what model selection optimises) is now the official ranking metric ``spellF1``.
    """
    res = frame_score(pred, labels, ball_missing, max_lag)
    off = spell_f1(pred, labels)
    res["spellF1"] = off["spell_f1_mean"]
    res["macroF1"] = off["macro_f1"]
    res["spellF1_at_0.5"] = off["spell_f1_at_headline"]
    res["n_gt_spells"] = off["n_gt_spells"]
    res["n_pred_spells"] = off["n_pred_spells"]
    res["trans_f1_25"] = off["transitions"]["25"]["f1"]
    res["mean_latency_25"] = off["transitions"]["25"]["mean_latency"]
    res["official"] = off
    res["selection"] = res["spellF1"]
    return res


def to_submission(pred: pd.DataFrame, n_frames: int, frame_col: str = "video_frame") -> pd.DataFrame:
    """``frame,label`` table of exactly ``n_frames`` rows (X where ``pred`` has no row) — the challenge CSV."""
    labels = np.full(n_frames, "X", dtype="<U1")
    vf = pred[frame_col].to_numpy(np.int64)
    codes = codes_from_labels(pred["possession"])
    letters = np.array(list(official.LABELS))[codes]
    ok = (vf >= 0) & (vf < n_frames)
    labels[vf[ok]] = letters[ok]
    return pd.DataFrame({"frame": np.arange(n_frames, dtype=np.int64), "label": labels})


__all__ = ["official", "spell_f1", "score", "frame_score", "to_code_arrays", "codes_from_labels", "to_submission",
           "json_safe", "results_markdown", "save_json", "transition_error", "transitions", "KEY", "TO_CODE",
           "FROM_CODE"]
