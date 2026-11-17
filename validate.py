#!/usr/bin/env python
"""Validate a possession CSV exactly as the official scorer does (and optionally score it against a GT CSV).

    python validate.py out/<team>.csv [--rows 199319] [--gt evaluation/gt/BAR-ATH_possession_gt_public.csv]

Uses the organisers' ``possession_metrics.py`` (vendored unmodified in ``gdkb/scoring/official.py``), so a file that
passes here passes ``evaluation/validate_submission.py`` of metrica-sports/more-than-a-hack-2027.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdkb.scoring import official  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--rows", type=int, default=199319)
    ap.add_argument("--gt", type=Path)
    a = ap.parse_args()
    frame = official.read_label_csv(a.csv)
    problem = official.validate_submission(frame, a.rows)
    sha = hashlib.sha256(a.csv.read_bytes()).hexdigest()
    if problem:
        print(f"INVALID: {problem}  ({a.csv}, sha256 {sha})")
        return 1
    codes = official.label_codes(frame)
    counts = {lab: int((codes == i).sum()) for i, lab in enumerate(official.LABELS)}
    print(f"OK: {len(frame):,} rows, labels {counts}, sha256 {sha}")
    if a.gt:
        gt = official.label_codes(official.read_label_csv(a.gt))
        res = official.evaluate(gt, codes)
        print(f"spellF1 {res['spell_f1_mean']:.4f}  macroF1 {res['macro_f1']:.4f}  spells "
              f"{res['n_pred_spells']}/{res['n_gt_spells']}  graded {res['n_graded']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
