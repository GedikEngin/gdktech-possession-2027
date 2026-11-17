#!/usr/bin/env python
"""Bundle -> possession CSV. The one entry point of the public repository (``run.sh`` calls it).

    python predict.py <bundle_dir> <out.csv> [--game BAR-ATH] [--source tactical] [--models models]
                      [--spec spec.json] [--flat] [--gt <gt.csv>] [--timing-json <path>]
                      [--gap-guard on|off [--gap-min-s 60]]

``<bundle_dir>`` is the organisers' data folder: ``<bundle_dir>/tactical/BAR-ATH_tactical_{atd.xml,atd.txt,
homography.csv,smart_tagging.xml}`` (DATA.md §"Getting it"; ``--flat`` when the files sit directly in the folder, as
in ``tests/fixtures/challenge_mini``). The graded blocks, kick-offs and pitch come from ``GameSpec``: the BAR-ATH
constants of DATA.md by default, or ``--spec`` for another game.

Pipeline (deterministic, CPU, 2 threads): EPTS parse -> per-frame features (team-level ball cues + player-only
cues) -> segmenter: **bi-GRU** on the standardised per-frame features (``models/seq_model.npz``, default) or
LightGBM on windowed features (``--model gbm``, no torch needed) -> sticky Viterbi -> dead-ball detector (LightGBM
on ball cues; the shipped one uses no Smart-Tagging feature) merged as a union -> long-gap guard (D inside a
ball-missing stretch only on explicit dead-ball evidence) -> boundary refiner (nearest control onset) ->
fragmentation control (team spells < 20 frames merged) -> hard rules (forced D before the 2nd-half kick-off, last
label carried, X only outside the graded blocks) -> ``frame,label`` CSV -> official validator (-> official scorer
when ``--gt`` is given).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

# macOS wheels: torch and lightgbm each bundle a libomp; the second one to load aborts (OMP Error #15). Both only run
# fixed-order CPU kernels here, and run-twice byte-identity is checked, so allow the duplicate runtime on Darwin.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm as lgb  # noqa: E402
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gdkb.loaders import challenge as ch  # noqa: E402
from gdkb.possession import postprocess as pp  # noqa: E402
from gdkb.possession.features import KEY, frame_features  # noqa: E402
from gdkb.possession.model import GBM, windowed  # noqa: E402
from gdkb.scoring import official  # noqa: E402


def load_models(models_dir: Path, model: str | None = None) -> dict:
    """``pipeline.json`` + text / npz models -> {model, gbm, seq, seqs, ens_alpha, cfg, dead_model, meta}.

    ``model``: ``seq`` (bi-GRU, needs torch), ``gbm`` (LightGBM only) or ``ens``; default = what the export says."""
    spec = json.loads((models_dir / "pipeline.json").read_text())
    model = model or spec.get("model", "gbm")
    if model in ("seq", "ens") and not spec.get("seq"):
        raise SystemExit("pipeline.json carries no sequence model; use --model gbm")
    gbm = GBM()
    gbm.model = lgb.Booster(model_str=(models_dir / spec["files"]["segmenter"]).read_text())
    gbm.cols = list(spec["segmenter"]["cols"])
    seq, seq_all = None, []
    if model in ("seq", "ens"):
        from gdkb.possession.seq import SeqModel

        seq = SeqModel.from_npz(models_dir / spec["files"]["seq"], spec["seq"])
        # PIPE6 seed ensemble: optional extra members (``seq_extra`` in pipeline.json); a pack without it is a single model
        seq_all = [seq] + [SeqModel.from_npz(models_dir / e["file"], e["spec"]) for e in spec.get("seq_extra") or []]
    cfg = pp.PostprocessConfig.from_dict(spec["postprocess"])
    dead_model = None
    if spec.get("dead_model"):
        dm = spec["dead_model"]
        dead_model = pp.DeadBallModel(threshold=dm["threshold"], min_run_frames=dm["min_run_frames"])
        dead_model.model = lgb.Booster(model_str=(models_dir / spec["files"]["dead_model"]).read_text())
        dead_model.cols = list(dm["cols"])
    return {"model": model, "gbm": gbm, "seq": seq, "seqs": seq_all, "ens_alpha": spec.get("ens_alpha"), "cfg": cfg,
            "dead_model": dead_model, "meta": spec.get("meta", {}),
            "uses_smart_tagging": bool((spec.get("dead_model") or {}).get("uses_smart_tagging", True))}


def segment(models: dict, feat: pd.DataFrame, timing: dict) -> pd.DataFrame:
    """Segmenter probabilities (KEY + p_home/p_away/p_none) from the GRU, the GBM or their blend."""
    kind = models["model"]
    probs = models["gbm"].predict(windowed(feat)) if kind in ("gbm", "ens") else None
    if kind in ("seq", "ens"):
        import torch

        from gdkb.possession.seq import predict_seq_ensemble

        # Darwin: torch and LightGBM each bundle a libomp; with both loaded, a multi-thread OpenMP team in torch deadlocks
        # at its join barrier once LightGBM has run, so torch uses one thread there. (Output is thread-count independent.)
        torch.set_num_threads(1 if sys.platform == "darwin" else 2)
        ts = time.time()
        sp_ = predict_seq_ensemble(models.get("seqs") or [models["seq"]], feat, device="cpu")
        timing["seq_model"] = time.time() - ts
        if kind == "ens":
            al = float(models["ens_alpha"])
            for c in ("p_home", "p_away", "p_none"):
                probs[c] = al * sp_[c].to_numpy(np.float64) + (1 - al) * probs[c].to_numpy(np.float64)
        else:
            probs = sp_
    return probs


def set_piece_table(tagging: pd.DataFrame | None, spec: ch.GameSpec, match_id: str) -> pd.DataFrame | None:
    """Smart Tagging SET PIECES -> (match_id, period, start_frame, end_frame) in vendor frame units (video + 1)."""
    if tagging is None or tagging.empty:
        return None
    sp = ch.set_pieces(tagging)
    if sp.empty:
        return None
    start = sp["start_frame"].to_numpy(np.int64)
    end = sp["end_frame"].to_numpy(np.int64)
    per = ch.period_of_video_frame(np.maximum(start, 0), spec)
    # a window opening before a graded block (kick-off) belongs to the block it ends in
    per2 = ch.period_of_video_frame(np.maximum(end - 1, 0), spec)
    per = np.where(per > 0, per, per2)
    out = pd.DataFrame({"match_id": match_id, "period": per, "start_frame": start + 1, "end_frame": end + 1,
                        "type": sp["type"].to_numpy(object), "team": sp["team"].to_numpy(object)})
    return out[out["period"] > 0].reset_index(drop=True)


def rss_mb() -> float:
    # ru_maxrss is bytes on macOS, KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path, help="data folder with <source>/<game>_<source>_*.{xml,txt,csv}")
    ap.add_argument("out", type=Path, help="output CSV (name it <team_name>.csv)")
    ap.add_argument("--game", default="BAR-ATH")
    ap.add_argument("--source", default="tactical")
    ap.add_argument("--models", type=Path, default=HERE / "models")
    ap.add_argument("--model", choices=["seq", "gbm", "ens"], default=None,
                    help="segmenter: seq = bi-GRU (default when exported), gbm = LightGBM fallback (no torch)")
    ap.add_argument("--spec", type=Path, help="GameSpec json (default: the BAR-ATH constants)")
    ap.add_argument("--flat", action="store_true", help="files sit directly in the bundle folder (fixture layout)")
    ap.add_argument("--gt", type=Path, help="ground-truth CSV to score against (e.g. the public 5-minute window)")
    ap.add_argument("--timing-json", type=Path, help="write stage timings, peak RSS and the output sha256 here")
    ap.add_argument("--gap-guard", choices=["on", "off"], default=None,
                    help="override the export's long-gap guard (D inside a ball-missing stretch only on explicit "
                         "dead-ball evidence); default = what models/pipeline.json says (shipped: off)")
    ap.add_argument("--gap-min-s", type=float, default=60.0,
                    help="guarded stretch length in seconds when --gap-guard on creates the guard (default 60)")
    a = ap.parse_args()

    t0 = time.time()
    timing: dict = {}
    spec = ch.GameSpec.load(a.spec) if a.spec else ch.BAR_ATH
    models = load_models(a.models, a.model)
    if a.gap_guard == "off":
        models["cfg"].gap_guard = None
    elif a.gap_guard == "on" and models["cfg"].gap_guard is None:
        models["cfg"].gap_guard = pp.GapGuardConfig(min_gap_s=a.gap_min_s)
    dchan = ("no separate D model (segmenter's own dead class)" if models["dead_model"] is None else
             f"D model {'with' if models['uses_smart_tagging'] else 'without'} Smart-Tagging features")
    gg = models["cfg"].gap_guard
    print(f"models: segmenter={models['model']}, {dchan}, gap guard "
          f"{f'on (min {gg.min_gap_s:g} s)' if gg else 'off'}", flush=True)
    paths = ch.asset_paths(a.bundle, a.game, a.source)
    if a.flat:
        paths = {k: a.bundle / v.name for k, v in paths.items()}
    for k in ("atd_meta", "atd_raw"):
        if not paths[k].exists():
            print(f"missing {paths[k]}", file=sys.stderr)
            return 2
    frames, meta = ch.load_atd(paths["atd_meta"], paths["atd_raw"], spec, a.source)
    timing["load_atd"] = time.time() - t0
    print(f"frames: {len(frames):,} rows, {meta.extra['n_records']:,} records, {meta.extra['n_slots']} slots, "
          f"home={meta.home_name!r} away={meta.away_name!r} ({timing['load_atd']:.0f} s, {rss_mb():.0f} MB)",
          flush=True)
    tagging = None
    if "smart_tagging" in paths and paths["smart_tagging"].exists():
        tagging = ch.load_smart_tagging(paths["smart_tagging"], spec, meta.home_name, meta.away_name)
        print(f"smart tagging: {len(tagging)} instances, unmapped teams {tagging.attrs['unmapped_teams']}")
    else:
        print("smart tagging: none (D detector runs on ball cues only)")

    t1 = time.time()
    feat = frame_features(frames, meta.to_dict())
    del frames
    timing["features"] = time.time() - t1
    print(f"features: {len(feat):,} frames ({timing['features']:.0f} s, {rss_mb():.0f} MB)", flush=True)

    t1 = time.time()
    probs = segment(models, feat, timing)
    timing["segmenter"] = time.time() - t1
    t1 = time.time()
    cfg = models["cfg"]
    sp = set_piece_table(tagging, spec, meta.match_id)
    pred = pp.postprocess(probs, feat, cfg, sp, models["dead_model"])
    timing["postprocess"] = time.time() - t1
    n_sp = 0 if sp is None else len(sp)
    print(f"segmenter ({models['model']}) {timing['segmenter']:.0f} s; post-processing ({cfg.dead_mode} D, refine "
          f"{cfg.refine.strategy if cfg.refine else 'off'}, min spell {cfg.min_spell_frames}, {n_sp} set-piece "
          f"windows) {timing['postprocess']:.0f} s; {pp.spell_rate(pred):.2f} spells/min ({rss_mb():.0f} MB)",
          flush=True)

    pred = pred[KEY + ["possession"]].copy()
    pred["video_frame"] = pred["frame"] - 1
    sub = ch.submission(pred, spec)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(a.out, index=False, lineterminator="\n")
    problem = official.validate_submission(sub, spec.n_video_frames)
    counts = {k: int(v) for k, v in sub["label"].value_counts().to_dict().items()}
    timing["total"] = time.time() - t0
    timing["max_rss_mb"] = rss_mb()
    import hashlib

    timing["sha256"] = hashlib.sha256(a.out.read_bytes()).hexdigest()
    timing["labels"] = counts
    print(f"wrote {a.out} ({len(sub):,} rows, {counts}); validator: {problem or 'OK'}; sha256 {timing['sha256']} "
          f"({timing['total']:.0f} s total, max RSS {timing['max_rss_mb']:.0f} MB)")
    if a.gt:
        gt = official.label_codes(official.read_label_csv(a.gt))
        res = official.evaluate(gt, official.label_codes(sub))
        t25 = res["transitions"][25]
        print(f"score vs {a.gt.name}: spellF1 {res['spell_f1_mean']:.4f} macroF1 {res['macro_f1']:.4f} "
              f"spells {res['n_pred_spells']}/{res['n_gt_spells']} spellF1@0.7 {res['spell_f1'][0.7]['f1']:.4f} "
              f"mean|lat|@25 {t25['mean_abs_latency']} graded {res['n_graded']}")
        timing["score"] = {"spell_f1_mean": res["spell_f1_mean"], "macro_f1": res["macro_f1"],
                           "n_pred_spells": res["n_pred_spells"], "n_gt_spells": res["n_gt_spells"],
                           "spell_f1_07": res["spell_f1"][0.7]["f1"], "mean_abs_latency_25": t25["mean_abs_latency"]}
    if a.timing_json:
        a.timing_json.parent.mkdir(parents=True, exist_ok=True)
        a.timing_json.write_text(json.dumps(timing, indent=1, default=float) + "\n")
    return 1 if problem else 0


if __name__ == "__main__":
    sys.exit(main())
