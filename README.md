# gdkBarcelona — Talent Arena 2027 "More Than a Hack", Initial Challenge

Per-frame team possession (`H` / `A` / `D` / `X`) for the tactical video of FC Barcelona vs Athletic Club from the
provided Metrica automatic tracking data (ATD) and Smart Tagging. This repository regenerates our submission file
`<team_name>.csv` from the organisers' bundle with one command (SUBMISSION.md §2 of
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)).

**The match data is not in this repository and must not be added to it** (RULES.md §"The data").

## Run

```bash
./run.sh /path/to/bundle out/<team_name>.csv
```

* `/path/to/bundle` is the organisers' data folder: `<bundle>/tactical/BAR-ATH_tactical_{atd.xml,atd.txt,
  homography.csv,smart_tagging.xml}` (DATA.md §"Getting it"; the videos and the broadcast feed are not read).
* `run.sh` creates `./.venv` from `requirements.txt` on first use (uses [`uv`](https://docs.astral.sh/uv/) when
  installed, else `python3.12 -m venv` + `pip`), then runs `predict.py`. Nothing is downloaded at run time except
  the pinned wheels (hashes in `requirements.txt`; `torch==2.14.1+cpu` comes from the PyTorch CPU index named in
  that file — ~190 MB, no CUDA; `uv` resolves it with `--index-strategy unsafe-best-match`, which `run.sh` sets).
* Requirements: Python 3.12, CPU only. Linux x86_64 uses the pinned CPU torch wheel. On macOS (tested: Apple
  silicon, Python 3.12.16) `run.sh` detects Darwin and installs the same torch version (2.14.1) from PyPI instead,
  because no `+cpu` wheel exists there; everything else is the same hash-pinned set (torch on macOS is installed by version from PyPI, not by hash). On macOS the GRU runs on one
  torch thread (torch and LightGBM bundle separate OpenMP runtimes, `predict.py` sets `KMP_DUPLICATE_LIB_OK`);
  the output does not depend on the thread count. Our submitted file was generated on macOS arm64 (Python 3.12.16);
  the same bundle run on Linux x86_64 (WSL2 Ubuntu, Python 3.12.16, torch 2.14.1+cpu) gives a byte-identical CSV.
* Measured cost on the 199,319-frame match (five GRU forward passes, ≈ 17 s of the total): 21–25 s (25 s cold, then 21 and 21 s) and 2.38–2.43 GB peak RAM on
  Apple silicon; 52 s and 2.0–2.1 GB on Linux x86_64 CPU (allow 3 GB; EPTS parse ≈ 3–6 s). `--model gbm`
  (after the two paths) switches to the LightGBM-only segmenter (no torch import; time not re-measured); it is the
  fallback, not what the submission file used.
* The last line printed is the row count, label counts, the official validator verdict and the file's sha256.
  `python validate.py out/<team_name>.csv` re-runs the validator alone (`--gt <csv>` also scores it).

Docker (CPU, python 3.12):

```bash
docker build -t gdkb-submission .
docker run --rm -v /path/to/bundle:/data:ro -v "$PWD/out":/out gdkb-submission /data /out/<team_name>.csv
```

Smoke test without match data (synthetic 700-frame fixture in the same file formats):

```bash
make test            # ./run.sh twice on tests/fixtures/challenge_mini -> byte-identical, validator OK
```

## Determinism

Two runs on the same bundle give byte-identical CSVs: the parser is streaming numpy, the LightGBM boosters are
loaded from text (`models/*.lgb.txt`, sha256 in `models/pipeline.json`) and inference is thread-count
independent, the Viterbi decoder and every post-processing step are deterministic, and the CSV is written with
`\n` line endings; the GRU weights are float32 arrays loaded from `models/seq_model.npz` and run on 2 CPU threads
(`torch.set_num_threads(2)`; 1 on macOS, where torch and LightGBM ship separate OpenMP runtimes that cannot share a
thread team; the output does not depend on the thread count). Identity is proven on two platforms: the real-bundle
CSV (sha256 9f52a6e2…1e00) is byte-identical on macOS arm64 and on Linux x86_64 (2026-11-17, five-GRU ensemble: three Mac runs from a fresh `git archive` of commit 158cf2b with a venv built by `run.sh`; Linux runs in an earlier pinned venv with the identical `requirements.txt` (three runs were made, default threads twice and 16 OpenMP threads, all giving this sha256, but only the log of one is kept); Python 3.12.16, torch 2.14.1+cpu, LightGBM 4.7.0, numpy 2.5.3). The evidence is in the technical document.

## What it does

`predict.py`: bundle → EPTS parse (`gdkb.loaders.challenge`, graded blocks from DATA.md: first half 11185–90651,
second half 110808–190197, kick-off 113742) → per-frame team-level features that tolerate anonymous `Track N`
slots and the missing ball (`gdkb.possession.features`, 32 columns: ball cues plus eight player-only cues — team
centroid motion, spread, player speeds, share of standing players — that stay informative while the ball is lost)
→ **bi-GRU segmenter** (a residual dilated-convolution front-end — kernel 5, dilations 1, 2, 4, 8, about 2.4 s receptive field — then 2 × 128 GRU units over the 32 standardised per-frame features; five independently seeded copies, `models/seq_model.npz` + `seq_model_s1..s4.npz`, whose home / away / none probabilities are averaged per frame) → sticky **Viterbi** (stay 0.95) → **dead-ball channel** = the segmenter's own
`none` class (`D` runs shorter than 50 frames take the carried team label; **no separate dead-ball model and no
Smart-Tagging feature**) → **boundary refiner** (each hand-over re-timed to the receiving team's nearest control
onset within ±12 frames) → **fragmentation control** (team spells < 20 frames merged into the previous spell) →
**hard rules** (`D` on 110808–113741, last label carried over any abstention inside a graded block, `X` only
outside the graded blocks) → `frame,label` CSV of 199,319 rows → the organisers' validator
(`gdkb/scoring/official.py` = `evaluation/possession_metrics.py`, vendored unmodified, MIT). A long-gap guard
(`D` inside a ≥ 3 s ball-missing stretch only on explicit dead-ball evidence) is implemented in
`gdkb.possession.postprocess` and configured **off** in `models/pipeline.json`; the Smart Tagging XML is read
but unused.

## How the models were trained (not needed to reproduce the CSV)

The models in `models/` (bi-GRU segmenter, LightGBM fallback segmenter) were trained on openly licensed tracking
data of ten other matches only (a first set was frozen on 2026-10-08, before the challenge data was released; it was
replaced by a set retrained on six matches on 2026-11-16 (commit 93ac8ac) after a self-audit against the 3 Nov clarification,
and that by the present ten-match set on 2026-11-17 (seed 0 frozen in commit dfbea9b, seeds 1-4 and the ensemble in commit 158cf2b; models frozen before the challenge file was written));
the challenge bundle is only ever read by `predict.py`. No weight, threshold or post-processing constant was fitted
on the provided match's video, tracking labels or ground truth, including the public 5-minute window; the only
things taken from the provided match are summary statistics of the training-time degradation simulator (below).
The public window was scored as a record on the submitted file (the five-GRU ensemble) after the weights were frozen (commit 158cf2b), by one call whose command, timestamp (2026-11-17 08:48 MST) and output are logged in our run records (spellF1 0.4625, macroF1 0.7259, 15 / 17 spells); no choice used the score. An earlier single-model candidate file (not the submitted one) had been scored in the same way by one call that we did not log.

| training match | source | licence |
|---|---|---|
| Sample Game 1, 2, 3 | Metrica Sports sample data, https://github.com/metrica-sports/sample-data | no licence file; README "Legal stuff": be responsible, acknowledge the source (acknowledged here) |
| DFL-MAT-J03WMX, J03WN1, J03WPY, J03WOH, J03WQQ, J03WOY, J03WR9 | IDSSE / DFL open Bundesliga dataset, Bassek et al. 2025, https://doi.org/10.6084/m9.figshare.28196177 (paper: https://doi.org/10.1038/s41597-025-04505-y) | CC BY 4.0, attribution given here and in the technical document |

Labels: a "control" rule over the vendors' events (Metrica) / per-frame `BallPossession` + `BallStatus` (DFL)
that follows the text of CHALLENGE.md §"Possession rules": controlled on-ball events (including a tagged recovery)
flip possession, touches, duels and deflections do not; spells shorter than 5 frames
are merged, a constant chosen by a leave-one-match-out rule pre-registered in commit 660638f. Tracking
was degraded to look like the challenge ATD before training (ball hidden on 41 % of frames, alive and dead alike —
the degradation is independent of the dead/alive state — in log-normal stretches, median 10.7 s, 90th percentile
34 s, up to 150 s, half of them starting or ending at a stoppage; 46 anonymous slots with breaks, swaps,
duplicates, rates matched to the object counts in DATA.md). The 41 % and the object counts come from the organisers' DATA.md; the stretch statistics (median 10.7 s, p90 34 s, 150 stretches, maximum 221 s) come from the organisers' public `evaluation/results/baseline_raw.csv` ball-missing mask. They describe the ball
channel of the provided file, not possession. An earlier version carried a separate dead-ball model with simulated Smart-Tagging windows; our audit
found they leaked the restart times of the evaluation matches (the simulated windows revealed when play restarted in the
held-out matches), and the honest version drifts to `D` inside long alive ball-missing stretches, so the shipped file uses the
segmenter's own dead-ball class only. bi-GRU: PyTorch, 2 layers × 128 units, AdamW, 20 s training windows, 6 epochs on all ten matches, a **5-seed ensemble** (seeds 0-4, same data and configuration; the five models' class probabilities are averaged per frame before the unchanged decoding; five forward passes at inference), an auxiliary spell-start output used only during training (not read at inference), trained on
a desktop GPU (RTX 4070 Ti SUPER), inference on CPU. LightGBM: `deterministic=True`, `force_row_wise=True`, `seed=0`, 2 threads; 300
rounds / 153 features (fallback segmenter). Training code, data converters and the
leave-one-match-out study live in our development repository; the model card is reproduced in the technical
document. Nothing in this repository or in the training path uses GPL/AGPL code or non-commercial weights
(`THIRD_PARTY_NOTICES.md`). No video, detector or tracker is involved in the Initial Challenge file.

## Layout

```
run.sh  predict.py  validate.py  Makefile  Dockerfile  requirements.txt  pyproject.toml
gdkb/            inference package (loaders/challenge.py, possession/{features,model,postprocess,rules,seq}.py,
                 scoring/official.py = the organisers' metrics, formats.py, pitch.py)
models/          seq_model.npz + seq_model_s1..s4.npz (the five bi-GRU seeds, averaged before decoding; seed 0 sha256 706a38a3…; a pack without `seq_extra` in pipeline.json is a single model), segmenter.lgb.txt (fallback, sha256 81e5e596…),
                 pipeline.json (columns, GRU config, post-processing config incl. the long-gap guard = off, sha256 of
                 the model files, meta)
tests/           fixture end-to-end test + the synthetic fixture (tests/fixtures/challenge_mini)
```

## Licence

Our code: Apache-2.0 (`LICENSE`). Third-party code and data: `THIRD_PARTY_NOTICES.md`. The match data stays
under the organisers' terms and is not part of this repository.
