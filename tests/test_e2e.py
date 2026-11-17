"""End-to-end smoke test of the public repository on the synthetic fixture (no match data needed).

``./run.sh`` twice on ``tests/fixtures/challenge_mini`` -> byte-identical CSVs, the official validator accepts them,
the score against the fixture's ground truth is the one recorded when the models were frozen.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIX = HERE / "fixtures" / "challenge_mini"
# PIPE6: 5-seed GRU ensemble (class probabilities averaged before decode); the single seed-0 model gave 7553a425...
FIXTURE_SHA256 = "6c35e15e835a5e665f205859bd9f11b3afab2f3a837e365455d4a6795bc92f54"
FIXTURE_SHA256_SEED0 = "7553a425cbae3cb391482147b16de2dd062de0489e2463e4c6ba3ff6e0f964a7"


def _run(out: Path, timing: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(ROOT / "predict.py"), str(FIX), str(out), "--flat", "--game", "MINI", "--spec",
           str(FIX / "spec.json"), "--gt", str(FIX / "MINI_possession_gt.csv"), "--timing-json", str(timing), *extra]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "PYTHONHASHSEED": "0"}
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)


def test_fixture_twice_byte_identical(tmp_path: Path):
    shas, scores = [], []
    for i in (1, 2):
        out, timing = tmp_path / f"run{i}.csv", tmp_path / f"run{i}.json"
        r = _run(out, timing)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "validator: OK" in r.stdout
        shas.append(hashlib.sha256(out.read_bytes()).hexdigest())
        t = json.loads(timing.read_text())
        assert t["sha256"] == shas[-1]
        scores.append(t["score"])
        lines = out.read_text().splitlines()
        assert lines[0] == "frame,label" and len(lines) == 701
    assert shas[0] == shas[1] == FIXTURE_SHA256
    assert scores[0] == scores[1]
    assert abs(scores[0]["spell_f1_mean"] - 0.8571) < 1e-3 and abs(scores[0]["macro_f1"] - 0.8984) < 1e-3


def test_validator_cli(tmp_path: Path):
    out = tmp_path / "v.csv"
    assert _run(out, tmp_path / "v.json").returncode == 0
    r = subprocess.run([sys.executable, str(ROOT / "validate.py"), str(out), "--rows", "700", "--gt",
                        str(FIX / "MINI_possession_gt.csv")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and r.stdout.startswith("OK: 700 rows"), r.stdout + r.stderr
    bad = tmp_path / "bad.csv"
    bad.write_text("frame,label\n0,H\n1,Z\n")
    r = subprocess.run([sys.executable, str(ROOT / "validate.py"), str(bad), "--rows", "2"], capture_output=True,
                       text=True, cwd=ROOT)
    assert r.returncode == 1 and r.stdout.startswith("INVALID")


def test_models_match_manifest():
    spec = json.loads((ROOT / "models" / "pipeline.json").read_text())
    for key, name in spec["files"].items():
        assert hashlib.sha256((ROOT / "models" / name).read_bytes()).hexdigest() == spec["sha256"][key], name
    extra = spec.get("seq_extra") or []
    assert len(extra) == 4  # PIPE6: seed 0 (files.seq) + 4 extra GRU seeds, probabilities averaged before decode
    for e in extra:
        assert hashlib.sha256((ROOT / "models" / e["file"]).read_bytes()).hexdigest() == e["sha256"], e["file"]


def test_gap_guard_flag(tmp_path: Path):
    """``--gap-guard on`` (the Oct 20 switch, docs/submission/READINESS.md) runs, validates and keeps the labels a
    valid submission; ``off`` reproduces the shipped file (the export ships without a guard)."""
    on, off = tmp_path / "on.csv", tmp_path / "off.csv"
    r = _run(on, tmp_path / "on.json", "--gap-guard", "on", "--gap-min-s", "3")
    assert r.returncode == 0 and "gap guard on (min 3 s)" in r.stdout and "validator: OK" in r.stdout, r.stdout + r.stderr
    r = _run(off, tmp_path / "off.json", "--gap-guard", "off")
    assert r.returncode == 0 and "gap guard off" in r.stdout, r.stdout + r.stderr
    assert hashlib.sha256(off.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert len(on.read_text().splitlines()) == 701


def test_single_model_pack_still_loads(tmp_path: Path):
    """An older pack (no ``seq_extra`` in pipeline.json) is a single GRU: same loader, the seed-0 fixture file."""
    import shutil

    models = tmp_path / "models"
    shutil.copytree(ROOT / "models", models, ignore=shutil.ignore_patterns("seq_model_s*.npz"))
    spec = json.loads((models / "pipeline.json").read_text())
    spec.pop("seq_extra")
    (models / "pipeline.json").write_text(json.dumps(spec))
    out = tmp_path / "single.csv"
    r = _run(out, tmp_path / "single.json", "--models", str(models))
    assert r.returncode == 0, r.stdout + r.stderr
    assert hashlib.sha256(out.read_bytes()).hexdigest() == FIXTURE_SHA256_SEED0 != FIXTURE_SHA256
