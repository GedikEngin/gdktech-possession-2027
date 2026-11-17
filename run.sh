#!/usr/bin/env bash
# One command: ./run.sh /path/to/bundle out/<team_name>.csv
#
# Creates ./.venv with the pinned dependencies (requirements.txt) if it does not exist, then regenerates the
# possession CSV from the organisers' bundle. CPU only, deterministic (2 threads; torch runs on 1 thread on macOS),
# about 10-25 s and 2.0-2.4 GB RAM (allow 3 GB) on a 199,319-frame match (bi-GRU segmenter on 2 CPU threads; --model gbm for the LightGBM-only path). Extra arguments after the two paths go to predict.py (e.g. --flat --spec ... --gt ...).
#
# Environment: PYTHON (interpreter used to build the venv when uv is absent; default python3.12, then python3),
#              GDKB_VENV (venv path, default ./.venv).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -lt 2 ]; then
  echo "usage: $0 <bundle_dir> <out.csv> [predict.py options]" >&2
  exit 64
fi
BUNDLE="$1"; OUT="$2"; shift 2
VENV="${GDKB_VENV:-$HERE/.venv}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "[run.sh] creating $VENV" >&2
  if command -v uv >/dev/null 2>&1; then
    uv venv -p 3.12 "$VENV" -q
    # unsafe-best-match: torch==2.14.1+cpu lives on the pytorch CPU index named in requirements.txt, not on PyPI
    if [ "$(uname -s)" = "Darwin" ]; then
      # macOS has no torch==X+cpu wheel: install everything else hash-pinned, then the plain PyPI torch of the same version
      TMPREQ="$(mktemp)"
      awk '/^torch==/{skip=1; next} skip && /^[ \t]/{next} {skip=0; print}' "$HERE/requirements.txt" > "$TMPREQ"
      uv pip install -q -p "$VENV/bin/python" -r "$TMPREQ" --index-strategy unsafe-best-match
      uv pip install -q -p "$VENV/bin/python" "torch==$(sed -n 's/^torch==\([0-9.]*\)+cpu.*/\1/p' "$HERE/requirements.txt")"
      rm -f "$TMPREQ"
    else
      uv pip install -q -p "$VENV/bin/python" -r "$HERE/requirements.txt" --index-strategy unsafe-best-match
    fi
  else
    PY="${PYTHON:-}"
    if [ -z "$PY" ]; then
      if command -v python3.12 >/dev/null 2>&1; then PY=python3.12; else PY=python3; fi
    fi
    "$PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install -q --upgrade pip
    if [ "$(uname -s)" = "Darwin" ]; then
      TMPREQ="$(mktemp)"
      awk '/^torch==/{skip=1; next} skip && /^[ \t]/{next} {skip=0; print}' "$HERE/requirements.txt" > "$TMPREQ"
      "$VENV/bin/python" -m pip install -q -r "$TMPREQ"
      "$VENV/bin/python" -m pip install -q "torch==$(sed -n 's/^torch==\([0-9.]*\)+cpu.*/\1/p' "$HERE/requirements.txt")"
      rm -f "$TMPREQ"
    else
      "$VENV/bin/python" -m pip install -q -r "$HERE/requirements.txt"
    fi
  fi
fi

# determinism / resource caps (LightGBM inference is thread-count independent; this keeps the box quiet)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
exec "$VENV/bin/python" "$HERE/predict.py" "$BUNDLE" "$OUT" "$@"
