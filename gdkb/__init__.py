"""gdkb — inference subset of the gdkBarcelona toolkit (Talent Arena 2027 "More Than a Hack", Initial Challenge).

Only what ``predict.py`` needs to turn the provided bundle into the possession CSV:

- ``gdkb.loaders.challenge``: EPTS metadata/raw parser, Smart Tagging reader, graded blocks (``GameSpec``),
  hard rules and the ``frame,label`` writer.
- ``gdkb.possession.features`` / ``.model`` / ``.postprocess``: per-frame features, windowed features + LightGBM
  segmenter + Viterbi smoothing, dead-ball detector, boundary refiner, fragmentation control.
- ``gdkb.scoring.official``: the organisers' ``possession_metrics.py`` (MIT, vendored unmodified) for the validator
  and the public-window score.

Every module here is a byte-identical copy of the development repository's module (``scripts/sync_submission.py``).
"""

__version__ = "1.0.0"
