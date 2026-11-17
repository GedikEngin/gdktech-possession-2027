"""Internal frame format v0 — schemas, ``MatchMeta`` and validators (see docs/frame-format.md).

Only this module knows the column list; loaders build tables through :func:`make_frames` and everything
downstream calls :func:`validate_frames` / :func:`validate_labels`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# column -> pandas dtype
FRAME_COLUMNS: dict[str, str] = {
    "match_id": "string",
    "period": "int16",
    "frame": "int64",
    "time_s": "float64",
    "team": "string",
    "player_id": "string",
    "x": "float64",
    "y": "float64",
    "z": "float64",
    "ball_status": "string",
}
TEAMS = ("home", "away", "ball")
BALL_STATUS = ("alive", "dead")

LABEL_COLUMNS: dict[str, str] = {
    "match_id": "string",
    "period": "int16",
    "frame": "int64",
    "possession": "string",
    "label_source": "string",
}
POSSESSION = ("home", "away", "none")
LABEL_SOURCES = ("provided", "events-derived")
KEY = ["match_id", "period", "frame"]

# how far outside the marked pitch an object may be before validation complains (metres)
OUT_OF_PITCH_MARGIN_M = 15.0


@dataclass
class MatchMeta:
    """`meta.json` contract. Extra vendor info goes in ``extra`` (never read by other modules)."""

    match_id: str
    source: str
    fps: float
    pitch_length_m: float
    pitch_width_m: float
    home_name: str
    away_name: str
    home_attacks_right: dict[int, bool]
    players: dict[str, dict[str, Any]]
    licence: str
    provenance: dict[str, Any]
    periods: dict[int, dict[str, Any]] = field(default_factory=dict)  # {period: {start_frame, end_frame}}
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["home_attacks_right"] = {str(k): bool(v) for k, v in self.home_attacks_right.items()}
        d["periods"] = {str(k): v for k, v in self.periods.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MatchMeta:
        d = dict(d)
        d["home_attacks_right"] = {int(k): bool(v) for k, v in d["home_attacks_right"].items()}
        d["periods"] = {int(k): v for k, v in d.get("periods", {}).items()}
        d.setdefault("extra", {})
        # top-level keys outside the contract (the m5 pseudo-ATD export writes ``coordinate_note``, ``pipeline``,
        # ``calib_status`` next to the contract fields) are vendor info: keep them under ``extra``
        known = {f.name for f in fields(cls)}
        for k in [k for k in d if k not in known]:
            d["extra"].setdefault(k, d.pop(k))
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False, default=_json_default))

    @classmethod
    def load(cls, path: str | Path) -> MatchMeta:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def attacks_right(self, team: str, period: int) -> bool:
        h = self.home_attacks_right[int(period)]
        return h if team == "home" else (not h)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def make_frames(**columns: Any) -> pd.DataFrame:
    """Build a frames table from per-column arrays/lists, casting to the contract dtypes.

    Missing optional columns (``z``, ``ball_status``) are filled with NaN/NA.
    """
    n = len(columns["frame"])
    data = {}
    for col, dtype in FRAME_COLUMNS.items():
        if col in columns:
            v = columns[col]
        elif col == "z":
            v = np.full(n, np.nan)
        elif col == "ball_status":
            v = pd.array([None] * n, dtype="string")
        else:
            raise KeyError(f"frames table needs column {col!r}")
        if dtype == "string":
            data[col] = (
                v.astype("string")
                if isinstance(v, (pd.Series, pd.api.extensions.ExtensionArray))
                else pd.array(v, dtype="string")
            )
        else:
            data[col] = np.asarray(v).astype(dtype)
    df = pd.DataFrame(data)
    # sort on integer codes (lexsort on string keys costs several GB for a 3M-row match)
    team_code = df["team"].map({"home": 0, "away": 1, "ball": 2}).fillna(3).to_numpy(dtype=np.int8)
    pid_code, _ = pd.factorize(df["player_id"], sort=True)
    order = np.lexsort((pid_code, team_code, df["frame"].to_numpy(), df["period"].to_numpy()))
    return df.take(order).reset_index(drop=True)


def make_labels(match_id: str, period, frame, possession, label_source: str) -> pd.DataFrame:
    n = len(frame)
    df = pd.DataFrame(
        {
            "match_id": pd.array([match_id] * n, dtype="string"),
            "period": np.asarray(period).astype("int16"),
            "frame": np.asarray(frame).astype("int64"),
            "possession": pd.array(np.asarray(possession, dtype=object), dtype="string"),
            "label_source": pd.array([label_source] * n, dtype="string"),
        }
    )
    return df.sort_values(["period", "frame"], kind="stable").reset_index(drop=True)


def validate_frames(frames: pd.DataFrame, meta: MatchMeta | None = None, strict: bool = True) -> list[str]:
    """Check a frames table against the contract. Returns the list of problems (raises if ``strict``)."""
    problems: list[str] = []
    missing = [c for c in FRAME_COLUMNS if c not in frames.columns]
    if missing:
        problems.append(f"missing columns {missing}")
        return _finish(problems, strict)
    for col, dtype in FRAME_COLUMNS.items():
        got = str(frames[col].dtype)
        want = dtype
        ok = (
            (got == want)
            or (want == "string" and got in ("string", "object", "string[python]", "string[pyarrow]"))
            or (want.startswith("int") and got.startswith("int"))
            or (want.startswith("float") and got.startswith("float"))
        )
        if not ok:
            problems.append(f"column {col}: dtype {got}, want {want}")
    bad_team = set(frames["team"].dropna().unique()) - set(TEAMS)
    if bad_team:
        problems.append(f"bad team values {sorted(bad_team)}")
    bad_bs = set(frames["ball_status"].dropna().unique()) - set(BALL_STATUS)
    if bad_bs:
        problems.append(f"bad ball_status values {sorted(bad_bs)}")
    if frames["match_id"].nunique() != 1:
        problems.append(f"expected one match_id, got {frames['match_id'].nunique()}")
    if frames["period"].isna().any() or frames["frame"].isna().any():
        problems.append("NaN in period/frame")
    if (frames["time_s"] < -1e-6).any():
        problems.append("negative time_s")
    # key uniqueness
    dup = frames.duplicated(["period", "frame", "player_id"]).sum()
    if dup:
        problems.append(f"{dup} duplicate (period, frame, player_id) rows")
    # frame monotonic within the match (sorted by period, frame)
    key = frames[["period", "frame"]].drop_duplicates().sort_values(["period", "frame"])
    if not key["frame"].is_monotonic_increasing:
        problems.append("frame index not monotonic across periods")
    # ball row present in every frame
    ball = frames[frames["team"] == "ball"]
    nf = len(key)
    if len(ball) != nf:
        problems.append(f"ball rows {len(ball)} != frames {nf} (ball row must exist in every frame)")
    if (ball["player_id"] != "ball").any():
        problems.append("ball rows must have player_id == 'ball'")
    players = frames[frames["team"] != "ball"]
    if players[["x", "y"]].isna().any().any():
        problems.append("player rows with NaN x/y (absent players must be absent rows)")
    if meta is not None:
        hx, hy = meta.pitch_length_m / 2 + OUT_OF_PITCH_MARGIN_M, meta.pitch_width_m / 2 + OUT_OF_PITCH_MARGIN_M
        out = ((frames["x"].abs() > hx) | (frames["y"].abs() > hy)).sum()
        if out:
            problems.append(f"{out} rows more than {OUT_OF_PITCH_MARGIN_M} m outside the pitch")
        per = set(frames["period"].unique()) - set(meta.home_attacks_right)
        if per:
            problems.append(f"periods {sorted(per)} missing in meta.home_attacks_right")
        if frames["match_id"].iloc[0] != meta.match_id:
            problems.append("match_id differs between frames and meta")
        unknown = set(players["player_id"].unique()) - set(meta.players)
        if unknown:
            problems.append(f"{len(unknown)} player ids not in meta.players, e.g. {sorted(unknown)[:3]}")
    return _finish(problems, strict)


def validate_labels(labels: pd.DataFrame, frames: pd.DataFrame | None = None, strict: bool = True) -> list[str]:
    problems: list[str] = []
    missing = [c for c in LABEL_COLUMNS if c not in labels.columns]
    if missing:
        problems.append(f"missing columns {missing}")
        return _finish(problems, strict)
    bad = set(labels["possession"].dropna().unique()) - set(POSSESSION)
    if bad:
        problems.append(f"bad possession values {sorted(bad)}")
    if labels["possession"].isna().any():
        problems.append("NaN possession")
    bad = set(labels["label_source"].unique()) - set(LABEL_SOURCES)
    if bad:
        problems.append(f"bad label_source values {sorted(bad)}")
    if labels.duplicated(KEY).any():
        problems.append("duplicate (match_id, period, frame)")
    if frames is not None:
        fk = frames[KEY].drop_duplicates()
        m = fk.merge(labels[KEY], how="outer", indicator=True)["_merge"].value_counts()
        if m.get("left_only", 0) or m.get("right_only", 0):
            problems.append(
                f"labels/frames key mismatch: frames-only {m.get('left_only', 0)}, labels-only {m.get('right_only', 0)}"
            )
    return _finish(problems, strict)


def _finish(problems: list[str], strict: bool) -> list[str]:
    if problems and strict:
        raise ValueError("; ".join(problems))
    return problems
