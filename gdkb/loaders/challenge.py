"""Challenge bundle loader (Oct 20) — written blind from docs/spec/data-expectations.md (P2.trainshape step 3).

Files (DATA.md §"Getting it"): ``<game>_<source>_atd.xml`` + ``_atd.txt`` (Metrica FIFA/EPTS export),
``_homography.csv``, ``_smart_tagging.xml`` (tactical only). Everything the submission depends on is a
:class:`GameSpec` (video length, graded half blocks, kick-offs, pitch) so the same code runs on the synthetic
fixture ``tests/fixtures/challenge_mini/`` (its own ``spec.json``) and on the real bundle (``BAR_ATH``).

Conventions (docs/frame-format.md + public-gt.md §4):

* frames table = the internal contract plus ``video_frame`` (= ``frame_id - 1`` = submission row),
  ``x_norm``/``y_norm`` (the file's 0–1 values, what the homography wants) and ``speed`` (m/s, NaN if the export
  has none). ``frame`` stays the vendor ``frame_id``. ``period`` = the graded half block the video frame falls
  in (``spec.halves``); frames outside the blocks (pre-match, halftime, tail) are period 0 and dropped unless
  ``keep_ungraded=True`` (they are never scored). ``time_s`` = seconds since the block start.
* metres: ``x = (x_norm - 0.5) * L``, ``y = (0.5 - y_norm) * W`` with ``L, W = spec.pitch`` (105.3 × 68 measured,
  games.py; the metadata says 105 × 68) — same flip as the Metrica EPTS sample (gdkb.loaders.metrica).
* ``player_id`` = the metadata ``<Player>`` ``Name`` when names are unique (``Track N`` in the challenge export),
  else the ``id`` attribute. Team membership per slot from ``teamId``; home = ``Score idLocalTeam`` when the
  metadata has a score, else the first ``<Team>`` (kloppy's rule, baseline.py:84-86).
* the fast parser (``engine="fast"``) streams the raw file with numpy (the kloppy route needs minutes and
  gigabytes on 199k × 46 slots); ``engine="kloppy"`` is kept for cross-checks and is tested equal on the fixture.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lxml import etree

from gdkb.formats import KEY, MatchMeta, make_frames, make_labels
from gdkb.pitch import infer_home_attacks_right
from gdkb.scoring import to_submission


# ------------------------------------------------------------------------------------------------ game spec
@dataclass(frozen=True)
class GameSpec:
    game: str = "BAR-ATH"
    n_video_frames: int = 199319
    fps: float = 25.0
    pitch: tuple[float, float] = (105.3, 68.0)
    # graded half blocks as (period, first video frame, last video frame) — DATA.md §"Which frames are graded"
    halves: tuple[tuple[int, int, int], ...] = ((1, 11185, 90651), (2, 110808, 190197))
    kickoffs: dict[int, int] = field(default_factory=lambda: {1: 11185, 2: 113742})
    # inclusive video-frame ranges that are graded and known to be all D (FAQ: 2nd-half pre-kick-off)
    forced_dead: tuple[tuple[int, int], ...] = ((110808, 113741),)
    home_name: str | None = "Barcelona"  # DATA.md: H = Barcelona; None = trust the metadata order
    sources: tuple[str, ...] = ("tactical", "broadcast")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kickoffs"] = {str(k): v for k, v in self.kickoffs.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameSpec:
        d = dict(d)
        d["pitch"] = tuple(d["pitch"])
        d["halves"] = tuple(tuple(h) for h in d["halves"])
        d["forced_dead"] = tuple(tuple(r) for r in d.get("forced_dead", ()))
        d["kickoffs"] = {int(k): int(v) for k, v in d.get("kickoffs", {}).items()}
        d["sources"] = tuple(d.get("sources", ("tactical",)))
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> GameSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @property
    def graded_frames(self) -> int:
        return sum(last - first + 1 for _, first, last in self.halves)


BAR_ATH = GameSpec()


def period_of_video_frame(video_frame, spec: GameSpec = BAR_ATH) -> np.ndarray:
    """Graded half (1, 2, …) of each video frame; 0 outside the graded blocks."""
    vf = np.asarray(video_frame, dtype=np.int64)
    out = np.zeros(vf.shape, dtype=np.int16)
    for per, first, last in spec.halves:
        out[(vf >= first) & (vf <= last)] = per
    return out


def block_start(spec: GameSpec = BAR_ATH) -> dict[int, int]:
    return {per: first for per, first, _ in spec.halves}


def asset_paths(data_dir: str | Path, game: str = "BAR-ATH", source: str = "tactical") -> dict[str, Path]:
    """``games.asset_paths`` layout: ``<data_dir>/<source>/<game>_<source>_<asset>``."""
    d = Path(data_dir) / source
    stem = f"{game}_{source}"
    out = {
        "atd_meta": d / f"{stem}_atd.xml",
        "atd_raw": d / f"{stem}_atd.txt",
        "homography": d / f"{stem}_homography.csv",
        "video": d / f"{stem}.mp4",
    }
    if source == "tactical":
        out["smart_tagging"] = d / f"{stem}_smart_tagging.xml"
    return out


# ------------------------------------------------------------------------------------------- EPTS metadata
@dataclass
class EptsMeta:
    fps: float
    field_size: tuple[float, float] | None
    teams: list[tuple[str, str]]  # (id, name) in file order
    home_id: str
    players: dict[str, dict[str, Any]]  # player key -> {team, name, number, xml_id}
    player_key_of_id: dict[str, str]
    channels: dict[str, tuple[str, str | None]]  # channel id -> (sensor id, unit)
    player_channels: dict[str, tuple[str, str]]  # player channel id -> (player xml id, channel id)
    specs: list[dict[str, Any]]  # [{start, end, sep, registers}] see _parse_spec
    vendor_periods: dict[int, tuple[int | None, int | None]]
    provider: str | None
    raw: dict[str, Any]


def _text(elm, tag: str, default=None):
    e = elm.find(tag) if elm is not None else None
    return e.text.strip() if e is not None and e.text is not None else default


def read_epts_metadata(path: str | Path) -> EptsMeta:
    root = etree.parse(str(path)).getroot()
    md = root.find("Metadata")
    gc = md.find("GlobalConfig")
    fps = float(_text(gc, "FrameRate", "25"))
    provider = _text(gc, "ProviderName")
    vendor_periods: dict[int, tuple[int | None, int | None]] = {}
    pgp = gc.find("ProviderGlobalParameters")
    if pgp is not None:
        vals = {_text(p, "Name"): _text(p, "Value") for p in pgp.findall("ProviderParameter")}
        names = {1: ("first_half_start", "first_half_end"), 2: ("second_half_start", "second_half_end"),
                 3: ("first_extra_half_start", "first_extra_half_end"),
                 4: ("second_extra_half_start", "second_extra_half_end")}
        for per, (a, b) in names.items():
            if vals.get(a) or vals.get(b):
                vendor_periods[per] = (int(vals[a]) if vals.get(a) else None, int(vals[b]) if vals.get(b) else None)
    session = md.find("Sessions/Session")
    field_size = None
    home_id = None
    if session is not None:
        fs = session.find("FieldSize")
        if fs is not None:
            field_size = (float(_text(fs, "Width")), float(_text(fs, "Height")))
        score = session.find("MatchParameters/Score")
        if score is not None and score.get("idLocalTeam"):
            home_id = score.get("idLocalTeam")
    teams = [(t.get("id"), _text(t, "Name", t.get("id"))) for t in md.findall("Teams/Team")]
    if not teams:
        raise ValueError(f"{path}: no <Team> in metadata")
    if home_id is None:
        home_id = teams[0][0]
    ground = {tid: ("home" if tid == home_id else "away") for tid, _ in teams}
    raw_players = []
    for p in md.findall("Players/Player"):
        raw_players.append({"xml_id": p.get("id"), "team": ground.get(p.get("teamId"), "away"),
                            "name": _text(p, "Name"), "number": _text(p, "ShirtNumber")})
    names = [p["name"] for p in raw_players]
    use_names = all(names) and len(set(names)) == len(names)
    players, key_of = {}, {}
    for p in raw_players:
        key = p["name"] if use_names else p["xml_id"]
        num = p["number"]
        players[key] = {"team": p["team"], "name": p["name"], "number": int(num) if num and num.isdigit() else num,
                        "xml_id": p["xml_id"]}
        key_of[p["xml_id"]] = key
    channels: dict[str, tuple[str, str | None]] = {}
    for sensor in md.findall("Devices/Device/Sensors/Sensor"):
        sid = sensor.get("id")
        for ch in sensor.findall("Channels/Channel"):
            channels[ch.get("id")] = (sid, _text(ch, "Unit"))
    player_channels = {pc.get("id"): (pc.get("playerId"), pc.get("channelId"))
                       for pc in md.findall("PlayerChannels/PlayerChannel")}
    specs = [_parse_spec(s) for s in root.findall("DataFormatSpecifications/DataFormatSpecification")]
    if not specs:
        raise ValueError(f"{path}: no DataFormatSpecification")
    return EptsMeta(fps=fps, field_size=field_size, teams=teams, home_id=home_id, players=players,
                    player_key_of_id=key_of, channels=channels, player_channels=player_channels, specs=specs,
                    vendor_periods=vendor_periods, provider=provider,
                    raw={"use_player_names": use_names, "n_players": len(players)})


def _parse_spec(elm) -> dict[str, Any]:
    """Registers of one DataFormatSpecification in file order: each top-level child is
    ``{"kind": "string", "name"}`` or ``{"kind": "split", "sep", "groups": [[(ref_kind, ref_id), ...], ...]}``
    (a SplitRegister of SplitRegisters = one group per slot) or ``{"kind": "group", "sep", "refs": [...]}``."""
    regs = []
    for child in elm:
        tag = etree.QName(child).localname
        if tag == "StringRegister":
            regs.append({"kind": "string", "name": child.get("name")})
        elif tag == "SplitRegister":
            inner = list(child)
            if inner and etree.QName(inner[0]).localname == "SplitRegister":
                groups = [[(etree.QName(r).localname, r.get("playerChannelId") or r.get("channelId")) for r in g]
                          for g in inner]
                regs.append({"kind": "split", "sep": child.get("separator"), "group_sep": inner[0].get("separator"),
                             "groups": groups})
            else:
                regs.append({"kind": "group", "sep": child.get("separator"),
                             "refs": [(etree.QName(r).localname, r.get("playerChannelId") or r.get("channelId"))
                                      for r in inner]})
    return {"start": int(elm.get("startFrame")), "end": int(elm.get("endFrame")), "sep": elm.get("separator", ":"),
            "registers": regs}


def _spec_layout(spec: dict[str, Any], md: EptsMeta) -> dict[str, Any]:
    """Column layout of one spec after flattening every separator: field index of frame id, per slot the
    (x, y, speed) field indices and the player key, and the ball (x, y, speed) field indices."""
    seps = {spec["sep"]}
    idx = 0
    frame_idx = None
    slots: list[dict[str, Any]] = []
    ball: dict[str, int | None] = {"x": None, "y": None, "s": None}
    for reg in spec["registers"]:
        if reg["kind"] == "string":
            frame_idx = idx
            idx += 1
        elif reg["kind"] == "split":
            seps.update({reg["sep"], reg["group_sep"]})
            for g in reg["groups"]:
                slot = {"x": None, "y": None, "s": None, "player": None, "fields": len(g)}
                for kind, ref in g:
                    if kind == "PlayerChannelRef":
                        pid, cid = md.player_channels[ref]
                        slot["player"] = md.player_key_of_id[pid]
                        sensor = md.channels.get(cid, (None, None))[0]
                        if cid == "x":
                            slot["x"] = idx
                        elif cid == "y":
                            slot["y"] = idx
                        elif sensor == "speed" or cid in ("s", "speed"):
                            slot["s"] = idx
                    idx += 1
                slots.append(slot)
        else:  # group (the ball)
            seps.add(reg["sep"])
            for kind, ref in reg["refs"]:
                if kind == "BallChannelRef":
                    sensor = md.channels.get(ref, (None, None))[0]
                    if ref == "x":
                        ball["x"] = idx
                    elif ref == "y":
                        ball["y"] = idx
                    elif sensor == "speed" or ref in ("s", "speed"):
                        ball["s"] = idx
                idx += 1
    return {"n_fields": idx, "frame_idx": frame_idx, "slots": slots, "ball": ball, "seps": sorted(seps - {None})}


# ------------------------------------------------------------------------------------------- raw tracking
def iter_raw_blocks(raw_path: str | Path, md: EptsMeta, chunk: int = 20000, limit: int | None = None):
    """Yield ``(layout, array)`` chunks: ``array`` is (n_lines, n_fields) float64 parsed with the spec whose
    frame range covers the lines (every separator replaced, then one ``np.fromstring``)."""
    specs = sorted(md.specs, key=lambda s: s["start"])
    layouts = [_spec_layout(s, md) for s in specs]
    si = 0
    buf: list[str] = []
    n_lines = 0

    def flush(i: int):
        if not buf:
            return None
        text = "\n".join(buf)
        for sep in layouts[i]["seps"]:
            if sep != ",":
                text = text.replace(sep, ",")
        arr = np.fromstring(text.replace("\n", ","), sep=",", dtype=np.float64)
        n = layouts[i]["n_fields"]
        if arr.size % n:
            raise ValueError(f"{raw_path}: {arr.size} fields is not a multiple of {n} (spec {specs[i]['start']}-"
                             f"{specs[i]['end']})")
        return layouts[i], arr.reshape(-1, n)

    with open(raw_path, encoding="ascii") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fid = int(line[:line.index(specs[si]["sep"])])
            while fid > specs[si]["end"] and si + 1 < len(specs):
                out = flush(si)
                if out is not None:
                    yield out
                buf = []
                si += 1
            buf.append(line)
            n_lines += 1
            if len(buf) >= chunk:
                yield flush(si)
                buf = []
            if limit and n_lines >= limit:
                break
    out = flush(si)
    if out is not None:
        yield out


def _frames_from_blocks(blocks, md: EptsMeta, spec: GameSpec, match_id: str, keep_ungraded: bool):
    L, W = spec.pitch
    starts = block_start(spec)
    parts = []
    n_records = 0
    fid_min, fid_max = None, None
    for layout, arr in blocks:
        fid = arr[:, layout["frame_idx"]].astype(np.int64)
        n_records += len(fid)
        fid_min = int(fid.min()) if fid_min is None else min(fid_min, int(fid.min()))
        fid_max = int(fid.max()) if fid_max is None else max(fid_max, int(fid.max()))
        vf = fid - 1
        per = period_of_video_frame(vf, spec)
        keep = np.ones(len(fid), dtype=bool) if keep_ungraded else per > 0
        if not keep.any():
            continue
        arr, fid, vf, per = arr[keep], fid[keep], vf[keep], per[keep]
        t0 = np.array([starts.get(int(p), 0) for p in per])
        time_s = np.where(per > 0, (vf - t0) / spec.fps, vf / spec.fps)
        cols = {"frame": [], "video_frame": [], "period": [], "time_s": [], "team": [], "player_id": [],
                "x_norm": [], "y_norm": [], "speed": []}
        for slot in layout["slots"]:
            if slot["x"] is None or slot["y"] is None:
                continue
            xn, yn = arr[:, slot["x"]], arr[:, slot["y"]]
            present = ~(np.isnan(xn) | np.isnan(yn))
            if not present.any():
                continue
            sp = arr[present, slot["s"]] if slot["s"] is not None else np.full(int(present.sum()), np.nan)
            pl = md.players[slot["player"]]
            cols["frame"].append(fid[present])
            cols["video_frame"].append(vf[present])
            cols["period"].append(per[present])
            cols["time_s"].append(time_s[present])
            cols["team"].append(np.full(int(present.sum()), pl["team"], dtype=object))
            cols["player_id"].append(np.full(int(present.sum()), slot["player"], dtype=object))
            cols["x_norm"].append(xn[present])
            cols["y_norm"].append(yn[present])
            cols["speed"].append(sp)
        b = layout["ball"]
        bx = arr[:, b["x"]] if b["x"] is not None else np.full(len(fid), np.nan)
        by = arr[:, b["y"]] if b["y"] is not None else np.full(len(fid), np.nan)
        bs = arr[:, b["s"]] if b["s"] is not None else np.full(len(fid), np.nan)
        cols["frame"].append(fid)
        cols["video_frame"].append(vf)
        cols["period"].append(per)
        cols["time_s"].append(time_s)
        cols["team"].append(np.full(len(fid), "ball", dtype=object))
        cols["player_id"].append(np.full(len(fid), "ball", dtype=object))
        cols["x_norm"].append(bx)
        cols["y_norm"].append(by)
        cols["speed"].append(bs)
        parts.append({k: np.concatenate(v) for k, v in cols.items()})
    if not parts:
        raise ValueError("no frames inside the graded blocks (check spec.halves against the file)")
    c = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    frames = make_frames(
        match_id=[match_id] * len(c["frame"]), period=c["period"], frame=c["frame"], time_s=c["time_s"],
        team=c["team"], player_id=c["player_id"], x=(c["x_norm"] - 0.5) * L, y=(0.5 - c["y_norm"]) * W,
    )
    # make_frames sorts; re-attach the extra columns in the same order
    order = np.lexsort((pd.factorize(c["player_id"], sort=True)[0],
                        pd.Series(c["team"]).map({"home": 0, "away": 1, "ball": 2}).fillna(3).to_numpy(np.int8),
                        c["frame"], c["period"]))
    frames["video_frame"] = c["video_frame"][order]
    frames["x_norm"] = c["x_norm"][order]
    frames["y_norm"] = c["y_norm"][order]
    frames["speed"] = c["speed"][order]
    return frames, {"n_records": n_records, "frame_id_min": fid_min, "frame_id_max": fid_max}


def _frames_from_kloppy(meta_path, raw_path, md: EptsMeta, spec: GameSpec, match_id: str, keep_ungraded: bool,
                        limit: int | None):
    from kloppy import metrica as kmetrica

    ds = kmetrica.load_tracking_epts(str(meta_path), str(raw_path), coordinates="metrica", limit=limit or 0)
    L, W = spec.pitch
    starts = block_start(spec)
    rows = []
    n_records = 0
    fids = []
    for fr in ds.records:
        fid = int(fr.frame_id)
        n_records += 1
        fids.append(fid)
        vf = fid - 1
        per = int(period_of_video_frame([vf], spec)[0])
        if per == 0 and not keep_ungraded:
            continue
        t = (vf - starts[per]) / spec.fps if per else vf / spec.fps
        for pl, pdata in fr.players_data.items():
            cc = pdata.coordinates
            if cc is None or cc.x is None or cc.x != cc.x or cc.y != cc.y:
                continue
            key = md.player_key_of_id.get(str(pl.player_id), str(pl.player_id))
            rows.append((per, fid, vf, t, pl.team.ground.value, key, cc.x, cc.y,
                         np.nan if pdata.speed is None else float(pdata.speed)))
        b = fr.ball_coordinates
        if b is None or b.x is None or b.x != b.x:
            rows.append((per, fid, vf, t, "ball", "ball", np.nan, np.nan, np.nan))
        else:
            rows.append((per, fid, vf, t, "ball", "ball", b.x, b.y, np.nan))
    if not rows:
        raise ValueError("no frames inside the graded blocks (check spec.halves against the file)")
    cols = list(zip(*rows, strict=True))
    per, fid, vf, t, team, pid, xn, yn, sp = (np.asarray(v) for v in cols)
    frames = make_frames(match_id=[match_id] * len(fid), period=per.astype(int), frame=fid.astype(int),
                         time_s=t.astype(float), team=team, player_id=pid,
                         x=(xn.astype(float) - 0.5) * L, y=(0.5 - yn.astype(float)) * W)
    order = np.lexsort((pd.factorize(pid, sort=True)[0],
                        pd.Series(team).map({"home": 0, "away": 1, "ball": 2}).fillna(3).to_numpy(np.int8),
                        fid.astype(int), per.astype(int)))
    frames["video_frame"] = vf.astype(int)[order]
    frames["x_norm"] = xn.astype(float)[order]
    frames["y_norm"] = yn.astype(float)[order]
    frames["speed"] = sp.astype(float)[order]
    return frames, {"n_records": n_records, "frame_id_min": min(fids), "frame_id_max": max(fids)}


def load_atd(meta_path: str | Path, raw_path: str | Path, spec: GameSpec = BAR_ATH, source: str = "tactical",
             engine: str = "fast", keep_ungraded: bool = False, limit: int | None = None,
             check: bool = True) -> tuple[pd.DataFrame, MatchMeta]:
    """EPTS metadata + raw tracking → (frames, meta). See the module docstring for the conventions.

    ``check`` asserts what the organisers verified on the delivered file (load_atd.py:24-33): records numbered
    from 1, contiguous, one per video frame — a warning, not an error, when ``limit`` is set.
    """
    md = read_epts_metadata(meta_path)
    match_id = f"{spec.game}-{source}"
    if engine == "fast":
        frames, info = _frames_from_blocks(iter_raw_blocks(raw_path, md, limit=limit), md, spec, match_id,
                                           keep_ungraded)
    elif engine == "kloppy":
        frames, info = _frames_from_kloppy(meta_path, raw_path, md, spec, match_id, keep_ungraded, limit)
    else:
        raise ValueError("engine must be 'fast' or 'kloppy'")
    problems = []
    if info["frame_id_min"] != 1:
        problems.append(f"first frame_id is {info['frame_id_min']}, expected 1")
    if info["frame_id_max"] - info["frame_id_min"] + 1 != info["n_records"]:
        problems.append("frame ids are not contiguous")
    if not limit and info["n_records"] != spec.n_video_frames:
        problems.append(f"{info['n_records']} records, video has {spec.n_video_frames} frames")
    if problems and check:
        msg = "; ".join(problems) + " — do not invent a correction (load_atd.py)"
        if limit:
            warnings.warn(msg, stacklevel=2)
        else:
            raise ValueError(msg)
    teams = dict(md.teams)
    home_name = teams[md.home_id]
    away_name = next(n for i, n in md.teams if i != md.home_id)
    if spec.home_name and spec.home_name.lower() not in home_name.lower():
        warnings.warn(f"metadata home team is {home_name!r}, spec says {spec.home_name!r}: check H/A", stacklevel=2)
    graded = frames[frames["period"] > 0]
    har = infer_home_attacks_right(graded) if len(graded) else {}
    for per, _, _ in spec.halves:
        har.setdefault(per, True)
    if keep_ungraded:
        har.setdefault(0, True)
    periods = {per: {"start_frame": first + 1, "end_frame": last + 1, "first_video_frame": first,
                     "last_video_frame": last} for per, first, last in spec.halves}
    n_slots = len(md.players)
    meta = MatchMeta(
        match_id=match_id, source="metrica-atd", fps=spec.fps, pitch_length_m=spec.pitch[0],
        pitch_width_m=spec.pitch[1], home_name=home_name, away_name=away_name, home_attacks_right=har,
        players={k: {kk: vv for kk, vv in v.items() if kk != "xml_id"} | {"xml_id": v["xml_id"]}
                 for k, v in md.players.items()},
        licence="More Than a Hack 2027 participant data (T&Cs; local use only, never redistribute, never to a "
                "hosted LLM)",
        provenance={"files": [str(meta_path), str(raw_path)], "spec": spec.to_dict()},
        periods=periods,
        extra={"format": "epts", "engine": engine, "n_slots": n_slots, "field_size_metadata": md.field_size,
               "vendor_periods": {str(k): list(v) for k, v in md.vendor_periods.items()},
               "kickoffs": {str(k): v for k, v in spec.kickoffs.items()}, "frame_is_vendor_frame_id": True,
               "video_frame": "frame - 1", "channels": md.channels, "player_key": "name" if
               md.raw["use_player_names"] else "id", **info, "ungraded_kept": keep_ungraded},
    )
    return frames, meta


# ---------------------------------------------------------------------------------------------- homography
@dataclass
class Homography:
    """Per-video-frame homography (normalised pitch 0–1 → image pixels), ``H[valid]`` finite only."""

    frame: np.ndarray
    H: np.ndarray  # (n, 3, 3)
    reprojection_error: np.ndarray
    valid: np.ndarray

    def __len__(self) -> int:
        return len(self.frame)

    def matrix(self, video_frame: int) -> np.ndarray:
        i = int(np.searchsorted(self.frame, video_frame))
        if i >= len(self.frame) or self.frame[i] != video_frame:
            raise KeyError(f"no homography row for video frame {video_frame}")
        return self.H[i]

    def project(self, video_frame: int, x_norm, y_norm) -> np.ndarray:
        """Normalised pitch (0–1) → pixels (N, 2); NaN when the frame is invalid (starter/load_homography.py)."""
        return project_norm(self.matrix(video_frame), x_norm, y_norm)

    def pixel_to_pitch_matrix(self, video_frame: int, pitch: tuple[float, float]) -> np.ndarray:
        """OpenCV-style 3×3 pixel → metres (centre origin, y up) for ``gdkb.pitch.pixel_to_pitch``."""
        return pixel_to_pitch_matrix(self.matrix(video_frame), pitch)


MATRIX_COLUMNS = [f"h{r}{c}" for r in range(3) for c in range(3)]


def load_homography(path: str | Path) -> Homography:
    table = pd.read_csv(path)
    table.columns = [c.strip().lower() for c in table.columns]
    missing = [c for c in ["frame", "reprojection_error", "valid"] + MATRIX_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    table = table.sort_values("frame", kind="stable")
    H = table[MATRIX_COLUMNS].to_numpy(np.float64).reshape(-1, 3, 3)
    valid = table["valid"].to_numpy().astype(bool)
    H[~valid] = np.nan
    return Homography(frame=table["frame"].to_numpy(np.int64), H=H,
                      reprojection_error=table["reprojection_error"].to_numpy(np.float64), valid=valid)


def project_norm(H: np.ndarray, x_norm, y_norm) -> np.ndarray:
    x = np.asarray(x_norm, dtype=float)
    y = np.asarray(y_norm, dtype=float)
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([(H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den],
                        axis=-1)


def norm_from_metres_matrix(pitch: tuple[float, float]) -> np.ndarray:
    """Metres (centre origin, y up; docs/frame-format.md) → normalised pitch (top-left origin, y down)."""
    L, W = pitch
    return np.array([[1.0 / L, 0.0, 0.5], [0.0, -1.0 / W, 0.5], [0.0, 0.0, 1.0]])


def pixel_to_pitch_matrix(H_norm_to_px: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    """Compose the file's matrix with the metre↔normalised affine map and invert: pixels → metres."""
    return np.linalg.inv(np.asarray(H_norm_to_px, dtype=float) @ norm_from_metres_matrix(pitch))


# ------------------------------------------------------------------------------------------- smart tagging
def _norm_name(s: str | None) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def team_side_map(tag_values, home_name: str, away_name: str, explicit: dict[str, str] | None = None) -> dict:
    """Map the ``Team`` tag texts to home/away by name (explicit map wins; ``N/A`` and unknown → None)."""
    out: dict[str, str | None] = {}
    hn, an = _norm_name(home_name), _norm_name(away_name)
    for v in tag_values:
        if v is None:
            continue
        if explicit and v in explicit:
            out[v] = explicit[v]
            continue
        nv = _norm_name(v)
        if nv in ("na", "none", ""):
            out[v] = None
        elif nv == hn or (nv and (nv in hn or hn in nv)):
            out[v] = "home"
        elif nv == an or (nv and (nv in an or an in nv)):
            out[v] = "away"
        else:
            # last resort: shared first 4 letters
            out[v] = "home" if nv[:4] == hn[:4] else "away" if nv[:4] == an[:4] else None
    return out


def load_smart_tagging(path: str | Path, spec: GameSpec = BAR_ATH, home_name: str | None = None,
                       away_name: str | None = None, team_map: dict[str, str] | None = None) -> pd.DataFrame:
    """SportsCode XML → one row per instance: ``id, code, start_s, end_s, start_frame, end_frame`` (half-open,
    ``round(seconds × fps)``; starter/load_smart_tagging.py), ``team`` (home/away/None), ``half`` (1/2/None),
    ``type`` and one column per tag group (title case, as in the file). ``attrs['unmapped_teams']`` lists Team
    texts that could not be mapped."""
    root = etree.parse(str(path)).getroot()
    inst = root.find("ALL_INSTANCES")
    if inst is None:
        inst = root
    rows = []
    for e in inst.iter("instance"):
        start_s = float(_text(e, "start"))
        end_s = float(_text(e, "end"))
        tags: dict[str, Any] = {}
        for lab in e.findall("label"):
            g = _text(lab, "group") or "label"
            t = _text(lab, "text")
            if g in tags:
                tags[g] = tags[g] if isinstance(tags[g], list) else [tags[g]]
                tags[g].append(t)
            else:
                tags[g] = t
        rows.append({"id": _text(e, "ID"), "code": (_text(e, "code") or "").strip(), "start_s": start_s,
                     "end_s": end_s, "start_frame": int(round(start_s * spec.fps)),
                     "end_frame": int(round(end_s * spec.fps)), **tags})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    team_col = next((c for c in df.columns if c.lower() == "team"), None)
    if team_col is not None:
        m = team_side_map(df[team_col].dropna().unique(), home_name or spec.home_name or "", away_name or "",
                          team_map)
        df["team"] = df[team_col].map(m).astype(object).where(df[team_col].map(m).notna(), None)
        df.attrs["unmapped_teams"] = sorted(v for v, s in m.items() if s is None and _norm_name(v) not in ("na", ""))
    else:
        df["team"] = None
        df.attrs["unmapped_teams"] = []
    half_col = next((c for c in df.columns if c.lower() == "half"), None)
    if half_col is not None:
        df["half"] = df[half_col].astype(str).str.extract(r"(\d)")[0].astype(float).astype("Int64")
    else:
        df["half"] = pd.array([None] * len(df), dtype="Int64")
    type_col = next((c for c in df.columns if c.lower() == "type"), None)
    df["type"] = df[type_col] if type_col is not None else None
    return df


def set_pieces(tags: pd.DataFrame) -> pd.DataFrame:
    """Set-piece windows with the restart-team prior: the ``team`` of the next team-tagged phase that starts
    after the set piece starts (throw-ins are ``N/A``, DATA.md). Columns: start_frame, end_frame, type, team,
    restart_team."""
    if tags.empty:
        return tags
    sp = tags[tags["code"].str.upper() == "SET PIECES"].copy()
    phases = tags[(tags["code"].str.upper() != "SET PIECES") & tags["team"].notna()].sort_values("start_frame")
    nxt = []
    for s in sp["start_frame"]:
        after = phases[phases["start_frame"] >= s]
        nxt.append(after["team"].iloc[0] if len(after) else None)
    sp["restart_team"] = nxt
    sp["restart_team"] = sp["restart_team"].where(sp["team"].isna(), sp["team"])
    return sp[["id", "start_frame", "end_frame", "start_s", "end_s", "type", "team", "restart_team", "half"]]


# -------------------------------------------------------------------------------------- labels / submission
def labels_from_gt_csv(path: str | Path, spec: GameSpec = BAR_ATH, match_id: str | None = None) -> pd.DataFrame:
    """Official ``frame,label`` CSV (H/A/D/X) → labels table on graded frames (``frame`` = video frame + 1)."""
    gt = pd.read_csv(path)
    gt.columns = [c.strip().lower() for c in gt.columns]
    lab = gt["label"].astype(str).str.strip().str.upper()
    keep = lab != "X"
    vf = gt.loc[keep, "frame"].to_numpy(np.int64)
    per = period_of_video_frame(vf, spec)
    ok = per > 0
    pos = lab[keep].map({"H": "home", "A": "away", "D": "none"}).to_numpy(object)
    return make_labels(match_id or f"{spec.game}-tactical", per[ok], vf[ok] + 1, pos[ok], "provided")


def apply_hard_rules(pred: pd.DataFrame, spec: GameSpec = BAR_ATH, frame_col: str = "video_frame") -> pd.DataFrame:
    """Force ``none`` on ``spec.forced_dead`` ranges (2nd-half pre-kick-off), carry the last label over
    ``abstain`` rows inside a graded block (scorer.md §6.3: never write X), keep everything else."""
    out = pred.copy()
    if frame_col not in out.columns:
        out[frame_col] = out["frame"].to_numpy(np.int64) - 1
    vf = out[frame_col].to_numpy(np.int64)
    pos = out["possession"].to_numpy(object).copy()
    per = period_of_video_frame(vf, spec)
    order = np.argsort(vf, kind="stable")
    for p in np.unique(per[per > 0]):
        idx = order[per[order] == p]
        s = pd.Series(pos[idx]).replace({"abstain": None, "X": None}).ffill().bfill()
        pos[idx] = s.fillna("none").to_numpy(object)
    for lo, hi in spec.forced_dead:
        pos[(vf >= lo) & (vf <= hi)] = "none"
    out["possession"] = pos
    return out


def submission(pred: pd.DataFrame, spec: GameSpec = BAR_ATH, frame_col: str = "video_frame",
               hard_rules: bool = True) -> pd.DataFrame:
    """Predictions table → the ``frame,label`` CSV table of exactly ``spec.n_video_frames`` rows
    (X outside the predicted rows, i.e. the ungraded frames)."""
    p = apply_hard_rules(pred, spec, frame_col) if hard_rules else pred
    if frame_col not in p.columns:
        p = p.assign(**{frame_col: p["frame"].to_numpy(np.int64) - 1})
    return to_submission(p, spec.n_video_frames, frame_col)


def write_submission(pred: pd.DataFrame, path: str | Path, spec: GameSpec = BAR_ATH, **kw) -> Path:
    sub = submission(pred, spec, **kw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path


def load_bundle(data_dir: str | Path, game: str = "BAR-ATH", source: str = "tactical", spec: GameSpec = BAR_ATH,
                engine: str = "fast", limit: int | None = None, keep_ungraded: bool = False) -> dict[str, Any]:
    """Everything of one source: ``frames, meta, homography, tagging`` (tagging None for broadcast/missing)."""
    paths = asset_paths(data_dir, game, source)
    frames, meta = load_atd(paths["atd_meta"], paths["atd_raw"], spec, source, engine, keep_ungraded, limit)
    hom = load_homography(paths["homography"]) if paths["homography"].exists() else None
    tagging = None
    if "smart_tagging" in paths and paths["smart_tagging"].exists():
        tagging = load_smart_tagging(paths["smart_tagging"], spec, meta.home_name, meta.away_name)
    return {"frames": frames, "meta": meta, "homography": hom, "tagging": tagging, "paths": paths}


__all__ = ["GameSpec", "BAR_ATH", "period_of_video_frame", "asset_paths", "read_epts_metadata", "load_atd",
           "Homography", "load_homography", "project_norm", "pixel_to_pitch_matrix", "norm_from_metres_matrix",
           "load_smart_tagging", "set_pieces", "team_side_map", "labels_from_gt_csv", "apply_hard_rules",
           "submission", "write_submission", "load_bundle", "KEY"]
