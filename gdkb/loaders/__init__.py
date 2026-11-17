"""Vendor loaders → internal frame format. Each module exposes

- ``load_tracking(...) -> (frames_df, meta)``
- ``load_events(...) -> events_df`` (normalised events, see ``gdkb.labels.EVENT_COLUMNS``)
- ``provided_labels(...)`` where the vendor ships per-frame possession (DFL, SkillCorner).

Thin adapters over kloppy wherever kloppy reads the vendor (Metrica CSV/EPTS, Sportec events,
SkillCorner); hand parsers only for Metrica CSV events (no kloppy reader) and the DFL positions XML
(kloppy's reader peaks at ~3.9 GB RSS on one match, above the VPS 2 GB rule; the streaming parser
stays under 1 GB).
"""
