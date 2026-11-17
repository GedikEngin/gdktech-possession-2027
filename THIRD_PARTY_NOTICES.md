# Third-party notices

Our own code is Apache-2.0 (`LICENSE`). Everything below is what this repository redistributes or depends on.
Licence identifiers were read from the packages' PyPI metadata / bundled LICENSE files on 2026-10-08.

## Code vendored in this repository

| file | origin | licence |
|---|---|---|
| `gdkb/scoring/official.py` | `evaluation/possession_metrics.py` of https://github.com/metrica-sports/more-than-a-hack-2027 @ `b42969321c74e07c97972d271fffc3bd01405bb6` (2026-10-05), byte-identical below the licence block | MIT, Copyright (c) 2026-2027 Metrica Sports (full text in the file header). The MIT licence covers the repository code, docs and the public 5-minute ground truth only, **not** the match data |

## Runtime dependencies (`requirements.txt`, exact versions, hashes pinned)

| package | version | licence | home |
|---|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause (with 0BSD, MIT, Zlib, CC0-1.0 components) | https://numpy.org |
| pandas | 3.0.6 | BSD-3-Clause | https://pandas.pydata.org |
| lxml | 6.1.3 | BSD-3-Clause (libxml2 / libxslt: MIT) | https://lxml.de |
| lightgbm | 4.7.0 | MIT (Microsoft Corporation / the LightGBM developers) | https://github.com/microsoft/LightGBM |
| scipy | 1.18.1 | BSD-3-Clause (lightgbm dependency) | https://scipy.org |
| python-dateutil | 2.9.0.post0 | Apache-2.0 / BSD-3-Clause dual (pandas dependency) | https://github.com/dateutil/dateutil |
| six | 1.17.0 | MIT (dateutil dependency) | https://github.com/benjaminp/six |
| narwhals | 2.26.0 | MIT (pandas dependency) | https://github.com/narwhals-dev/narwhals |
| torch | 2.14.1+cpu (PyTorch CPU wheel index) | BSD-3-Clause (bi-GRU segmenter inference; weights are our own, `models/seq_model.npz`) | https://pytorch.org |
| filelock, fsspec, jinja2, markupsafe, networkx, sympy, mpmath, typing-extensions, setuptools | 4.0.12, 2026.9.0, 3.1.6, 3.0.4, 3.7, 1.14.0, 1.3.0, 4.16.0, 84.0.0 | Unlicense, BSD-3-Clause, BSD-3-Clause, BSD-3-Clause, BSD-3-Clause, BSD-3-Clause, BSD-3-Clause, PSF-2.0, MIT — torch dependencies, none of them exercised by `predict.py` beyond import | — |
| pytest, pluggy, iniconfig, packaging, pygments | 9.1.1, 1.6.0, 2.3.1, 26.3, 2.21.0 | MIT, MIT, MIT, Apache-2.0 OR BSD-2-Clause, BSD-2-Clause — tests only | https://docs.pytest.org |

System: `libgomp1` (GCC runtime library exception) for LightGBM's manylinux wheel (the torch CPU wheel bundles its own
OpenMP runtime); Python 3.12 (PSF-2.0).

No GPL, AGPL or non-commercial licence is in the inference path. No detector / tracker / vision model
(RF-DETR, YOLO, ByteTrack, TVCalib, …) is used for the Initial Challenge file.

## Models (`models/`)

`segmenter.lgb.txt` and `dead_model.lgb.txt` are LightGBM boosters we trained (Apache-2.0, like the code) on:

| dataset | used for | licence / terms | attribution |
|---|---|---|---|
| Metrica Sports sample data (Sample Game 1, 2, 3: tracking + events), https://github.com/metrica-sports/sample-data | training labels and features | no SPDX licence file. README "Legal stuff": "Please be responsible with the use of this data. If you use it for anything public, please aknowledge the source." | Metrica Sports — thank you for the sample data |
| IDSSE / DFL open dataset, 3 of 7 Bundesliga matches (DFL-MAT-J03WMX, J03WN1, J03WPY: positions + events), https://doi.org/10.6084/m9.figshare.28196177 | training labels and features; J03WMX also as a full-length rendered test bundle | CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ | Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event data in elite soccer." Sci Data 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y |

Neither dataset is redistributed here; the models do not contain any data rows. The challenge's own match data
was not available when the models were trained and is never used for training.

## Test fixture (`tests/fixtures/challenge_mini/`)

A synthetic 700-frame bundle in the challenge file formats, generated from two 12-second windows of Metrica
Sample Game 3 after heavy degradation (46 anonymous slots, ball masked, duplicates / dropouts, coordinates
re-normalised; see its README). Source acknowledged as above; it is test scaffolding, not a dataset.
