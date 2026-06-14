"""Track consolidation — merge fragmented track_ids into stable player_ids.

Pipeline:

1. Sample K representative crops per candidate track.
2. Send each track's crops to the configured VLM for jersey recognition.
3. Cluster tracks with the same (team, jersey) via ReID cosine.
4. Absorb orphan tracks (no number) into the nearest player centroid.
5. Write the consolidated ``tracks.json`` and ``team_assignments.json``
   into ``track_consolidation/`` (additive — the raw tracker outputs
   in ``tracking/`` stay untouched so downstream stages can re-run
   without losing the source-of-truth).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._runner import run_track_consolidation


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_tracks(
    run_dir: Path | str,
    prefer: str = "consolidated",
) -> dict[str, list[dict[str, Any]]] | None:
    """Read tracks.json honouring the additive-output rule.

    Layout:
      - ``<run>/tracking/tracks.json``: raw tracker output (int tids).
      - ``<run>/track_consolidation/tracks.json``: post-consolidation
        copy with player_id strings, jersey numbers, role, team.

    ``prefer="consolidated"`` (default) returns the consolidated file
    when consolidation has run, else the raw tracker output.
    ``prefer="raw"`` always returns the tracker output.
    """
    run_dir = Path(run_dir)
    if prefer == "raw":
        return _load_json(run_dir / "tracking" / "tracks.json")
    consolidated = run_dir / "track_consolidation" / "tracks.json"
    if consolidated.exists():
        return _load_json(consolidated)
    return _load_json(run_dir / "tracking" / "tracks.json")


def load_team_assignments(
    run_dir: Path | str,
    prefer: str = "consolidated",
) -> dict[str, str] | None:
    """Same convention as :func:`load_tracks` for team_assignments.json."""
    run_dir = Path(run_dir)
    if prefer == "raw":
        return _load_json(run_dir / "tracking" / "team_assignments.json")
    consolidated = run_dir / "track_consolidation" / "team_assignments.json"
    if consolidated.exists():
        return _load_json(consolidated)
    return _load_json(run_dir / "tracking" / "team_assignments.json")


__all__ = [
    "run_track_consolidation",
    "load_tracks",
    "load_team_assignments",
]
