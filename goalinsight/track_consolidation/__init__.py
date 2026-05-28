"""Track consolidation — merge fragmented track_ids into stable player_ids.

Pipeline:

1. Sample K representative crops per candidate track.
2. Send each track's crops to Claude (Bedrock) for jersey recognition.
3. Cluster tracks with the same (team, jersey) via OSNet ReID cosine.
4. Absorb orphan tracks (no number) into the nearest player centroid.
5. Rewrite ``tracks.json`` and ``team_assignments.json`` with player_ids.
"""

from __future__ import annotations

from ._runner import run_track_consolidation

__all__ = ["run_track_consolidation"]
