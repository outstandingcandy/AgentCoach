"""Player profile stage — per-player crops, heatmaps, distance run.

Public surface kept minimal so importing the package doesn't pull in
matplotlib until something actually needs it.
"""

from ._runner import build_player_profiles

__all__ = ["build_player_profiles"]
