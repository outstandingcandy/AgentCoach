"""Tests for goalinsight.utils.config_resolver."""

from __future__ import annotations

import math

import pytest

from goalinsight.utils.config_resolver import resolve_config


def _hfov_to_f(hfov_deg: float, width: int) -> float:
    return width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


class TestProcessFpsDefault:
    def test_unset_defaults_to_min_30(self):
        out = resolve_config({}, width=1920, height=1080, fps=60.0)
        assert out["video"]["process_fps"] == 30.0

    def test_unset_below_30_defaults_to_source(self):
        out = resolve_config({}, width=1920, height=1080, fps=24.0)
        assert out["video"]["process_fps"] == 24.0

    def test_explicit_value_wins(self):
        out = resolve_config(
            {"video": {"process_fps": 5}},
            width=1920, height=1080, fps=60.0,
        )
        assert out["video"]["process_fps"] == 5

    def test_no_fps_no_default(self):
        out = resolve_config({}, width=None, height=None, fps=None)
        assert "process_fps" not in out.get("video", {})


class TestFocalBounds:
    def test_hfov_deg_converts_to_px(self):
        cfg = {
            "field_registration": {
                "physical": {
                    "focal_hfov_deg_bounds": [18.18, 51.28],
                }
            }
        }
        out = resolve_config(cfg, width=3840, height=2160, fps=60.0)
        bounds = out["field_registration"]["physical"]["focal_bounds"]
        # 18.18° on 4K → ~12000 px focal; 51.28° → ~4000 px
        assert bounds[0] == pytest.approx(4000.0, rel=0.01)
        assert bounds[1] == pytest.approx(12000.0, rel=0.01)

    def test_hfov_resolution_independent(self):
        """Same HFOV at 1080p should give exactly half the px focal."""
        cfg = {
            "field_registration": {
                "physical": {
                    "focal_hfov_deg_bounds": [18.18, 51.28],
                }
            }
        }
        out_4k = resolve_config(cfg, 3840, 2160, 60.0)
        out_hd = resolve_config(cfg, 1920, 1080, 60.0)
        f_4k = out_4k["field_registration"]["physical"]["focal_bounds"]
        f_hd = out_hd["field_registration"]["physical"]["focal_bounds"]
        assert f_4k[0] == pytest.approx(f_hd[0] * 2, rel=1e-6)
        assert f_4k[1] == pytest.approx(f_hd[1] * 2, rel=1e-6)

    def test_legacy_focal_bounds_wins(self):
        """If user sets focal_bounds (px) directly, resolver doesn't touch it."""
        cfg = {
            "field_registration": {
                "physical": {
                    "focal_bounds": [4000.0, 12000.0],
                    "focal_hfov_deg_bounds": [10.0, 80.0],  # would resolve to different px
                }
            }
        }
        out = resolve_config(cfg, 3840, 2160, 60.0)
        assert out["field_registration"]["physical"]["focal_bounds"] == [4000.0, 12000.0]


class TestGapFillReprojFrac:
    def test_frac_converts_to_px(self):
        cfg = {
            "field_registration": {
                "physical": {
                    "gap_fill_chain": {
                        "anchor_max_reproj_frac": 0.01,
                        "overwrite_above_reproj_frac": 0.01,
                    }
                }
            }
        }
        out = resolve_config(cfg, width=3840, height=2160, fps=60.0)
        chain = out["field_registration"]["physical"]["gap_fill_chain"]
        assert chain["anchor_max_reproj_px"] == pytest.approx(38.4)
        assert chain["overwrite_above_reproj_px"] == pytest.approx(38.4)

    def test_legacy_px_wins(self):
        cfg = {
            "field_registration": {
                "physical": {
                    "gap_fill_chain": {
                        "anchor_max_reproj_px": 40.0,
                        "anchor_max_reproj_frac": 0.99,  # would be huge
                    }
                }
            }
        }
        out = resolve_config(cfg, 3840, 2160, 60.0)
        assert out["field_registration"]["physical"]["gap_fill_chain"][
            "anchor_max_reproj_px"
        ] == 40.0


class TestImgsz:
    def test_default_4k_caps_at_1920(self):
        out = resolve_config({}, width=3840, height=2160, fps=30.0)
        assert out["unified_detection"]["imgsz"] == 1920

    def test_default_1080p_uses_1920(self):
        out = resolve_config({}, width=1920, height=1080, fps=30.0)
        assert out["unified_detection"]["imgsz"] == 1920

    def test_default_below_1080p_uses_long_edge(self):
        out = resolve_config({}, width=1280, height=720, fps=30.0)
        assert out["unified_detection"]["imgsz"] == 1280

    def test_explicit_value_wins(self):
        out = resolve_config(
            {"unified_detection": {"imgsz": 640}},
            width=3840, height=2160, fps=30.0,
        )
        assert out["unified_detection"]["imgsz"] == 640


class TestMinTrackSeconds:
    def test_seconds_converts_to_frames(self):
        cfg = {
            "video": {"process_fps": 30},
            "track_consolidation": {"min_track_seconds": 1.0},
        }
        out = resolve_config(cfg, 1920, 1080, 60.0)
        # effective_fps = min(30, 60) = 30; 1.0 * 30 = 30 frames
        assert out["track_consolidation"]["min_track_frames"] == 30

    def test_legacy_frames_wins(self):
        cfg = {
            "video": {"process_fps": 30},
            "track_consolidation": {
                "min_track_frames": 50,
                "min_track_seconds": 999.0,  # would clobber if precedence wrong
            },
        }
        out = resolve_config(cfg, 1920, 1080, 60.0)
        assert out["track_consolidation"]["min_track_frames"] == 50

    def test_seconds_at_lower_process_fps(self):
        """At process_fps=10, 3.0 s → 30 frames (matches old default)."""
        cfg = {
            "video": {"process_fps": 10},
            "track_consolidation": {"min_track_seconds": 3.0},
        }
        out = resolve_config(cfg, 1920, 1080, 30.0)
        assert out["track_consolidation"]["min_track_frames"] == 30


class TestNoMutation:
    def test_caller_dict_untouched(self):
        cfg = {
            "field_registration": {
                "physical": {"focal_hfov_deg_bounds": [20.0, 50.0]}
            }
        }
        before = {
            "field_registration": {
                "physical": {"focal_hfov_deg_bounds": [20.0, 50.0]}
            }
        }
        resolve_config(cfg, 3840, 2160, 60.0)
        assert cfg == before


class TestSundayConfigEquivalence:
    """Spot-check: resolving the migrated sunday_soccer.yaml at 4K@60fps
    yields the same operationally-significant values as the pre-migration
    config did."""

    def test_focal_bounds_close_to_pre_migration(self):
        cfg = {
            "field_registration": {
                "physical": {
                    "focal_hfov_deg_bounds": [18.18, 51.28],
                }
            }
        }
        out = resolve_config(cfg, 3840, 2160, 60.0)
        bounds = out["field_registration"]["physical"]["focal_bounds"]
        # Pre-migration: [4000, 12000]. Allow 1% drift from finite precision.
        assert bounds[0] == pytest.approx(4000.0, rel=0.01)
        assert bounds[1] == pytest.approx(12000.0, rel=0.01)

    def test_min_track_frames_matches_30(self):
        cfg = {
            "video": {"process_fps": 30},
            "track_consolidation": {"min_track_seconds": 1.0},
        }
        out = resolve_config(cfg, 3840, 2160, 60.0)
        assert out["track_consolidation"]["min_track_frames"] == 30
