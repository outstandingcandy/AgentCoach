"""Tests for the look-at parameterisation used by the locked-C LM
in :mod:`goalinsight.field_registration.physical_calibrator`.

The parameterisation has to round-trip every rotation matrix the
calibrator might encounter (otherwise warm starts get mangled), and
``_lookat_to_R`` has to produce a proper rotation (orthonormal,
det=+1) for any in-bound ``(yaw, el, roll)``.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from goalinsight.field_registration.physical_calibrator import (
    _R_to_lookat,
    _lookat_to_R,
)


def _is_proper_rotation(R: np.ndarray, tol: float = 1e-10) -> bool:
    return (
        np.allclose(R @ R.T, np.eye(3), atol=tol)
        and abs(float(np.linalg.det(R)) - 1.0) < tol
    )


class TestRoundTrip:
    """Every rotation matrix should reconstruct from its (yaw, el, roll)."""

    @pytest.mark.parametrize("seed", range(20))
    def test_random_rotations_round_trip(self, seed):
        """Random rvec → R → (yaw, el, roll) → R should be a no-op.

        Skip configurations where the optical axis is exactly vertical:
        ``yaw`` is then undefined (gimbal lock) and ``arctan2`` returns
        an arbitrary representative — round-trip still works in
        practice but the comparison is fragile.
        """
        rng = np.random.default_rng(seed)
        rvec = rng.uniform(-1.5, 1.5, size=3)
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        # skip when fwd_z is too close to ±1 (gimbal-lock zone)
        if abs(R[2, 2]) > 0.999:
            pytest.skip("near gimbal lock — yaw is undefined")
        yaw, el, roll = _R_to_lookat(R)
        R_back = _lookat_to_R(yaw, el, roll)
        assert np.allclose(R, R_back, atol=1e-10)

    def test_reasonable_pose_round_trip(self):
        """Sideline rig at ~6° below horizon, modest yaw, near-zero roll —
        the normal operating point for the Sunday cup setup."""
        for yaw_deg in [-30, 0, 33.14, 59.15]:
            for el_deg in [3.0, 6.12, 13.04]:
                for roll_deg in [-0.5, 0.0, 1.5]:
                    yaw, el, roll = (
                        math.radians(yaw_deg),
                        math.radians(el_deg),
                        math.radians(roll_deg),
                    )
                    R = _lookat_to_R(yaw, el, roll)
                    yaw2, el2, roll2 = _R_to_lookat(R)
                    assert math.degrees(yaw2) == pytest.approx(yaw_deg, abs=1e-6)
                    assert math.degrees(el2) == pytest.approx(el_deg, abs=1e-6)
                    assert math.degrees(roll2) == pytest.approx(roll_deg, abs=1e-6)


class TestProperRotation:
    """``_lookat_to_R`` must produce a proper rotation for any input."""

    @pytest.mark.parametrize(
        "yaw_deg, el_deg, roll_deg",
        [
            (0, 6, 0),
            (90, 10, 0),
            (-180, 5, -10),
            (45, 2, 15),
            (-45, 15, -15),
            (33.14, 6.12, 0.81),
        ],
    )
    def test_orthonormal_det_one(self, yaw_deg, el_deg, roll_deg):
        R = _lookat_to_R(
            math.radians(yaw_deg),
            math.radians(el_deg),
            math.radians(roll_deg),
        )
        assert _is_proper_rotation(R)


class TestSemantics:
    """Sanity-check the meaning of each angle."""

    def test_el_zero_is_horizon(self):
        """el=0 → optical axis horizontal → fwd_z = 0."""
        R = _lookat_to_R(yaw=0.0, el=0.0, roll=0.0)
        assert R[2, 2] == pytest.approx(0.0, abs=1e-12)

    def test_positive_el_looks_down(self):
        """Positive el (matching pitch_bounds_deg semantics) → fwd_z < 0."""
        R = _lookat_to_R(yaw=0.0, el=math.radians(10.0), roll=0.0)
        assert R[2, 2] < 0  # camera-z (= fwd) has negative world-z component

    def test_negative_el_looks_up(self):
        """Negative el → fwd_z > 0 (camera looking up). The locked LM
        bounds will reject this, but the parameterisation must still
        represent it cleanly."""
        R = _lookat_to_R(yaw=0.0, el=math.radians(-5.0), roll=0.0)
        assert R[2, 2] > 0

    def test_yaw_zero_faces_plus_y(self):
        """yaw=0 with el=0 → optical axis along +Y_world."""
        R = _lookat_to_R(yaw=0.0, el=0.0, roll=0.0)
        # camera +z (forward) in world = (sin(yaw), cos(yaw), 0) = (0, 1, 0)
        assert np.allclose(R[2], np.array([0.0, 1.0, 0.0]), atol=1e-12)
