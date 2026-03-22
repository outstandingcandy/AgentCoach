"""Convert between homography and BroadTrack camera parameters.

Provides bidirectional conversion between 3x3 homography matrices
and the camera parameter format used by BroadTrack (pan/tilt/roll/FOV).
"""

import numpy as np


# Default camera position (BroadTrack convention: negative Z for elevation)
DEFAULT_CAMERA_X = 0.0  # meters, center of pitch
DEFAULT_CAMERA_Y = 80.0  # meters, behind far touchline
DEFAULT_CAMERA_Z = -15.0  # meters, elevated (negative in BroadTrack coords)


def pan_tilt_roll_to_rotation(
    pan: float,
    tilt: float,
    roll: float,
) -> np.ndarray:
    """Convert euler angles (radians) to rotation matrix.

    Convention matches visualize_projection_v3.py:
    - Pan: rotation around Z axis
    - Tilt: rotation around X axis
    - Roll: rotation around Z axis (second)

    Args:
        pan: Pan angle in radians.
        tilt: Tilt angle in radians.
        roll: Roll angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    Rpan = np.array([
        [np.cos(pan), -np.sin(pan), 0],
        [np.sin(pan), np.cos(pan), 0],
        [0, 0, 1],
    ])
    Rtilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt), np.cos(tilt)],
    ])
    Rroll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1],
    ])
    return Rpan @ Rtilt @ Rroll


def rotation_to_pan_tilt_roll(R: np.ndarray) -> tuple[float, float, float]:
    """Extract euler angles from rotation matrix.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Tuple of (pan, tilt, roll) in radians.
    """
    # Extract tilt from R[1,2] and R[2,2]
    tilt = np.arctan2(-R[1, 2], R[2, 2])

    # Extract pan from R[0,0] and R[1,0]
    cos_tilt = np.cos(tilt)
    if abs(cos_tilt) > 1e-6:
        pan = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock case
        pan = np.arctan2(-R[0, 1], R[1, 1])

    # Extract roll - approximate as 0 for most broadcast cameras
    # Full extraction would require more complex decomposition
    roll = 0.0

    return pan, tilt, roll


class CameraParamConverter:
    """Convert between homography and BroadTrack camera parameters.

    BroadTrack format:
        {
            "panDegrees": float,
            "tiltDegrees": float,
            "rollDegrees": float,
            "horizontalFieldOfViewDegrees": float,
            "positionXMeters": float,
            "positionYMeters": float,
            "positionZMeters": float,
            "sensorResolutionWidthPixels": int,
            "sensorResolutionHeightPixels": int,
        }
    """

    def __init__(
        self,
        image_width: int = 1920,
        image_height: int = 1080,
        default_fov: float = 90.0,
        camera_position: tuple[float, float, float] | None = None,
    ):
        """Initialize converter.

        Args:
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            default_fov: Default horizontal FOV in degrees.
            camera_position: Default camera position (x, y, z) in meters.
        """
        self.image_width = image_width
        self.image_height = image_height
        self.default_fov = default_fov

        if camera_position:
            self.camera_x, self.camera_y, self.camera_z = camera_position
        else:
            self.camera_x = DEFAULT_CAMERA_X
            self.camera_y = DEFAULT_CAMERA_Y
            self.camera_z = DEFAULT_CAMERA_Z

    def camera_params_to_homography(
        self,
        camera_params: dict,
    ) -> np.ndarray:
        """Convert camera parameters to world->image homography.

        Args:
            camera_params: BroadTrack camera parameters dict.

        Returns:
            3x3 homography matrix (world XY -> image).
        """
        # Extract parameters
        pan = np.radians(camera_params.get("panDegrees", 0))
        tilt = np.radians(camera_params.get("tiltDegrees", 0))
        roll = np.radians(camera_params.get("rollDegrees", 0))
        hfov = np.radians(camera_params.get("horizontalFieldOfViewDegrees", self.default_fov))

        width = camera_params.get("sensorResolutionWidthPixels", self.image_width)
        height = camera_params.get("sensorResolutionHeightPixels", self.image_height)

        pos_x = camera_params.get("positionXMeters", self.camera_x)
        pos_y = camera_params.get("positionYMeters", self.camera_y)
        pos_z = camera_params.get("positionZMeters", self.camera_z)

        # Build intrinsic matrix
        focal = width / (2 * np.tan(hfov / 2))
        cx, cy = width / 2, height / 2

        K = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1],
        ])

        # Build rotation matrix
        R = pan_tilt_roll_to_rotation(pan, tilt, roll).T

        # Camera position
        t = np.array([pos_x, pos_y, pos_z])

        # Compute homography for ground plane (z=0)
        # P = K @ R @ [I | -t]
        # For z=0: x_img = H @ [x_world, y_world, 1]

        # Extract first two columns of R and third column as R@(-t)
        r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
        Rt = R @ (-t)

        # Homography for z=0 plane
        H = K @ np.column_stack([r1, r2, Rt])

        return H / H[2, 2]

    def homography_to_camera_params(
        self,
        H: np.ndarray,
        reference_params: dict | None = None,
    ) -> dict:
        """Estimate camera parameters from homography.

        This is an approximate inverse - some parameters (like camera
        position) cannot be uniquely determined from H alone.

        Args:
            H: World->image homography (3x3).
            reference_params: Reference params to inherit fixed values from.

        Returns:
            Estimated camera parameters dict.
        """
        if reference_params:
            # Use reference values for fixed parameters
            width = reference_params.get("sensorResolutionWidthPixels", self.image_width)
            height = reference_params.get("sensorResolutionHeightPixels", self.image_height)
            hfov = reference_params.get("horizontalFieldOfViewDegrees", self.default_fov)
            pos_x = reference_params.get("positionXMeters", self.camera_x)
            pos_y = reference_params.get("positionYMeters", self.camera_y)
            pos_z = reference_params.get("positionZMeters", self.camera_z)
        else:
            width = self.image_width
            height = self.image_height
            hfov = self.default_fov
            pos_x = self.camera_x
            pos_y = self.camera_y
            pos_z = self.camera_z

        # Estimate pan and tilt from homography
        # This uses the relationship between H columns and camera orientation

        focal = width / (2 * np.tan(np.radians(hfov) / 2))

        # Normalize H
        H = H / H[2, 2]

        # Extract columns
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

        # Estimate intrinsic-normalized columns
        K_inv = np.array([
            [1/focal, 0, -width/(2*focal)],
            [0, 1/focal, -height/(2*focal)],
            [0, 0, 1],
        ])

        r1 = K_inv @ h1
        r2 = K_inv @ h2

        # Normalize
        r1 = r1 / np.linalg.norm(r1)
        r2 = r2 / np.linalg.norm(r2)
        r3 = np.cross(r1, r2)

        R = np.column_stack([r1, r2, r3]).T

        # Extract euler angles
        pan, tilt, roll = rotation_to_pan_tilt_roll(R)

        return {
            "panDegrees": float(np.degrees(pan)),
            "tiltDegrees": float(np.degrees(tilt)),
            "rollDegrees": float(np.degrees(roll)),
            "horizontalFieldOfViewDegrees": float(hfov),
            "positionXMeters": float(pos_x),
            "positionYMeters": float(pos_y),
            "positionZMeters": float(pos_z),
            "sensorResolutionWidthPixels": int(width),
            "sensorResolutionHeightPixels": int(height),
        }

    def update_params_with_delta_h(
        self,
        base_params: dict,
        delta_H: np.ndarray,
    ) -> dict:
        """Update camera params with frame-to-frame homography change.

        Args:
            base_params: Base camera parameters.
            delta_H: Frame-to-frame homography (image space).

        Returns:
            Updated camera parameters.
        """
        # Convert base params to homography
        H_base = self.camera_params_to_homography(base_params)

        # Apply delta (in image space)
        # H_new = delta_H @ H_base (for world->image)
        H_new = delta_H @ H_base

        # Convert back to params
        return self.homography_to_camera_params(H_new, base_params)

    def interpolate_params(
        self,
        params1: dict,
        params2: dict,
        t: float,
    ) -> dict:
        """Linearly interpolate between camera parameters.

        Args:
            params1: First camera parameters.
            params2: Second camera parameters.
            t: Interpolation factor [0, 1].

        Returns:
            Interpolated camera parameters.
        """
        def lerp(a, b, t):
            return a + t * (b - a)

        # Interpolate varying parameters
        return {
            "panDegrees": lerp(params1["panDegrees"], params2["panDegrees"], t),
            "tiltDegrees": lerp(params1["tiltDegrees"], params2["tiltDegrees"], t),
            "rollDegrees": lerp(params1["rollDegrees"], params2["rollDegrees"], t),
            "horizontalFieldOfViewDegrees": lerp(
                params1["horizontalFieldOfViewDegrees"],
                params2["horizontalFieldOfViewDegrees"],
                t,
            ),
            # Keep position fixed
            "positionXMeters": params1["positionXMeters"],
            "positionYMeters": params1["positionYMeters"],
            "positionZMeters": params1["positionZMeters"],
            "sensorResolutionWidthPixels": params1["sensorResolutionWidthPixels"],
            "sensorResolutionHeightPixels": params1["sensorResolutionHeightPixels"],
        }
