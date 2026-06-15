"""SIFT-based feature extraction and matching for homography estimation.

Implements robust feature matching with Lowe's ratio test and RANSAC
for computing frame-to-frame homographies.
"""

import cv2
import numpy as np


class FeatureMatcher:
    """SIFT feature extraction and matching with RANSAC homography.

    Uses SIFT for robust feature detection and matching between frames,
    with Lowe's ratio test for filtering good matches.
    """

    def __init__(
        self,
        n_features: int = 2000,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
        ratio_threshold: float = 0.75,
        ransac_reproj_threshold: float = 5.0,
        min_matches: int = 10,
        max_long_side: int | None = 1920,
    ):
        """Initialize feature matcher.

        Args:
            n_features: Maximum number of SIFT features.
            contrast_threshold: SIFT contrast threshold.
            edge_threshold: SIFT edge threshold.
            ratio_threshold: Lowe's ratio test threshold.
            ransac_reproj_threshold: RANSAC reprojection threshold.
            min_matches: Minimum number of matches for valid homography.
            max_long_side: Cap the long side of the image fed to SIFT to
                this many pixels (default 1920 = downsample 4K to 1080p).
                SIFT cost scales ~linearly with pixel count and 4K is
                roughly 4× slower than 1080p with no benefit on field
                imagery — the same descriptors come out either way.
                Keypoint coords are scaled BACK to the input frame's
                resolution so callers see native-resolution coordinates.
                Set to None to disable.
        """
        self.n_features = n_features
        self.ratio_threshold = ratio_threshold
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.min_matches = min_matches
        self.max_long_side = max_long_side

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
        )

        # FLANN matcher for efficient matching
        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # BF matcher as fallback
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2)

    def extract_features(
        self,
        frame: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[tuple, np.ndarray | None]:
        """Extract SIFT features from frame.

        Args:
            frame: Input frame (BGR or grayscale).
            mask: Optional mask where 255 = valid region.

        Returns:
            Tuple of (keypoints, descriptors).
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Optional downscale before SIFT to cap CPU cost on 4K input.
        # Each output keypoint's (x, y) is rescaled back to the input
        # frame's resolution so callers can keep using native coords.
        scale = 1.0
        if self.max_long_side is not None:
            long_side = max(gray.shape[:2])
            if long_side > self.max_long_side:
                scale = self.max_long_side / long_side
                new_w = int(round(gray.shape[1] * scale))
                new_h = int(round(gray.shape[0] * scale))
                gray = cv2.resize(gray, (new_w, new_h),
                                  interpolation=cv2.INTER_AREA)
                if mask is not None:
                    mask = cv2.resize(mask, (new_w, new_h),
                                      interpolation=cv2.INTER_NEAREST)

        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)

        if scale != 1.0 and keypoints:
            inv = 1.0 / scale
            for kp in keypoints:
                kp.pt = (kp.pt[0] * inv, kp.pt[1] * inv)
                kp.size *= inv

        return keypoints, descriptors

    def match_features(
        self,
        keypoints1: tuple,
        descriptors1: np.ndarray,
        keypoints2: tuple,
        descriptors2: np.ndarray,
        use_bf: bool = False,
    ) -> list[tuple[cv2.KeyPoint, cv2.KeyPoint, cv2.DMatch]]:
        """Match features between two frames using Lowe's ratio test.

        Args:
            keypoints1: Keypoints from first frame.
            descriptors1: Descriptors from first frame.
            keypoints2: Keypoints from second frame.
            descriptors2: Descriptors from second frame.
            use_bf: Use brute-force matcher instead of FLANN.

        Returns:
            List of (kp1, kp2, match) tuples for good matches.
        """
        if descriptors1 is None or descriptors2 is None:
            return []

        if len(descriptors1) < 2 or len(descriptors2) < 2:
            return []

        # Use KNN matching with k=2 for ratio test
        matcher = self.bf_matcher if use_bf else self.matcher

        try:
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        except cv2.error:
            # Fall back to BF matcher if FLANN fails
            matches = self.bf_matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < self.ratio_threshold * n.distance:
                kp1 = keypoints1[m.queryIdx]
                kp2 = keypoints2[m.trainIdx]
                good_matches.append((kp1, kp2, m))

        return good_matches

    def compute_homography(
        self,
        matches: list[tuple[cv2.KeyPoint, cv2.KeyPoint, cv2.DMatch]],
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        """Compute homography from matched features using RANSAC.

        Args:
            matches: List of (kp1, kp2, match) tuples.

        Returns:
            Tuple of (homography, inlier_mask, metadata).
            Homography transforms points from frame1 to frame2.
        """
        metadata = {
            "num_matches": len(matches),
            "num_inliers": 0,
            "inlier_ratio": 0.0,
        }

        if len(matches) < self.min_matches:
            return None, None, metadata

        # Extract point correspondences
        pts1 = np.float32([m[0].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([m[1].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(
            pts1, pts2,
            cv2.RANSAC,
            self.ransac_reproj_threshold,
        )

        if H is None:
            return None, None, metadata

        # Count inliers
        num_inliers = int(mask.sum()) if mask is not None else len(matches)
        inlier_ratio = num_inliers / len(matches) if matches else 0

        metadata["num_inliers"] = num_inliers
        metadata["inlier_ratio"] = inlier_ratio

        if num_inliers < self.min_matches:
            return None, mask, metadata

        return H, mask, metadata

    def compute_frame_homography(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        mask1: np.ndarray | None = None,
        mask2: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, dict]:
        """Compute homography between two frames.

        Args:
            frame1: Source frame.
            frame2: Target frame.
            mask1: Mask for source frame.
            mask2: Mask for target frame.

        Returns:
            Tuple of (homography, metadata).
            Homography transforms points from frame1 to frame2.
        """
        kp1, desc1 = self.extract_features(frame1, mask1)
        kp2, desc2 = self.extract_features(frame2, mask2)

        metadata = {
            "num_keypoints_frame1": len(kp1),
            "num_keypoints_frame2": len(kp2),
            "keypoints1": kp1,
            "keypoints2": kp2,
        }

        matches = self.match_features(kp1, desc1, kp2, desc2)

        H, inlier_mask, match_meta = self.compute_homography(matches)

        metadata.update(match_meta)
        metadata["matches"] = matches
        metadata["inliers"] = inlier_mask.flatten().tolist() if inlier_mask is not None else []

        return H, metadata

    def compute_reprojection_error(
        self,
        H: np.ndarray,
        matches: list[tuple[cv2.KeyPoint, cv2.KeyPoint, cv2.DMatch]],
    ) -> float:
        """Compute mean reprojection error for homography.

        Args:
            H: Homography matrix.
            matches: List of matched keypoints.

        Returns:
            Mean reprojection error in pixels.
        """
        if H is None or not matches:
            return float("inf")

        pts1 = np.float32([m[0].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([m[1].pt for m in matches]).reshape(-1, 1, 2)

        # Transform pts1 using homography
        pts1_transformed = cv2.perspectiveTransform(pts1, H)

        # Compute error
        errors = np.sqrt(np.sum((pts2 - pts1_transformed) ** 2, axis=2)).flatten()

        return float(np.mean(errors))
