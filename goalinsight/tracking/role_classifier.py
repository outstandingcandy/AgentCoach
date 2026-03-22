"""Role and team classification using ReID embeddings (SoccerMaster paper approach).

This module implements team affiliation using ReID embeddings clustering,
as described in the SoccerMaster paper. Role classification (player/goalkeeper/referee)
should ideally come from a fine-tuned YOLOv8 detector output, but we provide
legacy color-based fallbacks for compatibility.

Paper approach:
- Team Affiliation: ReID embeddings clustering + field position (left/right half)
- Role Classification: YOLOv8 fine-tuned with 3 classes (player/goalkeeper/referee)

Legacy approach (not recommended):
- Team Affiliation: Color histogram K-Means clustering
- Role Classification: HSV color thresholds
"""

from typing import Any

import cv2
import numpy as np

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


class ReIDTeamClassifier:
    """Team classification using ReID embeddings (SoccerMaster paper approach).

    This classifier uses tracklet-averaged ReID embeddings to cluster players
    into two teams, optionally using field position to assign left/right labels.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize ReID-based team classifier.

        Args:
            config: Classification configuration.
        """
        if KMeans is None:
            raise ImportError(
                "scikit-learn package is required. "
                "Install with: pip install scikit-learn"
            )

        self.config = config or {}
        self.n_clusters = self.config.get("n_clusters", 2)  # Two teams
        self.use_field_position = self.config.get("use_field_position", True)

        self.kmeans = None
        self.cluster_labels = {}  # cluster_id -> team label

        # Store embeddings for re-clustering
        self._track_embeddings: dict[int, list[np.ndarray]] = {}
        self._track_positions: dict[int, list[tuple[float, float]]] = {}

    def add_track_embedding(
        self,
        track_id: int,
        embedding: np.ndarray,
        pitch_position: tuple[float, float] | None = None,
    ) -> None:
        """Add a ReID embedding observation for a track.

        Args:
            track_id: Track identifier.
            embedding: ReID embedding vector.
            pitch_position: Optional pitch position (x, y) in meters.
        """
        if track_id not in self._track_embeddings:
            self._track_embeddings[track_id] = []
            self._track_positions[track_id] = []

        self._track_embeddings[track_id].append(embedding)
        if pitch_position is not None:
            self._track_positions[track_id].append(pitch_position)

    def get_track_mean_embedding(self, track_id: int) -> np.ndarray | None:
        """Get mean ReID embedding for a track.

        Args:
            track_id: Track identifier.

        Returns:
            Mean embedding vector or None if track not found.
        """
        if track_id not in self._track_embeddings:
            return None
        embeddings = self._track_embeddings[track_id]
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)

    def fit(
        self,
        embeddings: np.ndarray | None = None,
        track_ids: list[int] | None = None,
    ) -> None:
        """Fit K-Means classifier on ReID embeddings.

        Can either use pre-computed embeddings or use accumulated track embeddings.

        Args:
            embeddings: Optional array of embeddings (N, D).
            track_ids: Optional list of track IDs (if using accumulated embeddings).
        """
        if embeddings is None:
            # Use accumulated track embeddings
            track_ids = list(self._track_embeddings.keys())
            embeddings_list = []
            valid_track_ids = []

            for tid in track_ids:
                mean_emb = self.get_track_mean_embedding(tid)
                if mean_emb is not None:
                    embeddings_list.append(mean_emb)
                    valid_track_ids.append(tid)

            if not embeddings_list:
                return

            embeddings = np.array(embeddings_list)
            track_ids = valid_track_ids

        if len(embeddings) < self.n_clusters:
            return

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        labels = self.kmeans.fit_predict(embeddings)

        # Assign team labels based on field position if available
        if self.use_field_position and track_ids is not None:
            self._assign_labels_by_position(track_ids, labels)
        else:
            # Default assignment
            self.cluster_labels = {0: "left", 1: "right"}

    def _assign_labels_by_position(
        self,
        track_ids: list[int],
        cluster_labels: np.ndarray,
    ) -> None:
        """Assign team labels based on average field position.

        Args:
            track_ids: List of track IDs.
            cluster_labels: Cluster assignment for each track.
        """
        cluster_positions: dict[int, list[float]] = {i: [] for i in range(self.n_clusters)}

        for tid, cluster_id in zip(track_ids, cluster_labels):
            positions = self._track_positions.get(tid, [])
            if positions:
                # Use x-coordinate (pitch width direction)
                mean_x = np.mean([p[0] for p in positions])
                cluster_positions[cluster_id].append(mean_x)

        # Calculate mean x for each cluster
        cluster_mean_x = {}
        for cluster_id, x_positions in cluster_positions.items():
            if x_positions:
                cluster_mean_x[cluster_id] = np.mean(x_positions)
            else:
                cluster_mean_x[cluster_id] = 52.5  # Middle of pitch

        # Assign "left" to cluster with lower mean x
        sorted_clusters = sorted(cluster_mean_x.keys(), key=lambda k: cluster_mean_x[k])
        self.cluster_labels = {
            sorted_clusters[0]: "left",
            sorted_clusters[1]: "right" if len(sorted_clusters) > 1 else "unknown",
        }

    def predict(self, embedding: np.ndarray) -> int:
        """Predict team cluster for a single embedding.

        Args:
            embedding: ReID embedding vector.

        Returns:
            Cluster ID.
        """
        if self.kmeans is None:
            return -1

        embedding = embedding.reshape(1, -1)
        return int(self.kmeans.predict(embedding)[0])

    def predict_batch(self, embeddings: np.ndarray) -> list[int]:
        """Predict team clusters for multiple embeddings.

        Args:
            embeddings: Array of embeddings (N, D).

        Returns:
            List of cluster IDs.
        """
        if self.kmeans is None or len(embeddings) == 0:
            return [-1] * len(embeddings)

        return self.kmeans.predict(embeddings).tolist()

    def get_team(self, cluster_id: int) -> str:
        """Get team label for a cluster.

        Args:
            cluster_id: Cluster identifier.

        Returns:
            Team label string.
        """
        return self.cluster_labels.get(cluster_id, "unknown")

    def clear(self) -> None:
        """Clear accumulated embeddings and reset classifier."""
        self._track_embeddings.clear()
        self._track_positions.clear()
        self.kmeans = None
        self.cluster_labels.clear()


class RoleClassifier:
    """Role and team classification with ReID support.

    Supports both:
    - ReID-based team classification (recommended, paper approach)
    - Legacy color-based classification (fallback)

    Role classification (player/goalkeeper/referee) should ideally come from
    a fine-tuned detector. The color-based methods are provided as legacy fallbacks.
    """

    # Grass/field green color range in HSV for background removal
    GRASS_COLOR = {
        "lower": np.array([30, 30, 30]),
        "upper": np.array([85, 255, 220]),
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize role classifier.

        Args:
            config: Classification configuration.
        """
        if KMeans is None:
            raise ImportError(
                "scikit-learn package is required. "
                "Install with: pip install scikit-learn"
            )

        self.config = config or {}
        self.n_clusters = self.config.get("n_clusters", 2)

        # Classification mode: "reid" (recommended) or "color" (legacy)
        self.mode = self.config.get("mode", "reid")

        # ReID-based classifier (primary)
        self.reid_classifier = ReIDTeamClassifier(config)

        # Color-based fallback (legacy)
        self.color_space = self.config.get("color_space", "hsv")
        self.feature_type = self.config.get("feature_type", "histogram")
        self.kmeans = None
        self.cluster_labels = {}
        self.use_background_removal = self.config.get("use_background_removal", True)

    def remove_grass_background(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remove grass/field background from player crop.

        Args:
            crop: Player image crop (BGR format).

        Returns:
            Tuple of (foreground_mask, masked_crop).
        """
        if crop.size == 0:
            return np.zeros((1, 1), dtype=np.uint8), crop

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, self.GRASS_COLOR["lower"], self.GRASS_COLOR["upper"])
        foreground_mask = cv2.bitwise_not(grass_mask)

        kernel = np.ones((3, 3), np.uint8)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

        masked_crop = cv2.bitwise_and(crop, crop, mask=foreground_mask)
        return foreground_mask, masked_crop

    # =========================================================================
    # ReID-based methods (recommended - SoccerMaster paper approach)
    # =========================================================================

    def fit_with_reid(
        self,
        embeddings: np.ndarray,
        track_ids: list[int] | None = None,
        positions: list[tuple[float, float] | None] | None = None,
    ) -> None:
        """Fit team classifier using ReID embeddings.

        Args:
            embeddings: ReID embeddings array (N, D).
            track_ids: Optional list of track IDs.
            positions: Optional list of pitch positions.
        """
        # Add embeddings to reid classifier
        if track_ids is not None and positions is not None:
            for i, (tid, pos) in enumerate(zip(track_ids, positions)):
                if pos is not None:
                    self.reid_classifier.add_track_embedding(tid, embeddings[i], pos)

        self.reid_classifier.fit(embeddings)

        # Sync cluster labels
        self.cluster_labels = self.reid_classifier.cluster_labels.copy()

    def predict_with_reid(self, embedding: np.ndarray) -> int:
        """Predict team using ReID embedding.

        Args:
            embedding: ReID embedding vector.

        Returns:
            Cluster ID.
        """
        return self.reid_classifier.predict(embedding)

    def get_team_from_reid(self, cluster_id: int) -> str:
        """Get team label from ReID classification.

        Args:
            cluster_id: Cluster identifier.

        Returns:
            Team label string.
        """
        return self.reid_classifier.get_team(cluster_id)

    # =========================================================================
    # Legacy color-based methods (fallback)
    # =========================================================================

    def extract_color_features(
        self,
        crop: np.ndarray,
        mask_lower: bool = True,
    ) -> np.ndarray:
        """Extract color features from a player crop (legacy).

        Args:
            crop: Player image crop (BGR format).
            mask_lower: Whether to mask lower body.

        Returns:
            Feature vector.
        """
        if crop.size == 0:
            return np.zeros(256 * 3 if self.feature_type == "histogram" else 3)

        h, w = crop.shape[:2]
        if mask_lower and h > 20:
            crop = crop[:int(h * 0.6), :]

        mask = None
        if self.use_background_removal:
            mask, crop_masked = self.remove_grass_background(crop)
            foreground_ratio = np.sum(mask > 0) / (mask.size + 1e-8)
            if foreground_ratio > 0.1:
                crop = crop_masked
            else:
                mask = None

        if self.color_space == "hsv":
            color_img = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        elif self.color_space == "lab":
            color_img = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        else:
            color_img = crop

        if self.feature_type == "histogram":
            features = []
            for i in range(3):
                hist = cv2.calcHist([color_img], [i], mask, [256], [0, 256])
                hist = hist.flatten() / (hist.sum() + 1e-8)
                features.extend(hist)
            return np.array(features)
        else:
            if mask is not None and np.sum(mask > 0) > 0:
                foreground_pixels = color_img[mask > 0]
                return np.mean(foreground_pixels, axis=0)
            else:
                return np.mean(color_img.reshape(-1, 3), axis=0)

    def extract_batch_features(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract features from multiple crops (legacy).

        Args:
            crops: List of player image crops.

        Returns:
            Feature matrix (N, feature_dim).
        """
        features = []
        for crop in crops:
            feat = self.extract_color_features(crop)
            features.append(feat)
        return np.array(features) if features else np.array([])

    def fit(self, crops: list[np.ndarray]) -> None:
        """Fit classifier on player crops (legacy color-based method).

        Note: For ReID-based classification, use fit_with_reid() instead.

        Args:
            crops: List of player image crops.
        """
        player_crops = []
        for crop in crops:
            if crop.size == 0:
                continue
            # Skip referee/goalkeeper for team clustering
            # Note: This uses legacy color detection - not recommended
            if not self._is_referee_legacy(crop) and not self._is_goalkeeper_legacy(crop):
                player_crops.append(crop)

        n_clusters = min(2, len(player_crops))
        if n_clusters < 2:
            return

        features = self.extract_batch_features(player_crops)
        if len(features) < n_clusters:
            return

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        self.kmeans.fit(features)
        self.cluster_labels = {0: "left", 1: "right"}

    def predict(self, crop: np.ndarray) -> int:
        """Predict cluster for a single crop (legacy).

        Note: For ReID-based classification, use predict_with_reid() instead.

        Args:
            crop: Player image crop.

        Returns:
            Cluster ID.
        """
        if self.kmeans is None:
            return -1

        features = self.extract_color_features(crop).reshape(1, -1)
        return int(self.kmeans.predict(features)[0])

    def predict_batch(self, crops: list[np.ndarray]) -> list[int]:
        """Predict clusters for multiple crops (legacy).

        Args:
            crops: List of player image crops.

        Returns:
            List of cluster IDs.
        """
        if self.kmeans is None or not crops:
            return [-1] * len(crops)

        features = self.extract_batch_features(crops)
        return self.kmeans.predict(features).tolist()

    def get_team(self, cluster_id: int) -> str:
        """Get team label for a cluster.

        Args:
            cluster_id: Cluster identifier.

        Returns:
            Team label string.
        """
        return self.cluster_labels.get(cluster_id, "unknown")

    # =========================================================================
    # Role Classification (should ideally come from detector)
    # These are legacy color-based fallbacks - NOT recommended
    # =========================================================================

    def _is_referee_legacy(self, crop: np.ndarray) -> bool:
        """Check if crop is likely a referee using color (LEGACY - not recommended).

        Note: This should ideally come from a fine-tuned YOLOv8 detector output.
        Color-based detection is unreliable and should only be used as fallback.

        Args:
            crop: Player image crop.

        Returns:
            True if likely referee.
        """
        REFEREE_COLORS = [
            {"lower": np.array([0, 0, 0]), "upper": np.array([180, 80, 120])},
            {"lower": np.array([20, 100, 100]), "upper": np.array([35, 255, 255])},
            {"lower": np.array([140, 50, 50]), "upper": np.array([170, 255, 255])},
        ]

        if crop.size == 0:
            return False

        h = crop.shape[0]
        upper = crop[:int(h * 0.6), :]

        fg_mask, _ = self.remove_grass_background(upper)
        fg_pixels = np.sum(fg_mask > 0)

        if fg_pixels < 100:
            return False

        hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)

        for color_range in REFEREE_COLORS:
            color_mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
            combined_mask = cv2.bitwise_and(color_mask, fg_mask)
            ratio = np.sum(combined_mask > 0) / (fg_pixels + 1e-8)
            if ratio > 0.4:
                return True

        return False

    def _is_goalkeeper_legacy(self, crop: np.ndarray) -> bool:
        """Check if crop is likely a goalkeeper using color (LEGACY - not recommended).

        Note: This should ideally come from a fine-tuned YOLOv8 detector output.
        Color-based detection is unreliable and should only be used as fallback.

        Args:
            crop: Player image crop.

        Returns:
            True if likely goalkeeper.
        """
        GOALKEEPER_COLORS = [
            {"lower": np.array([45, 180, 180]), "upper": np.array([75, 255, 255])},
            {"lower": np.array([22, 150, 150]), "upper": np.array([38, 255, 255])},
            {"lower": np.array([10, 150, 150]), "upper": np.array([22, 255, 255])},
            {"lower": np.array([150, 100, 100]), "upper": np.array([170, 255, 255])},
        ]

        if crop.size == 0:
            return False

        h = crop.shape[0]
        upper = crop[:int(h * 0.6), :]

        fg_mask, _ = self.remove_grass_background(upper)
        fg_pixels = np.sum(fg_mask > 0)

        if fg_pixels < 100:
            return False

        hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)

        for color_range in GOALKEEPER_COLORS:
            color_mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
            combined_mask = cv2.bitwise_and(color_mask, fg_mask)
            ratio = np.sum(combined_mask > 0) / (fg_pixels + 1e-8)
            if ratio > 0.5:
                return True

        return False

    # Keep old method names for backward compatibility (aliased to legacy methods)
    def is_referee(self, crop: np.ndarray) -> bool:
        """Check if crop is a referee (LEGACY - prefer detector output).

        Deprecated: This uses unreliable color-based detection.
        Role should come from a fine-tuned YOLOv8 detector output.
        """
        return self._is_referee_legacy(crop)

    def is_goalkeeper(self, crop: np.ndarray) -> bool:
        """Check if crop is a goalkeeper (LEGACY - prefer detector output).

        Deprecated: This uses unreliable color-based detection.
        Role should come from a fine-tuned YOLOv8 detector output.
        """
        return self._is_goalkeeper_legacy(crop)

    def classify_role(self, crop: np.ndarray) -> str:
        """Classify player role (LEGACY color-based fallback).

        Note: This method uses unreliable color-based detection.
        For production use, role should come from a fine-tuned YOLOv8 detector.

        Args:
            crop: Player image crop.

        Returns:
            Role string: "player", "referee", or "goalkeeper".
        """
        if self._is_goalkeeper_legacy(crop):
            return "goalkeeper"
        if self._is_referee_legacy(crop):
            return "referee"
        return "player"

    def classify_role_from_detection(self, detection: dict[str, Any]) -> str:
        """Get role from detector output (RECOMMENDED approach).

        This is the proper way to get role - from a fine-tuned detector
        that outputs class labels for player/goalkeeper/referee.

        Args:
            detection: Detection dictionary with 'class_name' or 'role' field.

        Returns:
            Role string.
        """
        # Check if detector provides role
        if "role" in detection:
            return detection["role"]

        # Check class name from detector
        class_name = detection.get("class_name", "").lower()
        if class_name in ["player", "goalkeeper", "referee"]:
            return class_name

        # Fallback to legacy color-based method
        return "player"  # Default, since we can't determine without crop


class TeamClassifier:
    """Enhanced team classifier using ReID embeddings and field position.

    This is the recommended approach from the SoccerMaster paper.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize team classifier.

        Args:
            config: Classification configuration.
        """
        self.config = config or {}
        self.reid_classifier = ReIDTeamClassifier(config)

        # Legacy fallback
        self.role_classifier = RoleClassifier(config)

        # Team assignments by track
        self.team_assignments: dict[int, str] = {}

    def initialize_with_reid(
        self,
        embeddings: np.ndarray,
        track_ids: list[int] | None = None,
        positions: list[tuple[float, float] | None] | None = None,
    ) -> None:
        """Initialize classifier with ReID embeddings.

        Args:
            embeddings: ReID embeddings (N, D).
            track_ids: Optional track IDs.
            positions: Optional pitch positions.
        """
        if positions is not None:
            for i, (tid, pos) in enumerate(zip(track_ids or range(len(embeddings)), positions)):
                if pos is not None and i < len(embeddings):
                    self.reid_classifier.add_track_embedding(tid, embeddings[i], pos)

        self.reid_classifier.fit(embeddings)

    def classify_team(
        self,
        embedding: np.ndarray,
        position: tuple[float, float] | None = None,
    ) -> str:
        """Classify team using ReID embedding.

        Args:
            embedding: ReID embedding vector.
            position: Optional pitch position.

        Returns:
            Team label ("left", "right", or "unknown").
        """
        cluster_id = self.reid_classifier.predict(embedding)
        return self.reid_classifier.get_team(cluster_id)

    def update_track_assignment(
        self,
        track_id: int,
        embedding: np.ndarray,
        position: tuple[float, float] | None = None,
    ) -> str:
        """Update team assignment for a track.

        Args:
            track_id: Track identifier.
            embedding: ReID embedding.
            position: Optional pitch position.

        Returns:
            Team label.
        """
        team = self.classify_team(embedding, position)

        if track_id not in self.team_assignments:
            self.team_assignments[track_id] = team

        return self.team_assignments.get(track_id, team)

    # Legacy compatibility methods
    def initialize_from_frame(
        self,
        crops: list[np.ndarray],
        positions: list[tuple[float, float]] | None = None,
    ) -> None:
        """Initialize classifier from crops (legacy fallback).

        Note: Prefer initialize_with_reid() with ReID embeddings.
        """
        self.role_classifier.fit(crops)

    def classify(
        self,
        crop: np.ndarray,
        position: tuple[float, float] | None = None,
    ) -> tuple[str, str]:
        """Classify role and team (legacy fallback).

        Note: This uses color-based classification. For production,
        use classify_team() with ReID embeddings and get role from detector.

        Args:
            crop: Player image crop.
            position: Optional pitch position.

        Returns:
            Tuple of (role, team).
        """
        role = self.role_classifier.classify_role(crop)

        if role != "player":
            return role, "none"

        cluster_id = self.role_classifier.predict(crop)
        team = self.role_classifier.get_team(cluster_id)

        return role, team
