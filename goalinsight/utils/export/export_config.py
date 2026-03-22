"""Configuration for intermediate result export."""

from dataclasses import dataclass, field


@dataclass
class DetectionExportConfig:
    """Detection export configuration."""
    enabled: bool = True
    save_json: bool = True
    save_summary: bool = True
    include_filtered: bool = True
    frame_interval: int = 10


@dataclass
class ReidExportConfig:
    """ReID export configuration."""
    enabled: bool = True
    save_embeddings: bool = True
    save_crops: bool = False
    save_roles: bool = True
    frame_interval: int = 10


@dataclass
class CalibrationExportConfig:
    """Calibration export configuration."""
    enabled: bool = True
    save_json: bool = True
    save_homography: bool = True
    frame_interval: int = 10


@dataclass
class TrackingExportConfig:
    """Tracking export configuration."""
    enabled: bool = True
    save_json: bool = True
    save_trajectories: bool = True
    frame_interval: int = 1


@dataclass
class ClusteringExportConfig:
    """Clustering export configuration."""
    enabled: bool = True
    save_summary: bool = True
    save_embeddings: bool = True


@dataclass
class VisualizationExportConfig:
    """Visualization export configuration."""
    enabled: bool = True
    save_detection_vis: bool = False
    save_calibration_vis: bool = False
    show_filter_reasons: bool = True
    frame_interval: int = 50


@dataclass
class IntermediateExportConfig:
    """Main configuration for intermediate result export."""
    enabled: bool = False
    output_subdir: str = "intermediates"

    detection: DetectionExportConfig = field(default_factory=DetectionExportConfig)
    reid: ReidExportConfig = field(default_factory=ReidExportConfig)
    calibration: CalibrationExportConfig = field(default_factory=CalibrationExportConfig)
    tracking: TrackingExportConfig = field(default_factory=TrackingExportConfig)
    clustering: ClusteringExportConfig = field(default_factory=ClusteringExportConfig)
    visualization: VisualizationExportConfig = field(default_factory=VisualizationExportConfig)

    @classmethod
    def from_dict(cls, config: dict) -> "IntermediateExportConfig":
        """Create config from dictionary."""
        if not config:
            return cls()

        detection = DetectionExportConfig(**config.get('detection', {}))
        reid = ReidExportConfig(**config.get('reid', {}))
        calibration = CalibrationExportConfig(**config.get('calibration', {}))
        tracking = TrackingExportConfig(**config.get('tracking', {}))
        clustering = ClusteringExportConfig(**config.get('clustering', {}))
        visualization = VisualizationExportConfig(**config.get('visualization', {}))

        return cls(
            enabled=config.get('enabled', False),
            output_subdir=config.get('output_subdir', 'intermediates'),
            detection=detection,
            reid=reid,
            calibration=calibration,
            tracking=tracking,
            clustering=clustering,
            visualization=visualization,
        )

    @classmethod
    def all_enabled(cls, output_subdir: str = "intermediates") -> "IntermediateExportConfig":
        """Create config with all exports enabled."""
        return cls(
            enabled=True,
            output_subdir=output_subdir,
            detection=DetectionExportConfig(enabled=True, frame_interval=1),
            reid=ReidExportConfig(enabled=True, save_crops=True, frame_interval=1),
            calibration=CalibrationExportConfig(enabled=True, frame_interval=1),
            tracking=TrackingExportConfig(enabled=True, frame_interval=1),
            clustering=ClusteringExportConfig(enabled=True),
            visualization=VisualizationExportConfig(enabled=True, save_detection_vis=True, save_calibration_vis=True),
        )
