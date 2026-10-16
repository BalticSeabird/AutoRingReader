"""Configuration dataclasses for the step-based processing pipeline."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import yaml
import logging

try:
    from shapely import geometry
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    geometry = None

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_name: str
    model_root_path: str

    def get_model_path(self) -> Path:
        """Get full path to model file."""
        return Path(self.model_root_path) / self.model_name


@dataclass
class InputConfig:
    """Input configuration."""
    video_root_path: str
    cameras: List[str]
    resume: bool = True


@dataclass
class Step1Config:
    """Configuration for Step 1: Ring Detection."""
    n_processes_cpu: int = 16
    n_processes_gpu: int = 2
    gpu_devices: List[int] = None
    batch_size: int = 16
    frame_sample_rate: int = 1
    output_format: str = "jpg"
    save_crops: bool = True
    crop_size: int = 224
    conf_threshold: float = 0.7
    iou_threshold: float = 0.7
    max_det: int = 1

    def __post_init__(self):
        if self.gpu_devices is None:
            self.gpu_devices = [0, 1]


@dataclass
class Step2Config:
    """Configuration for Step 2: Tag Detection."""
    n_processes_cpu: int = 16
    n_processes_gpu: int = 2
    gpu_devices: List[int] = None
    batch_size: int = 16
    conf_threshold: float = 0.7
    iou_threshold: float = 0.7
    max_det: int = 2
    tag_height: int = 96
    output_format: str = "jpg"

    def __post_init__(self):
        if self.gpu_devices is None:
            self.gpu_devices = [0, 1]


@dataclass
class Step3Config:
    """Configuration for Step 3: Text Recognition."""
    n_processes_cpu: int = 32
    n_processes_gpu: int = 1
    gpu_devices: List[int] = None
    batch_size: int = 32
    conf_threshold: float = 0.90
    rotational_validation: bool = True
    output_format: str = "yaml"

    def __post_init__(self):
        if self.gpu_devices is None:
            self.gpu_devices = [0]


@dataclass
class ModelsConfig:
    """Configuration for all models."""
    ring_detector: ModelConfig
    tag_detector: ModelConfig
    text_recognition: ModelConfig


class ROIConfig:
    """Configuration for Region of Interest (ROI) masks."""

    def __init__(self, rois_yaml_path: str = "rois_config.yaml"):
        """Load ROI configuration from YAML file."""
        self.rois_data = {}
        self.rois_polygons = {}  # Cache for geometry.Polygon objects

        roi_path = Path(rois_yaml_path)
        if roi_path.exists():
            try:
                with open(roi_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.rois_data = config.get('rois', {})
                logger.info(f"Loaded ROI configuration from {rois_yaml_path}")
            except Exception as e:
                logger.warning(f"Failed to load ROI configuration: {e}")
                self.rois_data = {}
        else:
            logger.warning(f"ROI configuration file not found: {rois_yaml_path}")

    def get_roi_for_camera(self, camera: str, year: Optional[int] = None) -> Optional[Any]:
        """
        Get ROI polygon for a camera (station_region format).

        Args:
            camera: Camera name like "FAR3_CUTE" or "FAR6_SCALE"
            year: Optional year for year-specific ROIs (uses "default" if not found)

        Returns:
            geometry.Polygon or None if no ROI found or shapely not available
        """
        if not self.rois_data or not SHAPELY_AVAILABLE:
            return None

        # Parse camera name: "STATION_REGION" -> ("STATION", "REGION")
        parts = camera.split('_')
        if len(parts) < 2:
            logger.warning(f"Invalid camera name format: {camera}")
            return None

        station = parts[0]  # e.g., "FAR3"
        region = parts[1].lower()   # e.g., "cute", "scale" (lowercase for YAML)

        # Try to get ROI from config: rois -> station -> region -> year -> polygon
        try:
            year_key = str(year) if year else "default"

            if station not in self.rois_data:
                logger.debug(f"No ROI data for station: {station}")
                return None

            if region not in self.rois_data[station]:
                logger.debug(f"No ROI data for station {station}, region {region}")
                return None

            region_data = self.rois_data[station][region]

            # Try year-specific first, fall back to "default"
            polygon_data = None
            if year_key in region_data:
                polygon_data = region_data[year_key]
            elif "default" in region_data:
                polygon_data = region_data["default"]
            else:
                logger.debug(f"No ROI for {station}/{region}/{year_key}")
                return None

            # Convert list of [x, y] coords to geometry.Polygon
            coords = polygon_data.get('polygon', [])
            if not coords or len(coords) < 3:
                logger.warning(f"Invalid polygon for {camera}: {coords}")
                return None

            # Cache the polygon
            cache_key = f"{camera}_{year_key}"
            if cache_key not in self.rois_polygons:
                self.rois_polygons[cache_key] = geometry.Polygon(coords)

            return self.rois_polygons[cache_key]

        except Exception as e:
            logger.error(f"Error loading ROI for {camera}: {e}")
            return None

    def get_roi_bbox(self, camera: str, year: Optional[int] = None) -> Optional[Tuple[float, float, float, float]]:
        """
        Get bounding box (xmin, ymin, xmax, ymax) for ROI polygon.

        Works without shapely by computing bbox directly from coordinates.

        Returns:
            Tuple of (xmin, ymin, xmax, ymax) or None
        """
        if not self.rois_data:
            return None

        # Parse camera name
        parts = camera.split('_')
        if len(parts) < 2:
            return None

        station = parts[0]
        region = parts[1].lower()  # Convert to lowercase for YAML match

        try:
            year_key = str(year) if year else "default"

            if station not in self.rois_data or region not in self.rois_data[station]:
                return None

            region_data = self.rois_data[station][region]

            # Try year-specific first, fall back to "default"
            polygon_data = None
            if year_key in region_data:
                polygon_data = region_data[year_key]
            elif "default" in region_data:
                polygon_data = region_data["default"]
            else:
                return None

            # Extract coordinates and compute bbox directly
            coords = polygon_data.get('polygon', [])
            if not coords or len(coords) < 3:
                return None

            # Compute bounding box from coordinates
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            return (min(xs), min(ys), max(xs), max(ys))

        except Exception as e:
            logger.error(f"Error computing ROI bbox for {camera}: {e}")
            return None


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    input: InputConfig
    models: ModelsConfig
    output_root_path: str
    step1: Step1Config
    step2: Step2Config
    step3: Step3Config
    events_db_path: str = "log/events.db"
    roi_config: Optional[ROIConfig] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        yaml_file = Path(yaml_path)
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        # Parse nested configs
        input_cfg = InputConfig(**data['input'])

        models_data = data['models']
        models_cfg = ModelsConfig(
            ring_detector=ModelConfig(**models_data['ring_detector']),
            tag_detector=ModelConfig(**models_data['tag_detector']),
            text_recognition=ModelConfig(**models_data['text_recognition']),
        )

        step1_cfg = Step1Config(**data.get('step1', {}))
        step2_cfg = Step2Config(**data.get('step2', {}))
        step3_cfg = Step3Config(**data.get('step3', {}))

        # Load ROI configuration relative to the batch YAML if a relative path is used.
        roi_config_path = data.get('roi_config_path')
        if roi_config_path is None:
            roi_config_path = yaml_file.parent / 'rois_config.yaml'
        else:
            roi_config_path = Path(roi_config_path)
            if not roi_config_path.is_absolute():
                candidate_path = yaml_file.parent / roi_config_path
                if candidate_path.exists():
                    roi_config_path = candidate_path

        roi_config = ROIConfig(str(roi_config_path))

        return cls(
            input=input_cfg,
            models=models_cfg,
            output_root_path=data['output_root_path'],
            events_db_path=data.get('events_db_path', 'log/events.db'),
            step1=step1_cfg,
            step2=step2_cfg,
            step3=step3_cfg,
            roi_config=roi_config,
        )

    def get_output_dir(self) -> Path:
        """Get output root directory."""
        return Path(self.output_root_path)

    def get_db_path(self) -> Path:
        """Get path to events database."""
        db_path = Path(self.events_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path
