"""Rebuild the current pipeline database from completed step outputs.

This utility does not run inference.  It reads Step 1 metadata, Step 2 crop
filenames, and Step 3 YAML summaries and writes the same SQLite schema used by
the live pipeline.

RUN EXAMPLE
venv/bin/python utils/rebuild_database.py \
  --step1 /mnt/xdisk/data/work/bsp/ring_read_260416/step1_ring_crops \
  --step2 /mnt/xdisk/data/work/bsp/ring_read_260416/step2_tag_crops \
  --step3 /mnt/xdisk/data/work/bsp/ring_read_260416/step3_ocr_results \
  --database out/OCR_readings_2023-2025.db

"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Make direct execution from the repository root work as documented.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.pipeline_logger import PipelineLogger


LOGGER = logging.getLogger("rebuild_database")
STEP1_CROP_PATTERN = "*_ring.png"
STEP2_CROP_PATTERN = "*_ring_tag_*.jpg"


@dataclass
class RebuiltVideo:
    """Recoverable database values for one camera/video event."""

    camera: str
    video_stem: str
    video_path: str = ""
    step1: dict[str, Any] = field(default_factory=dict)
    step2: dict[str, Any] = field(default_factory=dict)
    step3: dict[str, Any] = field(default_factory=dict)


def _step_root(path: Path, step_name: str) -> Path:
    """Accept either a step directory or an output root containing it."""
    if path.name == step_name:
        return path
    candidate = path / step_name
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"{step_name} directory not found at {path} or {candidate}"
    )


def _get_video(
    videos: dict[tuple[str, str], RebuiltVideo], camera: str, stem: str
) -> RebuiltVideo:
    key = (camera, stem)
    if key not in videos:
        videos[key] = RebuiltVideo(camera=camera, video_stem=stem)
    return videos[key]


def _read_step1(step1_path: Path, videos: dict[tuple[str, str], RebuiltVideo]) -> None:
    root = _step_root(step1_path, "step1_ring_crops")
    for camera_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for video_dir in sorted(path for path in camera_dir.iterdir() if path.is_dir()):
            crop_count = len(list(video_dir.glob(STEP1_CROP_PATTERN)))
            metadata_path = video_dir / "_metadata.json"
            metadata: dict[str, Any] = {}
            if metadata_path.is_file():
                try:
                    with metadata_path.open(encoding="utf-8") as file:
                        loaded = json.load(file)
                    if not isinstance(loaded, dict):
                        raise ValueError("metadata root is not an object")
                    metadata = loaded
                except (OSError, json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"Could not read {metadata_path}: {error}") from error

            video = _get_video(videos, camera_dir.name, video_dir.name)
            video.video_path = str(metadata.get("video_path", ""))
            frames_processed = metadata.get("frames_processed")
            ring_count = metadata.get("ring_detect_count", crop_count)
            video.step1 = {
                "status": "success",
                "frames_total": metadata.get("frames_total"),
                "frames_processed": frames_processed,
                "ring_detect_count": ring_count,
                "ring_detect_rate": (
                    ring_count / frames_processed
                    if isinstance(ring_count, (int, float))
                    and isinstance(frames_processed, (int, float))
                    and frames_processed > 0
                    else None
                ),
                "output_path": str(video_dir),
                "queue_file": str(metadata_path) if metadata_path.is_file() else None,
            }


def _step2_frame_id(path: Path) -> str:
    return path.name.split("_ring_tag_", 1)[0]


def _read_step2(step2_path: Path, videos: dict[tuple[str, str], RebuiltVideo]) -> None:
    root = _step_root(step2_path, "step2_tag_crops")
    for camera_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for video_dir in sorted(path for path in camera_dir.iterdir() if path.is_dir()):
            crops = list(video_dir.glob(STEP2_CROP_PATTERN))
            frame_ids = {_step2_frame_id(path) for path in crops}
            video = _get_video(videos, camera_dir.name, video_dir.name)
            rings_processed = video.step1.get("ring_detect_count")
            video.step2 = {
                "status": "success",
                "rings_processed": rings_processed,
                "tags_detected_count": len(crops),
                "tags_detected_frames": len(frame_ids),
                "tag_detection_rate": (
                    len(crops) / rings_processed
                    if isinstance(rings_processed, (int, float)) and rings_processed > 0
                    else None
                ),
                "output_path": str(video_dir),
                "queue_file": None,
            }


def _read_step3(step3_path: Path, videos: dict[tuple[str, str], RebuiltVideo]) -> None:
    root = _step_root(step3_path, "step3_ocr_results")
    for camera_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for video_dir in sorted(path for path in camera_dir.iterdir() if path.is_dir()):
            yaml_files = sorted(video_dir.glob("*.yaml"))
            if not yaml_files:
                continue
            yaml_path = yaml_files[0]
            try:
                with yaml_path.open(encoding="utf-8") as file:
                    data = yaml.safe_load(file)
                if not isinstance(data, dict):
                    raise ValueError("YAML root is not an object")
            except (OSError, yaml.YAMLError, ValueError) as error:
                raise ValueError(f"Could not read {yaml_path}: {error}") from error

            video = _get_video(videos, camera_dir.name, video_dir.name)
            video.step3 = {
                "status": "success",
                "tags_processed": data.get("detection_count"),
                "ocr_matches_count": 1 if data.get("final_tag") else 0,
                "ocr_success_rate": 1.0 if data.get("final_tag") else 0.0,
                "letters_conf_min_mean": data.get("letters_confidence"),
                "numbers_conf_min_mean": data.get("numbers_confidence"),
                "letters_ema_min_mean": data.get("letters_confidence"),
                "numbers_ema_min_mean": data.get("numbers_confidence"),
                "output_path": str(yaml_path),
                "final_tag": data.get("final_tag"),
                "letters_conf": data.get("letters_confidence"),
                "numbers_conf": data.get("numbers_confidence"),
                "ocr_matches": 1 if data.get("final_tag") else 0,
            }


def _summary(video: RebuiltVideo) -> dict[str, Any]:
    step1, step2, step3 = video.step1, video.step2, video.step3
    return {
        "camera": video.camera,
        "video_stem": video.video_stem,
        "video_path": video.video_path or f"{video.camera}/{video.video_stem}",
        "step1_status": step1.get("status", "pending"),
        "step1_frames_total": step1.get("frames_total"),
        "step1_frames_processed": step1.get("frames_processed"),
        "step1_ring_detections": step1.get("ring_detect_count"),
        "step1_ring_conf_mean": None,
        "step1_ring_conf_max": None,
        "step1_time": None,
        "step1_error": None,
        "step2_status": step2.get("status", "pending"),
        "step2_ring_crops_count": step2.get("rings_processed"),
        "step2_tags_detected": step2.get("tags_detected_count"),
        "step2_tag_conf_mean": None,
        "step2_tag_conf_max": None,
        "step2_time": None,
        "step2_error": None,
        "step3_status": step3.get("status", "pending"),
        "step3_final_tag": step3.get("final_tag"),
        "step3_ocr_matches": step3.get("ocr_matches"),
        "step3_letters_conf": step3.get("letters_conf"),
        "step3_numbers_conf": step3.get("numbers_conf"),
        "step3_time": None,
        "step3_error": None,
    }


def rebuild_database(
    step1_path: Path, step2_path: Path, step3_path: Path, database_path: Path
) -> int:
    """Read all supplied outputs and rebuild ``database_path``."""
    videos: dict[tuple[str, str], RebuiltVideo] = {}
    _read_step1(step1_path, videos)
    _read_step2(step2_path, videos)
    _read_step3(step3_path, videos)
    if not videos:
        raise ValueError("No step output events were found")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_logger = PipelineLogger(str(database_path), LOGGER)
    try:
        with sqlite3.connect(database_path) as connection:
            connection.execute("DELETE FROM step1_ring_detections")
            connection.execute("DELETE FROM step2_tag_detections")
            connection.execute("DELETE FROM step3_text_recognition")
            connection.execute("DELETE FROM final_summary")
            connection.commit()

        for video in videos.values():
            if video.step1:
                if not pipeline_logger.log_step1_result(
                    video.video_path or video.video_stem, video.camera, video.step1
                ):
                    raise RuntimeError(f"Failed to write Step 1 row for {video.video_stem}")
            if video.step2:
                if not pipeline_logger.log_step2_result(
                    video.video_path or video.video_stem, video.camera, video.step2
                ):
                    raise RuntimeError(f"Failed to write Step 2 row for {video.video_stem}")
            if video.step3:
                if not pipeline_logger.log_step3_result(
                    video.video_path or video.video_stem, video.camera, video.step3
                ):
                    raise RuntimeError(f"Failed to write Step 3 row for {video.video_stem}")
        if not pipeline_logger.replace_final_summary([_summary(video) for video in videos.values()]):
            raise RuntimeError("Failed to write final_summary")
    finally:
        pipeline_logger.close()

    LOGGER.info("Rebuilt %d video events in %s", len(videos), database_path)
    return len(videos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the current SQLite database from existing Step 1-3 outputs."
    )
    parser.add_argument("--step1", type=Path, required=True, help="Step 1 output directory or output root")
    parser.add_argument("--step2", type=Path, required=True, help="Step 2 output directory or output root")
    parser.add_argument("--step3", type=Path, required=True, help="Step 3 output directory or output root")
    parser.add_argument("--database", type=Path, required=True, help="Database to create or replace")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    return rebuild_database(args.step1, args.step2, args.step3, args.database)


if __name__ == "__main__":
    raise SystemExit(main())
