"""
CHECK STEP 1 OUTPUTS - Inspect what was actually saved during step 1 execution

Purpose: Review the JSON metadata and crop files saved after step 1 runs
to understand exactly what the pipeline detected and saved.

Usage:
  python troubleshooting/check_step1_outputs.py <output_root> [--camera CAMERA] [--video VIDEO_STEM]

Examples:
  # Check all Step 1 outputs
  python troubleshooting/check_step1_outputs.py output_root

  # Check specific camera
  python troubleshooting/check_step1_outputs.py output_root --camera FAR3_CUTE

  # Check specific video
  python troubleshooting/check_step1_outputs.py output_root --video TRI5_SCALE_2025-06-09_10_04
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
import argparse
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_metadata_file(metadata_path: Path) -> Dict:
    """Analyze a single _metadata.json file."""
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to read {metadata_path}: {e}")
        return None


def check_step1_outputs(output_root: str, camera: Optional[str] = None,
                       video_stem: Optional[str] = None):
    """Check all Step 1 outputs."""
    output_path = Path(output_root)
    step1_dir = output_path / "step1_ring_crops"

    if not step1_dir.exists():
        logger.error(f"Step 1 output directory not found: {step1_dir}")
        return

    logger.info("=" * 80)
    logger.info("STEP 1 OUTPUT ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Output root: {output_root}")
    logger.info(f"Step 1 directory: {step1_dir}")
    logger.info("")

    # Find all video directories
    total_videos = 0
    total_detections = 0
    videos_with_detections = 0
    videos_no_detections = 0
    total_crops_saved = 0

    stats_by_camera = defaultdict(lambda: {
        'videos': 0,
        'detections': 0,
        'crops': 0,
        'videos_no_detection': 0,
    })

    # Iterate through camera directories
    for camera_dir in step1_dir.iterdir():
        if not camera_dir.is_dir():
            continue

        camera_name = camera_dir.name

        # Filter by camera if specified
        if camera and camera.upper() not in camera_name.upper():
            continue

        logger.info(f"\nCamera: {camera_name}")
        logger.info("-" * 80)

        # Iterate through video directories
        for video_dir in camera_dir.iterdir():
            if not video_dir.is_dir():
                continue

            video_name = video_dir.name

            # Filter by video stem if specified
            if video_stem and video_stem.upper() not in video_name.upper():
                continue

            total_videos += 1
            stats_by_camera[camera_name]['videos'] += 1

            # Check for metadata file
            metadata_path = video_dir / "_metadata.json"
            if not metadata_path.exists():
                logger.warning(f"  Video {video_name}: NO METADATA FILE")
                continue

            # Analyze metadata
            metadata = analyze_metadata_file(metadata_path)
            if not metadata:
                continue

            detection_count = metadata.get('ring_detect_count', 0)
            frames_total = metadata.get('frames_total', 0)
            frames_processed = metadata.get('frames_processed', 0)
            detections_list = metadata.get('detections', [])

            total_detections += detection_count
            stats_by_camera[camera_name]['detections'] += detection_count

            if detection_count == 0:
                videos_no_detections += 1
                stats_by_camera[camera_name]['videos_no_detection'] += 1
                logger.info(f"  Video {video_name}: 0 detections (processed {frames_processed}/{frames_total} frames)")
            else:
                videos_with_detections += 1
                logger.info(f"  Video {video_name}: {detection_count} detections (from {frames_processed} frames)")

                # Count valid crops saved
                crops_count = 0
                for crop_file in video_dir.glob("*_ring.png"):
                    crops_count += 1

                stats_by_camera[camera_name]['crops'] += crops_count
                total_crops_saved += crops_count

                # Show details of first few detections
                valid_detections = [d for d in detections_list if d.get('detected') and d.get('confidence', 0) >= 0.7]
                logger.info(f"    - Crops saved: {crops_count}")
                logger.info(f"    - Valid detections (conf >= 0.7): {len(valid_detections)}")

                # Sample detection details
                if valid_detections and len(valid_detections) <= 3:
                    for i, det in enumerate(valid_detections, 1):
                        if det.get('bbox'):
                            bbox = det['bbox']
                            logger.info(f"      Detection #{i}: frame {det['frame_idx']}, "
                                      f"conf={det.get('confidence', 0):.3f}, "
                                      f"bbox=({bbox['x_norm']:.3f}, {bbox['y_norm']:.3f}, "
                                      f"{bbox['w_norm']:.3f}x{bbox['h_norm']:.3f})")

    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total videos processed: {total_videos}")
    logger.info(f"Videos with detections: {videos_with_detections}")
    logger.info(f"Videos with NO detections: {videos_no_detections}")
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Total crop files saved: {total_crops_saved}")

    if stats_by_camera:
        logger.info("")
        logger.info("BREAKDOWN BY CAMERA:")
        logger.info("-" * 80)
        for camera_name, stats in sorted(stats_by_camera.items()):
            logger.info(f"{camera_name}:")
            logger.info(f"  Videos: {stats['videos']}")
            logger.info(f"  Videos with no detection: {stats['videos_no_detection']}")
            logger.info(f"  Total detections: {stats['detections']}")
            logger.info(f"  Crop files: {stats['crops']}")

    if videos_no_detections > 0:
        logger.warning("")
        logger.warning("⚠️  ATTENTION: Multiple videos with NO detections detected!")
        logger.warning("This suggests either:")
        logger.warning("  1. The confidence threshold is too high")
        logger.warning("  2. The model isn't detecting rings properly")
        logger.warning("  3. ROI filtering is removing all detections")
        logger.warning("  4. Frame extraction or YOLO inference issues")
        logger.warning("")
        logger.warning("Run diagnose_step1.py on a sample video for detailed troubleshooting.")


def main():
    parser = argparse.ArgumentParser(
        description="Check and analyze Step 1 outputs"
    )
    parser.add_argument("output_root", help="Output root directory from pipeline")
    parser.add_argument("--camera", help="Filter by camera name (optional)")
    parser.add_argument("--video", dest="video_stem", help="Filter by video stem (optional)")

    args = parser.parse_args()

    check_step1_outputs(
        output_root=args.output_root,
        camera=args.camera,
        video_stem=args.video_stem
    )


if __name__ == "__main__":
    main()
