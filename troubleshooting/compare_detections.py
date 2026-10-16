"""
COMPARE SYSTEM VS VISUAL DETECTIONS
Manually inspect frames and compare with automated detection output

Purpose: Load video and YOLO metadata, display side-by-side for comparison

Usage:
  python troubleshooting/compare_detections.py <video_path> <metadata_json> [--fps 5]

Example:
  python troubleshooting/compare_detections.py \
    path/to/TRI5_SCALE_2025-06-09_10_04.mp4 \
    output_root/step1_ring_crops/TRI5_SCALE/TRI5_SCALE_2025-06-09_10_04/_metadata.json \
    --fps 5
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import numpy as np
import cv2 as cv

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_utils import sample_frames_from_video

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_metadata(metadata_path: str) -> Dict:
    """Load metadata JSON."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def create_comparison_images(video_path: str, metadata_path: str, output_dir: str,
                             sample_rate: int = 10):
    """Create side-by-side comparison images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    detections = {d['frame_idx']: d for d in metadata.get('detections', [])}

    logger.info(f"Creating comparison images...")
    logger.info(f"Video: {video_path}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Sample rate: every {sample_rate}th frame")

    comparison_count = 0

    for frame_idx, frame in sample_frames_from_video(video_path, sample_rate=sample_rate):
        if frame_idx not in detections:
            continue

        det = detections[frame_idx]
        if not det.get('detected'):
            continue

        height, width = frame.shape[:2]

        # Create comparison image (side-by-side)
        comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
        comparison[:, :width] = frame  # Original frame
        comparison[:, width:] = frame  # Detection overlay (copy)

        # Draw detection on right side
        if det.get('bbox'):
            bbox = det['bbox']
            x_norm, y_norm, w_norm, h_norm = bbox['x_norm'], bbox['y_norm'], bbox['w_norm'], bbox['h_norm']
            confidence = det.get('confidence', 0)

            # Convert to pixel coordinates on RIGHT frame (offset by width)
            x_center, y_center = int(x_norm * width) + width, int(y_norm * height)
            w_pix, h_pix = int(w_norm * width), int(h_norm * height)

            x1, y1 = x_center - w_pix // 2, y_center - h_pix // 2
            x2, y2 = x_center + w_pix // 2, y_center + h_pix // 2

            color = (0, 255, 0) if confidence >= 0.7 else (0, 0, 255)
            cv.rectangle(comparison, (x1, y1), (x2, y2), color, 2)
            cv.putText(comparison, f"{confidence:.3f}", (x1, y1 - 5),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add labels
        cv.putText(comparison, "Original", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(comparison, "Detection", (width + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(comparison, f"Frame {frame_idx}", (10, height - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save
        output_file = output_path / f"comparison_frame_{frame_idx:06d}.jpg"
        cv.imwrite(str(output_file), comparison)
        comparison_count += 1

    logger.info(f"Saved {comparison_count} comparison images to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare system vs visual detections")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("metadata_path", help="Path to _metadata.json from Step 1 output")
    parser.add_argument("--output", default="troubleshooting/comparison_output",
                       help="Output directory for comparison images")
    parser.add_argument("--sample-rate", type=int, default=10,
                       help="Sample every Nth frame for comparison (default: 10)")

    args = parser.parse_args()

    if not Path(args.video_path).exists():
        logger.error(f"Video not found: {args.video_path}")
        sys.exit(1)

    if not Path(args.metadata_path).exists():
        logger.error(f"Metadata not found: {args.metadata_path}")
        sys.exit(1)

    create_comparison_images(args.video_path, args.metadata_path, args.output,
                            sample_rate=args.sample_rate)


if __name__ == "__main__":
    main()
