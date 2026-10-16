#!/usr/bin/env python3
"""
Visualize YOLO ring detections with bounding boxes and confidence scores.
Saves output video with overlaid detections.

Usage:
  python3 troubleshooting/visualize_detections.py \
    --video <path> \
    --model <path> \
    --output <output_video.mp4>
"""

import sys
import argparse
import logging
from pathlib import Path
import cv2 as cv
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ring_detector import RingDetector
from pipeline_utils import get_video_info

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def draw_detection(frame, bbox_norm, confidence, color=(0, 255, 0), thickness=2):
    """Draw bounding box on frame."""
    height, width = frame.shape[:2]

    # Convert normalized coords to pixel coords
    x_norm, y_norm, w_norm, h_norm = bbox_norm

    x_center = int(x_norm * width)
    y_center = int(y_norm * height)
    w_pixel = int(w_norm * width)
    h_pixel = int(h_norm * height)

    x1 = x_center - w_pixel // 2
    y1 = y_center - h_pixel // 2
    x2 = x1 + w_pixel
    y2 = y1 + h_pixel

    # Clamp to frame bounds
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    # Draw rectangle
    cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw confidence text
    text = f"Conf: {confidence:.2f}"
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]

    text_x = x1
    text_y = max(20, y1 - 5)

    # Draw background for text
    cv.rectangle(frame, (text_x, text_y - text_size[1] - 4),
                (text_x + text_size[0] + 4, text_y + 4), color, -1)
    cv.putText(frame, text, (text_x + 2, text_y - 2), font, font_scale, (0, 0, 0), font_thickness)

    return frame


def visualize_video(video_path: str, model_path: str, output_path: str,
                   conf_threshold: float = 0.5, batch_size: int = 16,
                   device_id: int = 0):
    """Process video and save with YOLO detections overlaid."""

    logger.info(f"Video: {video_path}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Confidence threshold: {conf_threshold}")

    # Get video info
    video_info = get_video_info(video_path)
    total_frames = video_info.get('frame_count', 0)
    width = video_info.get('width', 1920)
    height = video_info.get('height', 1080)
    fps = video_info.get('fps', 30)

    logger.info(f"Video: {total_frames} frames, {height}x{width} @ {fps} fps")

    # Load YOLO model
    logger.info("Loading YOLO model...")
    detector = RingDetector(
        model_path,
        conf=conf_threshold,
        iou=0.7,
        max_det=10,  # Allow multiple detections for visualization
        device_id=device_id,
    )

    # Open video for reading
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return False

    # Setup video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    detection_count = 0

    logger.info("Processing frames...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run YOLO inference
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = detector.model.predict(frame_rgb, verbose=False)

            # Draw detections on frame
            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                result = results[0]
                for box in result.boxes:
                    x_norm, y_norm, w_norm, h_norm = box.xywhn[0]
                    confidence = float(box.conf[0])

                    if confidence >= conf_threshold:
                        detection_count += 1
                        draw_detection(frame, (x_norm, y_norm, w_norm, h_norm),
                                     confidence, color=(0, 255, 0), thickness=2)
                    else:
                        # Draw low-confidence detections in red
                        draw_detection(frame, (x_norm, y_norm, w_norm, h_norm),
                                     confidence, color=(0, 0, 255), thickness=1)

            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Detections: {detection_count}"
            cv.putText(frame, info_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                      0.7, (0, 255, 0), 2)

            # Write frame to output video
            out.write(frame)

            if frame_count % 50 == 0:
                logger.info(f"  Processed {frame_count}/{total_frames} frames")

            torch.cuda.empty_cache()

    finally:
        cap.release()
        out.release()

    logger.info("")
    logger.info("=" * 80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Total detections (>= {conf_threshold}): {detection_count}")
    logger.info(f"Detection rate: {100*detection_count/max(1, frame_count):.1f}%")
    logger.info(f"Output video: {output_path}")
    logger.info("")
    logger.info("Green boxes = high confidence detections (>= threshold)")
    logger.info("Red boxes = low confidence detections (< threshold)")
    logger.info("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO detections on video with bounding boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python3 troubleshooting/visualize_detections.py \\
    --video video.mp4 \\
    --model model.pt \\
    --output output_with_boxes.mp4
        """
    )

    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--output", required=True, help="Output video path (.mp4)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID (default: 0)")

    args = parser.parse_args()

    # Validate inputs
    video_path = Path(args.video)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not video_path.exists():
        logger.error(f"❌ Video not found: {video_path}")
        sys.exit(1)

    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process
    success = visualize_video(
        str(video_path),
        str(model_path),
        str(output_path),
        conf_threshold=args.conf,
        device_id=args.device
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
