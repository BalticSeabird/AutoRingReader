"""
TEST ROI IMPACT - Check if ROI masking is the culprit

For TRI5_SCALE, the ROI is quite small: x=[399-614], y=[259-468]
This is only about 215x209 pixels out of a 1920x1080 frame.

This script tests:
1. Does YOLO find rings WITHOUT ROI masking?
2. Does YOLO find rings WITH ROI masking?
3. Are coordinates being transformed correctly?

Usage:
  python troubleshooting/test_roi_impact.py <video_path> <model_path>

Example:
  python troubleshooting/test_roi_impact.py \\
    path/to/TRI5_SCALE_2025-06-09_10_04.mp4 \\
    path/to/ring_detector_model.pt
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2 as cv

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_utils import sample_frames_from_video, get_video_info
from ring_detector import RingDetector
from pipeline_config import ROIConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def crop_to_roi(frame: np.ndarray, roi_bbox: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Crop frame to ROI."""
    xmin, ymin, xmax, ymax = [int(x) for x in roi_bbox]
    height, width = frame.shape[:2]

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)

    cropped_frame = frame[ymin:ymax, xmin:xmax]
    return cropped_frame, np.array([xmin, ymin])


def offset_detection(result, offset: np.ndarray, crop_shape: Tuple[int, int],
                     full_frame_shape: Tuple[int, int]):
    """Transform detection from crop coordinates back to full frame."""
    if not hasattr(result, 'boxes') or len(result.boxes) == 0:
        return False, None

    xmin_offset, ymin_offset = offset
    crop_height, crop_width = crop_shape
    full_height, full_width = full_frame_shape

    box = result.boxes[0]
    x_norm_crop, y_norm_crop, w_norm, h_norm = box.xywhn[0]

    # Convert to pixels in crop, then offset
    x_pixel_crop = x_norm_crop * crop_width
    y_pixel_crop = y_norm_crop * crop_height

    x_pixel_full = x_pixel_crop + xmin_offset
    y_pixel_full = y_pixel_crop + ymin_offset

    # Convert back to normalized for full frame
    x_norm_full = x_pixel_full / full_width
    y_norm_full = y_pixel_full / full_height

    confidence = float(box.conf[0])

    return True, {
        'confidence': confidence,
        'x_norm_crop': float(x_norm_crop),
        'y_norm_crop': float(y_norm_crop),
        'x_norm_full': float(x_norm_full),
        'y_norm_full': float(y_norm_full),
        'w_norm': float(w_norm),
        'h_norm': float(h_norm),
    }


def test_roi_impact(video_path: str, model_path: str, sample_frames: int = 20):
    """Test how ROI impacts detection."""

    print("\n" + "=" * 80)
    print("  ROI IMPACT TEST - TRI5_SCALE")
    print("=" * 80 + "\n")

    # Load video info
    video_info = get_video_info(video_path)
    logger.info(f"Video resolution: {video_info['width']}x{video_info['height']}")

    # Load ROI config
    roi_config = ROIConfig("config/rois_config.yaml")
    roi_bbox = roi_config.get_roi_bbox("TRI5_SCALE", year=2025)

    if roi_bbox:
        logger.info(f"ROI bounding box: {roi_bbox}")
        xmin, ymin, xmax, ymax = roi_bbox
        roi_width = xmax - xmin
        roi_height = ymax - ymin
        roi_area_pct = (roi_width * roi_height) / (video_info['width'] * video_info['height']) * 100
        logger.info(f"ROI size: {roi_width:.0f}x{roi_height:.0f} pixels ({roi_area_pct:.1f}% of frame)")
    else:
        logger.warning("No ROI configured!")
        roi_bbox = None

    # Initialize detector
    detector = RingDetector(
        model_path=model_path,
        conf=0.7,
        device_id=0,
    )
    logger.info("✓ Detector initialized\n")

    # TEST 1: WITHOUT ROI
    print("=" * 80)
    print("TEST 1: YOLO WITHOUT ROI MASKING")
    print("=" * 80 + "\n")

    results_no_roi = []
    frame_count = 0
    detection_count = 0

    for frame_idx, frame in sample_frames_from_video(video_path, max_frames=sample_frames):
        frame_count += 1
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = detector.model.predict(frame_rgb, verbose=False)[0]

        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            box = result.boxes[0]
            confidence = float(box.conf[0])
            x_norm, y_norm, w_norm, h_norm = box.xywhn[0]

            detection_count += 1
            results_no_roi.append({
                'frame_idx': frame_idx,
                'confidence': confidence,
                'x_norm': float(x_norm),
                'y_norm': float(y_norm),
                'w_norm': float(w_norm),
                'h_norm': float(h_norm),
            })

            logger.info(f"Frame {frame_idx}: detection (conf={confidence:.3f}, pos=({x_norm:.3f},{y_norm:.3f}))")

    logger.info(f"\nTotal: {detection_count} detections in {frame_count} frames")

    # TEST 2: WITH ROI
    print("\n" + "=" * 80)
    print("TEST 2: YOLO WITH ROI CROPPING")
    print("=" * 80 + "\n")

    results_with_roi = []
    frame_count = 0
    detection_count = 0
    roi_filtered_detections = 0

    for frame_idx, frame in sample_frames_from_video(video_path, max_frames=sample_frames):
        frame_count += 1

        # Crop to ROI
        cropped_frame, offset = crop_to_roi(frame, roi_bbox)
        crop_shape = cropped_frame.shape[:2]

        frame_rgb = cv.cvtColor(cropped_frame, cv.COLOR_BGR2RGB)
        result = detector.model.predict(frame_rgb, verbose=False)[0]

        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            has_det, det_info = offset_detection(
                result, offset, crop_shape,
                (video_info['height'], video_info['width'])
            )

            if has_det:
                detection_count += 1
                results_with_roi.append({
                    'frame_idx': frame_idx,
                    **det_info
                })

                # Check if detection is within original ROI area (sanity check)
                in_roi_x = roi_bbox[0] <= det_info['x_norm_full'] * video_info['width'] <= roi_bbox[2]
                in_roi_y = roi_bbox[1] <= det_info['y_norm_full'] * video_info['height'] <= roi_bbox[3]
                in_roi = in_roi_x and in_roi_y

                status = "✓ IN ROI" if in_roi else "⚠️  OUTSIDE ROI (transform error?)"

                logger.info(f"Frame {frame_idx}: detection (conf={det_info['confidence']:.3f})")
                logger.info(f"  Crop coords: ({det_info['x_norm_crop']:.3f},{det_info['y_norm_crop']:.3f})")
                logger.info(f"  Full coords: ({det_info['x_norm_full']:.3f},{det_info['y_norm_full']:.3f}) {status}")

    logger.info(f"\nTotal: {detection_count} detections in {frame_count} frames")

    # COMPARE
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80 + "\n")

    logger.info(f"Without ROI: {len(results_no_roi)} detections")
    logger.info(f"With ROI:    {len(results_with_roi)} detections")

    if len(results_no_roi) > 0 and len(results_with_roi) == 0:
        logger.warning("\n⚠️  PROBLEM IDENTIFIED!")
        logger.warning("ROI CROPPING IS LIKELY THE ISSUE")
        logger.warning("")
        logger.warning("Detections were found in full frames but not in ROI crops.")
        logger.warning("This suggests:")
        logger.warning("1. Rings are outside the defined ROI region")
        logger.warning("2. ROI crop is too aggressive and cutting off rings")
        logger.warning("3. ROI coordinates are incorrect for this camera")
        logger.warning("")
        logger.warning("FIX: Adjust ROI in config/rois_config.yaml for TRI5_SCALE")
    elif len(results_with_roi) > 0:
        logger.info("\n✓ ROI processing appears to be working")

        # Check coordinate transformations
        misaligned = 0
        for det in results_with_roi:
            in_roi_x = roi_bbox[0] <= det['x_norm_full'] * video_info['width'] <= roi_bbox[2]
            in_roi_y = roi_bbox[1] <= det['y_norm_full'] * video_info['height'] <= roi_bbox[3]
            if not (in_roi_x and in_roi_y):
                misaligned += 1

        if misaligned > 0:
            logger.warning(f"\n⚠️  Coordinate transformation issue detected!")
            logger.warning(f"{misaligned}/{len(results_with_roi)} detections are outside ROI after transformation")
            logger.warning("This suggests the offset calculation is incorrect")
    else:
        logger.warning("\n⚠️ No detections found in either test mode")
        logger.warning("This suggests the model is not seeing rings in this video at all")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test ROI impact on detections")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("model_path", help="Path to YOLO model")
    parser.add_argument("--samples", type=int, default=20, help="Number of frames to test")

    args = parser.parse_args()

    if not Path(args.video_path).exists():
        logger.error(f"Video not found: {args.video_path}")
        sys.exit(1)

    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    test_roi_impact(args.video_path, args.model_path, sample_frames=args.samples)


if __name__ == "__main__":
    main()
