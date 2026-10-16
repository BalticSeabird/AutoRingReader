"""
STEP 1 WALKTHROUGH - Guided understanding of ring detection pipeline

This script walks through exactly what Step 1 does, step-by-step:

1. Load video and extract frames
2. Initialize YOLO model
3. Process frames through YOLO
4. Filter detections by confidence threshold
5. Extract and save crops
6. Save metadata JSON

Then shows you the actual data at each stage.

Usage:
  python troubleshooting/step1_walkthrough.py <video_path> <model_path>
"""

import sys
import logging
from pathlib import Path
import numpy as np
import cv2 as cv

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_utils import sample_frames_from_video, get_video_info
from ring_detector import RingDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def walkthrough(video_path: str, model_path: str, confidence: float = 0.7,
                sample_frames: int = 5):
    """Walk through Step 1 process step-by-step."""

    print_header("STEP 1 RING DETECTION - WALKTHROUGH")

    # ========== 1. VIDEO INFO ==========
    print_header("1. VIDEO INFORMATION")

    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        return

    video_info = get_video_info(video_path)
    logger.info(f"Video: {video_path}")
    logger.info(f"Resolution: {video_info['width']}x{video_info['height']}")
    logger.info(f"FPS: {video_info['fps']:.2f}")
    logger.info(f"Total frames: {video_info['frame_count']}")
    logger.info(f"Duration: {video_info['duration_seconds']:.2f} seconds")

    # ========== 2. FRAME EXTRACTION ==========
    print_header("2. FRAME EXTRACTION")

    logger.info("Extracting first 5 frames from video...")
    frame_samples = []
    for i, (frame_idx, frame) in enumerate(sample_frames_from_video(video_path, max_frames=sample_frames)):
        frame_samples.append((frame_idx, frame))
        logger.info(f"  Frame {i+1}: index={frame_idx}, shape={frame.shape}, dtype={frame.dtype}")

    # ========== 3. YOLO INITIALIZATION ==========
    print_header("3. YOLO MODEL INITIALIZATION")

    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return

    logger.info(f"Loading YOLO model: {model_path}")
    logger.info(f"Model file size: {Path(model_path).stat().st_size / 1e6:.1f} MB")

    try:
        detector = RingDetector(
            model_path=model_path,
            conf=confidence,
            iou=0.7,
            max_det=1,
            device_id=0,
        )
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # ========== 4. YOLO INFERENCE ==========
    print_header("4. YOLO INFERENCE ON FRAMES")

    logger.info(f"Processing {len(frame_samples)} frames through YOLO...")
    logger.info(f"Confidence threshold: {confidence}")

    all_results = []

    for frame_idx, (idx, frame) in enumerate(frame_samples):
        logger.info(f"\n--- Frame {frame_idx + 1}: index={idx} ---")

        # Convert BGR to RGB (YOLO expects RGB)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        logger.info(f"  Converted to RGB")

        # Run YOLO inference
        results = detector.model.predict(frame_rgb, verbose=False)
        logger.info(f"  YOLO inference complete")

        if results and len(results) > 0:
            result = results[0]
            has_boxes = hasattr(result, 'boxes') and len(result.boxes) > 0

            logger.info(f"  Result has boxes: {has_boxes}")

            if has_boxes:
                box = result.boxes[0]
                confidence_val = float(box.conf[0])
                x_norm, y_norm, w_norm, h_norm = box.xywhn[0]

                logger.info(f"  ✓ Detection found!")
                logger.info(f"    - Confidence: {confidence_val:.3f}")
                logger.info(f"    - Position (norm): ({x_norm:.3f}, {y_norm:.3f})")
                logger.info(f"    - Size (norm): ({w_norm:.3f}x{h_norm:.3f})")

                # Convert to pixel coords for reference
                width, height = video_info['width'], video_info['height']
                x_pix = x_norm * width
                y_pix = y_norm * height
                w_pix = w_norm * width
                h_pix = h_norm * height
                logger.info(f"    - Position (pixels): ({x_pix:.1f}, {y_pix:.1f})")
                logger.info(f"    - Size (pixels): ({w_pix:.1f}x{h_pix:.1f})")

                # Check against threshold
                passes_threshold = confidence_val >= confidence
                logger.info(f"    - Passes threshold ({confidence}): {passes_threshold}")

                # This would generate a crop
                if passes_threshold:
                    logger.info(f"    → This detection WILL BE SAVED as a crop")
                else:
                    logger.info(f"    → This detection WILL BE FILTERED OUT (below threshold)")

                all_results.append({
                    'frame_idx': idx,
                    'confidence': confidence_val,
                    'x_norm': float(x_norm),
                    'y_norm': float(y_norm),
                    'w_norm': float(w_norm),
                    'h_norm': float(h_norm),
                    'passes_threshold': passes_threshold,
                })
            else:
                logger.info(f"  ✗ No detection in this frame")

    # ========== 5. SUMMARY ==========
    print_header("5. SUMMARY")

    valid_detections = [r for r in all_results if r['passes_threshold']]

    logger.info(f"Frames processed: {len(frame_samples)}")
    logger.info(f"Total detections found: {len(all_results)}")
    logger.info(f"Detections above threshold: {len(valid_detections)}")

    if valid_detections:
        logger.info("\nDetections that WILL BE SAVED:")
        for i, det in enumerate(valid_detections, 1):
            logger.info(f"  {i}. Frame {det['frame_idx']}: conf={det['confidence']:.3f}")

    # ========== 6. WHAT'S SAVED ==========
    print_header("6. WHAT GETS SAVED")

    logger.info("For each valid detection, Step 1 saves:")
    logger.info("  1. A crop image file: {frame_idx:06d}_ring.png (224x224 pixels)")
    logger.info("  2. Detection info in _metadata.json:")
    logger.info("     - frame_idx")
    logger.info("     - detected (boolean)")
    logger.info("     - bbox (x_norm, y_norm, w_norm, h_norm)")
    logger.info("     - confidence")

    logger.info("\nExample metadata entry:")
    if valid_detections:
        example = valid_detections[0]
        logger.info(f"""
  {{
    "frame_idx": {example['frame_idx']},
    "detected": true,
    "bbox": {{
      "x_norm": {example['x_norm']:.3f},
      "y_norm": {example['y_norm']:.3f},
      "w_norm": {example['w_norm']:.3f},
      "h_norm": {example['h_norm']:.3f}
    }},
    "confidence": {example['confidence']:.3f}
  }}
        """)

    # ========== 7. TROUBLESHOOTING ==========
    print_header("7. IF YOU'RE NOT GETTING RESULTS")

    logger.info("Common issues and how to debug them:")
    logger.info("")
    logger.info("Issue: No detections found at all")
    logger.info("  → YOLO is not seeing rings in the video")
    logger.info("  → Check: Is the model trained on this type of video?")
    logger.info("  → Check: Is the ring visible in the video?")
    logger.info("")
    logger.info("Issue: Detections found but all filtered by confidence threshold")
    logger.info(f"  → Model is detecting, but confidences are low (below {confidence})")
    logger.info(f"  → Try lowering confidence threshold (e.g., 0.5 or 0.4)")
    logger.info("")
    logger.info("Issue: Detections found AND above threshold, but no output files")
    logger.info("  → Crops are not being saved to disk")
    logger.info("  → Check: Output directory permissions")
    logger.info("  → Check: Disk space available")
    logger.info("")

    print_header("END OF WALKTHROUGH")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 1 Walkthrough")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("model_path", help="Path to YOLO model")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--sample-frames", type=int, default=5, help="Number of frames to process")

    args = parser.parse_args()

    walkthrough(
        video_path=args.video_path,
        model_path=args.model_path,
        confidence=args.confidence,
        sample_frames=args.sample_frames
    )


if __name__ == "__main__":
    main()
