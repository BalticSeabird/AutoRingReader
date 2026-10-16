#!/usr/bin/env python3
"""
Simple diagnostic to test if YOLO model works for ring detection and verify crops.
Processes entire video and saves detected ring crops.

Usage:
  python3 troubleshooting/run_diagnostic.py --video <path> --model <path> --output <dir>
"""

import sys
import argparse
import logging
from pathlib import Path
import json
import numpy as np
import cv2 as cv
import torch
from typing import Optional, Dict, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ring_detector import RingDetector
from pipeline_utils import get_video_info, sample_frames_from_video

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def crop_ring_from_frame(frame: np.ndarray, bbox: Dict, crop_size: int = 224) -> Optional[np.ndarray]:
    """Crop ring region from frame based on normalized bbox."""
    try:
        height, width = frame.shape[:2]
        x_center = round(bbox['x_norm'] * width)
        y_center = round(bbox['y_norm'] * height)
        x_center = max(crop_size // 2, min(width - crop_size // 2, x_center))
        y_center = max(crop_size // 2, min(height - crop_size // 2, y_center))

        x1 = x_center - crop_size // 2
        y1 = y_center - crop_size // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        crop = frame[y1:y2, x1:x2]

        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            padded = np.zeros((crop_size, crop_size, 3), dtype=frame.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        return crop
    except Exception as e:
        logger.debug(f"Crop extraction failed: {e}")
        return None


def process_video(video_path: str, model_path: str, output_dir: Path,
                  conf_threshold: float = 0.5, batch_size: int = 16,
                  device_id: int = 0) -> Dict[str, Any]:
    """Process entire video with YOLO and save crops."""

    video_stem = Path(video_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Confidence threshold: {conf_threshold}")

    try:
        # Get video info
        video_info = get_video_info(video_path)
        total_frames = video_info.get('frame_count', 0)
        height = video_info.get('height', 1080)
        width = video_info.get('width', 1920)

        logger.info(f"Video: {total_frames} frames, {height}x{width}")

        # Load YOLO model
        logger.info("Loading YOLO model...")
        detector = RingDetector(
            model_path,
            conf=conf_threshold,
            iou=0.7,
            max_det=1,
            device_id=device_id,
        )

        # Process video
        frame_count = 0
        detected_count = 0
        crop_count = 0
        conf_values = []
        detections = []

        batch_frames = []
        batch_indices = []

        logger.info("Processing frames...")

        for frame_idx, frame in sample_frames_from_video(video_path, sample_rate=1):
            batch_frames.append(frame)
            batch_indices.append(frame_idx)

            # Process batch when it reaches batch_size or at end
            if len(batch_frames) >= batch_size:
                # Convert to RGB for YOLO
                frames_rgb = [cv.cvtColor(f, cv.COLOR_BGR2RGB) for f in batch_frames]

                # Run YOLO inference
                batch_results = detector.model.predict(frames_rgb, verbose=False)

                # Process results
                for batch_frame_idx, result, original_frame in zip(batch_indices, batch_results, batch_frames):
                    frame_count += 1

                    if hasattr(result, 'boxes') and len(result.boxes) > 0:
                        box = result.boxes[0]
                        x_norm, y_norm, w_norm, h_norm = box.xywhn[0]
                        confidence = float(box.conf[0])

                        bbox = {
                            'x_norm': float(x_norm),
                            'y_norm': float(y_norm),
                            'w_norm': float(w_norm),
                            'h_norm': float(h_norm),
                        }

                        if confidence >= conf_threshold:
                            detected_count += 1
                            conf_values.append(confidence)

                            # Save crop
                            crop = crop_ring_from_frame(original_frame, bbox, crop_size=224)
                            if crop is not None:
                                crop_path = output_dir / f"{batch_frame_idx:06d}_ring.png"
                                cv.imwrite(str(crop_path), crop)
                                crop_count += 1

                        detections.append({
                            'frame_idx': batch_frame_idx,
                            'detected': True,
                            'confidence': confidence,
                            'bbox': bbox,
                        })
                    else:
                        detections.append({
                            'frame_idx': batch_frame_idx,
                            'detected': False,
                            'confidence': 0.0,
                            'bbox': None,
                        })

                # Clear batch and GPU memory
                batch_frames.clear()
                batch_indices.clear()
                torch.cuda.empty_cache()

                if frame_count % 100 == 0:
                    logger.info(f"  Processed {frame_count} frames, {detected_count} detections so far")

        # Process remaining frames
        if batch_frames:
            frames_rgb = [cv.cvtColor(f, cv.COLOR_BGR2RGB) for f in batch_frames]
            batch_results = detector.model.predict(frames_rgb, verbose=False)

            for batch_frame_idx, result, original_frame in zip(batch_indices, batch_results, batch_frames):
                frame_count += 1

                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    box = result.boxes[0]
                    x_norm, y_norm, w_norm, h_norm = box.xywhn[0]
                    confidence = float(box.conf[0])

                    bbox = {
                        'x_norm': float(x_norm),
                        'y_norm': float(y_norm),
                        'w_norm': float(w_norm),
                        'h_norm': float(h_norm),
                    }

                    if confidence >= conf_threshold:
                        detected_count += 1
                        conf_values.append(confidence)

                        crop = crop_ring_from_frame(original_frame, bbox, crop_size=224)
                        if crop is not None:
                            crop_path = output_dir / f"{batch_frame_idx:06d}_ring.png"
                            cv.imwrite(str(crop_path), crop)
                            crop_count += 1

                    detections.append({
                        'frame_idx': batch_frame_idx,
                        'detected': True,
                        'confidence': confidence,
                        'bbox': bbox,
                    })
                else:
                    detections.append({
                        'frame_idx': batch_frame_idx,
                        'detected': False,
                        'confidence': 0.0,
                        'bbox': None,
                    })

            torch.cuda.empty_cache()

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Frames with detections: {detected_count}")
        logger.info(f"Detection rate: {100*detected_count/max(1, frame_count):.1f}%")
        logger.info(f"Crops saved: {crop_count}")

        if conf_values:
            logger.info(f"Confidence - Mean: {np.mean(conf_values):.3f}, Max: {np.max(conf_values):.3f}, Min: {np.min(conf_values):.3f}")
        else:
            logger.info("⚠️  NO DETECTIONS - Model may not be working or rings not in video")

        logger.info(f"Crops saved to: {output_dir}")
        logger.info("=" * 80)

        # Save detections JSON
        metadata = {
            'video_path': video_path,
            'frames_total': total_frames,
            'frames_processed': frame_count,
            'ring_detect_count': detected_count,
            'detection_rate': detected_count / max(1, frame_count),
            'crops_saved': crop_count,
            'confidence_threshold': conf_threshold,
            'conf_mean': float(np.mean(conf_values)) if conf_values else 0.0,
            'conf_max': float(np.max(conf_values)) if conf_values else 0.0,
            'conf_min': float(np.min(conf_values)) if conf_values else 0.0,
        }

        metadata_path = output_dir / "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to: {metadata_path}")

        return metadata

    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Simple diagnostic: test YOLO model and verify crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python3 troubleshooting/run_diagnostic.py \\
    --video video.mp4 \\
    --model model.pt \\
    --output crops_output
        """
    )

    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--output", default="diagnostic_crops", help="Output directory for crops")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID (default: 0)")

    args = parser.parse_args()

    # Validate inputs
    video_path = Path(args.video)
    model_path = Path(args.model)
    output_dir = Path(args.output)

    if not video_path.exists():
        logger.error(f"❌ Video not found: {video_path}")
        sys.exit(1)

    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Process
    result = process_video(
        str(video_path),
        str(model_path),
        output_dir,
        conf_threshold=args.conf,
        batch_size=args.batch_size,
        device_id=args.device
    )

    if 'error' in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
