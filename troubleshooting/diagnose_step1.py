"""
STEP 1 DIAGNOSTIC SCRIPT - Trace ring detection through entire pipeline

Purpose: Identify exactly what's happening at each stage of Step 1:
1. Frame extraction from video
2. YOLO inference and detections
3. Filtering by confidence threshold
4. Cropping and JSON output
5. Compare with visual inspection

Usage:
  python troubleshooting/diagnose_step1.py <video_path> <model_path> [--confidence 0.7] [--max-frames 100]

Example:
  python troubleshooting/diagnose_step1.py \
    path/to/TRI5_SCALE_2025-06-09_10_04.mp4 \
    path/to/ring_detector_model.pt \
    --confidence 0.7 \
    --max-frames 100
"""

import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import numpy as np
import cv2 as cv
import torch
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_utils import sample_frames_from_video, get_video_info
from ring_detector import RingDetector

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class Step1Diagnostic:
    """Diagnostic tool for Step 1 ring detection."""

    def __init__(self, video_path: str, model_path: str, confidence: float = 0.7,
                 device_id: int = 0):
        self.video_path = video_path
        self.model_path = model_path
        self.confidence = confidence
        self.device_id = device_id

        self.video_stem = Path(video_path).stem
        self.detector = None

        # Diagnostics output
        self.output_dir = Path("troubleshooting/step1_diagnostic_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'total_frames_in_video': 0,
            'frames_extracted': 0,
            'frames_processed': 0,
            'detections_found': 0,
            'detections_above_confidence': 0,
            'confidence_values': [],
            'detection_examples': [],
        }

    def log_section(self, title: str):
        """Log a section header."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"  {title}")
        logger.info("=" * 80)

    def step_1_check_video(self):
        """Step 1: Check video file and get metadata."""
        self.log_section("STEP 1: VIDEO METADATA")

        if not Path(self.video_path).exists():
            logger.error(f"Video file not found: {self.video_path}")
            return False

        logger.info(f"Video file: {self.video_path}")
        logger.info(f"Video exists: ✓")
        logger.info(f"File size: {Path(self.video_path).stat().st_size / 1e9:.2f} GB")

        # Get video metadata
        video_info = get_video_info(self.video_path)
        if not video_info:
            logger.error("Failed to open video file")
            return False

        logger.info(f"Resolution: {video_info['width']}x{video_info['height']}")
        logger.info(f"FPS: {video_info['fps']:.2f}")
        logger.info(f"Total frames: {video_info['frame_count']}")
        logger.info(f"Duration: {video_info['duration_seconds']:.2f} seconds")

        self.stats['total_frames_in_video'] = video_info['frame_count']
        self.video_info = video_info

        return True

    def step_2_check_model(self):
        """Step 2: Check model file and initialize detector."""
        self.log_section("STEP 2: MODEL INITIALIZATION")

        if not Path(self.model_path).exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False

        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Model exists: ✓")
        logger.info(f"File size: {Path(self.model_path).stat().st_size / 1e6:.1f} MB")

        # Initialize detector
        try:
            logger.info(f"Initializing detector on GPU {self.device_id}...")
            self.detector = RingDetector(
                model_path=self.model_path,
                conf=self.confidence,
                iou=0.7,
                max_det=1,
                device_id=self.device_id,
            )
            logger.info(f"Detector initialized: ✓")
            logger.info(f"Model confidence threshold: {self.confidence}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            return False

    def step_3_extract_frames(self, max_frames: int = None):
        """Step 3: Extract frames from video and check they're being read."""
        self.log_section("STEP 3: FRAME EXTRACTION")

        logger.info("Extracting frames from video...")

        frame_samples = []
        for frame_idx, frame in sample_frames_from_video(self.video_path, max_frames=max_frames):
            frame_samples.append((frame_idx, frame))
            self.stats['frames_extracted'] += 1

            if self.stats['frames_extracted'] % 100 == 0:
                logger.info(f"  Extracted {self.stats['frames_extracted']} frames...")

        logger.info(f"Total frames extracted: {self.stats['frames_extracted']}")

        if frame_samples:
            first_idx, first_frame = frame_samples[0]
            logger.info(f"First frame index: {first_idx}")
            logger.info(f"First frame shape: {first_frame.shape}")
            logger.info(f"First frame dtype: {first_frame.dtype}")
            logger.info(f"First frame min/max: {first_frame.min()}/{first_frame.max()}")

            # Save first frame as reference
            first_frame_path = self.output_dir / "01_first_frame_raw.jpg"
            cv.imwrite(str(first_frame_path), first_frame)
            logger.info(f"Saved first frame to: {first_frame_path}")

        return frame_samples

    def step_4_run_yolo_inference(self, frame_samples: List[Tuple[int, np.ndarray]]):
        """Step 4: Run YOLO inference on extracted frames."""
        self.log_section("STEP 4: YOLO INFERENCE")

        if not self.detector:
            logger.error("Detector not initialized")
            return []

        logger.info(f"Running YOLO inference on {len(frame_samples)} frames...")

        all_detections = []

        for sample_idx, (frame_idx, frame) in enumerate(frame_samples):
            try:
                # Convert BGR to RGB
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # Run YOLO inference
                results = self.detector.model.predict(frame_rgb, verbose=False)

                # Process results
                if results and len(results) > 0:
                    result = results[0]

                    has_detection = hasattr(result, 'boxes') and len(result.boxes) > 0
                    self.stats['frames_processed'] += 1

                    if has_detection:
                        self.stats['detections_found'] += 1

                        box = result.boxes[0]
                        confidence = float(box.conf[0])
                        self.stats['confidence_values'].append(confidence)

                        x_norm, y_norm, w_norm, h_norm = box.xywhn[0]

                        detection = {
                            'frame_idx': frame_idx,
                            'sample_idx': sample_idx,
                            'confidence': confidence,
                            'x_norm': float(x_norm),
                            'y_norm': float(y_norm),
                            'w_norm': float(w_norm),
                            'h_norm': float(h_norm),
                            'above_threshold': confidence >= self.confidence,
                        }

                        all_detections.append(detection)

                        if confidence >= self.confidence:
                            self.stats['detections_above_confidence'] += 1

                        if len(self.stats['detection_examples']) < 5:
                            self.stats['detection_examples'].append(detection)
                            logger.info(f"  Frame {frame_idx}: detection conf={confidence:.3f}, "
                                      f"pos=({x_norm:.3f}, {y_norm:.3f}), "
                                      f"size=({w_norm:.3f}, {h_norm:.3f})")

                if (sample_idx + 1) % 50 == 0:
                    logger.info(f"  Processed {sample_idx + 1}/{len(frame_samples)} frames")

            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {e}")

        logger.info(f"Total frames processed: {self.stats['frames_processed']}")
        logger.info(f"Total detections found: {self.stats['detections_found']}")
        logger.info(f"Above confidence threshold ({self.confidence}): {self.stats['detections_above_confidence']}")

        if self.stats['confidence_values']:
            logger.info(f"Confidence stats:")
            logger.info(f"  Min: {min(self.stats['confidence_values']):.3f}")
            logger.info(f"  Max: {max(self.stats['confidence_values']):.3f}")
            logger.info(f"  Mean: {np.mean(self.stats['confidence_values']):.3f}")
            logger.info(f"  Median: {np.median(self.stats['confidence_values']):.3f}")

        return all_detections

    def step_5_visualize_detections(self, frame_samples: List[Tuple[int, np.ndarray]],
                                    detections: List[Dict]):
        """Step 5: Visualize detections on frames."""
        self.log_section("STEP 5: VISUALIZATION")

        logger.info("Creating visualizations of detections...")

        # Create a visualization of first N frames with and without detected boxes
        vis_count = 0
        for frame_idx, frame in frame_samples[:20]:  # First 20 frames
            frame_vis = frame.copy()
            height, width = frame.shape[:2]

            # Find detections for this frame
            frame_detections = [d for d in detections if d['frame_idx'] == frame_idx]

            # Draw all detections (mark by confidence)
            for det in frame_detections:
                x_norm, y_norm, w_norm, h_norm = det['x_norm'], det['y_norm'], det['w_norm'], det['h_norm']
                x_center, y_center = int(x_norm * width), int(y_norm * height)
                w_pix, h_pix = int(w_norm * width), int(h_norm * height)

                x1, y1 = x_center - w_pix // 2, y_center - h_pix // 2
                x2, y2 = x_center + w_pix // 2, y_center + h_pix // 2

                # Color based on confidence threshold
                if det['above_threshold']:
                    color = (0, 255, 0)  # Green - above threshold
                    label = f"VALID: {det['confidence']:.3f}"
                else:
                    color = (0, 0, 255)  # Red - below threshold
                    label = f"LOW: {det['confidence']:.3f}"

                cv.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame_vis, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX,
                          0.5, color, 2)

            # Save visualization
            vis_path = self.output_dir / f"02_frame_{frame_idx:06d}_detections.jpg"
            cv.imwrite(str(vis_path), frame_vis)
            vis_count += 1

        logger.info(f"Saved {vis_count} detection visualizations to {self.output_dir}")

    def step_6_save_diagnostics(self, detections: List[Dict]):
        """Step 6: Save comprehensive diagnostics report."""
        self.log_section("STEP 6: DIAGNOSTIC REPORT")

        # Create detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'video_path': self.video_path,
            'video_stem': self.video_stem,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence,
            'device_id': self.device_id,
            'statistics': self.stats,
            'all_detections': detections,
        }

        # Save as JSON
        report_path = self.output_dir / f"{self.video_stem}_diagnostic_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Diagnostic report saved to: {report_path}")

        # Save summary as text
        summary_path = self.output_dir / f"{self.video_stem}_diagnostic_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STEP 1 RING DETECTION DIAGNOSTIC SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Video: {self.video_path}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Confidence Threshold: {self.confidence}\n\n")

            f.write("STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total frames in video: {self.stats['total_frames_in_video']}\n")
            f.write(f"Frames extracted: {self.stats['frames_extracted']}\n")
            f.write(f"Frames processed by YOLO: {self.stats['frames_processed']}\n")
            f.write(f"Total detections found: {self.stats['detections_found']}\n")
            f.write(f"Detections above threshold: {self.stats['detections_above_confidence']}\n\n")

            if self.stats['confidence_values']:
                f.write("CONFIDENCE STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Min confidence: {min(self.stats['confidence_values']):.3f}\n")
                f.write(f"Max confidence: {max(self.stats['confidence_values']):.3f}\n")
                f.write(f"Mean confidence: {np.mean(self.stats['confidence_values']):.3f}\n")
                f.write(f"Median confidence: {np.median(self.stats['confidence_values']):.3f}\n\n")

            f.write("DETECTION EXAMPLES (First 5)\n")
            f.write("-" * 80 + "\n")
            for i, det in enumerate(self.stats['detection_examples'], 1):
                f.write(f"\nDetection #{i}\n")
                f.write(f"  Frame Index: {det['frame_idx']}\n")
                f.write(f"  Confidence: {det['confidence']:.3f}\n")
                f.write(f"  Position (norm): ({det['x_norm']:.3f}, {det['y_norm']:.3f})\n")
                f.write(f"  Size (norm): ({det['w_norm']:.3f}, {det['h_norm']:.3f})\n")
                f.write(f"  Above threshold: {det['above_threshold']}\n")

            f.write("\n\n" + "=" * 80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("=" * 80 + "\n")

            if self.stats['detections_above_confidence'] == 0:
                f.write("⚠️  NO DETECTIONS ABOVE THRESHOLD!\n\n")
                if self.stats['detections_found'] > 0:
                    f.write(f"However, {self.stats['detections_found']} detections were found below threshold.\n\n")
                    f.write("Possible causes:\n")
                    f.write("1. Confidence threshold is too high\n")
                    f.write("2. Model not trained on this type of video/ring\n")
                    f.write("3. ROI masking is filtering out all detections\n")
                    f.write("4. Video resolution/quality issues\n")
                else:
                    f.write("No detections found at all.\n\n")
                    f.write("Possible causes:\n")
                    f.write("1. Model not initialized correctly\n")
                    f.write("2. Frames not being extracted properly\n")
                    f.write("3. YOLO inference not running\n")
            else:
                f.write(f"✓ Found {self.stats['detections_above_confidence']} valid detections\n")

        logger.info(f"Summary saved to: {summary_path}")

    def run_diagnostics(self, max_frames: int = None):
        """Run complete diagnostic sequence."""
        self.log_section("STARTING STEP 1 DIAGNOSTIC")
        logger.info(f"Output directory: {self.output_dir}")

        if not self.step_1_check_video():
            return False

        if not self.step_2_check_model():
            return False

        frame_samples = self.step_3_extract_frames(max_frames=max_frames)
        if not frame_samples:
            logger.error("No frames extracted from video")
            return False

        detections = self.step_4_run_yolo_inference(frame_samples)

        self.step_5_visualize_detections(frame_samples, detections)

        self.step_6_save_diagnostics(detections)

        self.log_section("DIAGNOSTIC COMPLETE")
        logger.info(f"All output saved to: {self.output_dir}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Step 1 Ring Detection Diagnostic")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("model_path", help="Path to ring detector model")
    parser.add_argument("--confidence", type=float, default=0.7,
                       help="Confidence threshold for YOLO (default: 0.7)")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID (default: 0)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process (default: all)")

    args = parser.parse_args()

    diag = Step1Diagnostic(
        video_path=args.video_path,
        model_path=args.model_path,
        confidence=args.confidence,
        device_id=args.device
    )

    success = diag.run_diagnostics(max_frames=args.max_frames)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
