"""
Detailed performance profiling script for Step 1 - Ring Detection.

Traces execution through both passes with detailed timing, memory, and GPU metrics.
Helps identify bottlenecks in YOLO inference vs crop generation.

Usage:
    python debug_step1_performance.py <video_path> <model_path> [--batch-size 16] [--roi-config rois_config.yaml]
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from time import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2 as cv
import torch
import psutil
import tracemalloc

from pipeline_config import PipelineConfig, ROIConfig
from pipeline_utils import get_video_info, get_video_output_dir, sample_frames_from_video
from ring_detector import RingDetector
from step1_ring_detection import (
    _process_frame_batch, _filter_detections_by_roi_batch,
    crop_ring_from_frame, _save_crop_image, _extract_video_year
)
from concurrent.futures import ThreadPoolExecutor


def setup_logging(video_stem: str) -> logging.Logger:
    """Setup detailed logging for profiling."""
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = log_dir / f"{timestamp}_step1_profile_{video_stem}.txt"

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Performance profiling started: {video_stem}")
    logger.info(f"Log file: {log_file}")
    return logger


def get_gpu_memory() -> Dict[str, float]:
    """Get current GPU memory usage in MB."""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            return {'allocated_mb': allocated, 'reserved_mb': reserved}
    except:
        pass
    return {'allocated_mb': 0.0, 'reserved_mb': 0.0}


def get_process_memory() -> Dict[str, float]:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,
        'vms_mb': mem_info.vms / 1024 / 1024,
    }


def profile_yolo_inference_pass(video_path: str, model_path: str, conf_threshold: float,
                                frame_sample_rate: int, batch_size: int, device_id: int,
                                logger: logging.Logger) -> Tuple[list, Dict[str, Any]]:
    """
    Profile YOLO inference pass with detailed timing and memory tracking.

    Returns:
        Tuple of (detections, profile_stats)
    """
    logger.info("=" * 80)
    logger.info("PASS 1: YOLO INFERENCE PROFILING")
    logger.info("=" * 80)

    video_stem = Path(video_path).stem
    pass1_start = time()
    detections = []
    profile_stats = {
        'batch_times': [],
        'frame_load_times': [],
        'yolo_inference_times': [],
        'detection_extraction_times': [],
        'gpu_memory_peak': 0.0,
        'gpu_memory_reserved_peak': 0.0,
        'process_memory_peak': 0.0,
        'frames_processed': 0,
        'batches_processed': 0,
        'total_time': 0.0,
    }

    try:
        # Get video info
        video_info = get_video_info(video_path)
        frame_height = video_info.get('height', 1080)
        frame_width = video_info.get('width', 1920)
        total_frames = video_info.get('frame_count', 0)

        logger.info(f"Video: {video_stem}")
        logger.info(f"  Dimensions: {frame_width}x{frame_height}")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Batch size: {batch_size}, Sample rate: {frame_sample_rate}")

        # Initialize detector
        logger.info(f"\nInitializing RingDetector on GPU {device_id}...")
        detector_start = time()
        detector = RingDetector(
            model_path,
            conf=conf_threshold,
            iou=0.7,
            max_det=1,
            device_id=device_id,
        )
        detector_time = time() - detector_start
        logger.info(f"  Detector initialized in {detector_time:.2f}s")

        # Process frames in batches
        batch_frames = []
        batch_indices = []
        batch_num = 0

        logger.info(f"\nProcessing frames...")
        for frame_idx, frame in sample_frames_from_video(video_path, sample_rate=frame_sample_rate):
            batch_frames.append(frame)
            batch_indices.append(frame_idx)

            # Process batch when full
            if len(batch_frames) >= batch_size:
                batch_num += 1
                batch_start = time()

                # Time frame loading (already loaded above, but log for reference)
                frame_load_time = 0.0

                # Time YOLO inference
                yolo_start = time()
                batch_results = _process_frame_batch(batch_frames, batch_indices, detector, conf_threshold)
                yolo_time = time() - yolo_start

                # Time detection extraction
                extract_start = time()
                for frame_idx_batch, result, frame in batch_results:
                    detected = hasattr(result, 'boxes') and len(result.boxes) > 0

                    bbox = None
                    confidence = 0.0
                    if detected:
                        box = result.boxes[0]
                        x_norm, y_norm, w_norm, h_norm = box.xywhn[0]
                        confidence = float(box.conf[0])
                        bbox = {
                            'x_norm': float(x_norm),
                            'y_norm': float(y_norm),
                            'w_norm': float(w_norm),
                            'h_norm': float(h_norm),
                        }

                    detections.append({
                        'frame_idx': frame_idx_batch,
                        'detected': detected,
                        'bbox': bbox,
                        'confidence': confidence,
                    })

                extract_time = time() - extract_start

                batch_time = time() - batch_start
                profile_stats['batch_times'].append(batch_time)
                profile_stats['yolo_inference_times'].append(yolo_time)
                profile_stats['detection_extraction_times'].append(extract_time)
                profile_stats['frames_processed'] += len(batch_frames)
                profile_stats['batches_processed'] += 1

                # Log GPU memory
                gpu_mem = get_gpu_memory()
                profile_stats['gpu_memory_peak'] = max(profile_stats['gpu_memory_peak'], gpu_mem['allocated_mb'])
                profile_stats['gpu_memory_reserved_peak'] = max(profile_stats['gpu_memory_reserved_peak'], gpu_mem['reserved_mb'])

                # Log process memory
                proc_mem = get_process_memory()
                profile_stats['process_memory_peak'] = max(profile_stats['process_memory_peak'], proc_mem['rss_mb'])

                if batch_num % 5 == 0 or batch_num == 1:
                    logger.debug(f"Batch {batch_num}: {len(batch_frames)} frames in {batch_time:.3f}s "
                               f"(YOLO {yolo_time:.3f}s, extract {extract_time:.3f}s) "
                               f"| GPU: {gpu_mem['allocated_mb']:.0f}MB allocated, "
                               f"{gpu_mem['reserved_mb']:.0f}MB reserved | "
                               f"Process: {proc_mem['rss_mb']:.0f}MB RSS")

                batch_frames.clear()
                batch_indices.clear()
                torch.cuda.empty_cache()

        # Process remaining frames
        if batch_frames:
            batch_num += 1
            batch_start = time()

            yolo_start = time()
            batch_results = _process_frame_batch(batch_frames, batch_indices, detector, conf_threshold)
            yolo_time = time() - yolo_start

            extract_start = time()
            for frame_idx_batch, result, frame in batch_results:
                detected = hasattr(result, 'boxes') and len(result.boxes) > 0

                bbox = None
                confidence = 0.0
                if detected:
                    box = result.boxes[0]
                    x_norm, y_norm, w_norm, h_norm = box.xywhn[0]
                    confidence = float(box.conf[0])
                    bbox = {
                        'x_norm': float(x_norm),
                        'y_norm': float(y_norm),
                        'w_norm': float(w_norm),
                        'h_norm': float(h_norm),
                    }

                detections.append({
                    'frame_idx': frame_idx_batch,
                    'detected': detected,
                    'bbox': bbox,
                    'confidence': confidence,
                })

            extract_time = time() - extract_start

            batch_time = time() - batch_start
            profile_stats['batch_times'].append(batch_time)
            profile_stats['yolo_inference_times'].append(yolo_time)
            profile_stats['detection_extraction_times'].append(extract_time)
            profile_stats['frames_processed'] += len(batch_frames)
            profile_stats['batches_processed'] += 1

            logger.debug(f"Final batch {batch_num}: {len(batch_frames)} frames in {batch_time:.3f}s")

            torch.cuda.empty_cache()

        pass1_elapsed = time() - pass1_start
        profile_stats['total_time'] = pass1_elapsed

        # Summary statistics
        logger.info(f"\nPASS 1 SUMMARY:")
        logger.info(f"  Total time: {pass1_elapsed:.2f}s")
        logger.info(f"  Frames processed: {profile_stats['frames_processed']}")
        logger.info(f"  Batches processed: {profile_stats['batches_processed']}")
        logger.info(f"  Avg time per batch: {np.mean(profile_stats['batch_times']):.3f}s")
        logger.info(f"  Avg YOLO time per batch: {np.mean(profile_stats['yolo_inference_times']):.3f}s")
        logger.info(f"  Avg detection extraction per batch: {np.mean(profile_stats['detection_extraction_times']):.3f}s")
        logger.info(f"  GPU memory peak: {profile_stats['gpu_memory_peak']:.0f}MB allocated, {profile_stats['gpu_memory_reserved_peak']:.0f}MB reserved")
        logger.info(f"  Process memory peak: {profile_stats['process_memory_peak']:.0f}MB")

        return detections, profile_stats

    except Exception as e:
        logger.error(f"Pass 1 profiling failed: {e}")
        logger.error(traceback.format_exc())
        raise


def profile_crop_generation_pass(video_path: str, output_dir: Path, camera: str,
                                detections: list, frame_dims: Dict[str, int],
                                roi_config: Optional[ROIConfig] = None,
                                conf_threshold: float = 0.5,
                                video_year: Optional[int] = None,
                                logger: logging.Logger = None) -> Tuple[int, int, Dict[str, Any]]:
    """
    Profile crop generation pass with detailed timing and memory tracking.

    Returns:
        Tuple of (crop_count, filtered_count, profile_stats)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("PASS 2: CROP GENERATION PROFILING")
    logger.info("=" * 80)

    pass2_start = time()
    crop_count = 0
    filtered_count = 0
    profile_stats = {
        'frame_iteration_time': 0.0,
        'roi_filtering_times': [],
        'crop_extract_times': [],
        'crop_save_times': [],
        'frames_with_detections': 0,
        'frames_skipped_conf': 0,
        'frames_filtered_roi': 0,
        'crop_save_total_time': 0.0,
        'total_time': 0.0,
    }

    try:
        logger.info(f"Starting crop generation for {len(detections)} detections")

        # Get ROI bbox if available
        roi_bbox = None
        if roi_config:
            roi_bbox = roi_config.get_roi_bbox(camera, year=video_year)
            if roi_bbox:
                logger.info(f"Using ROI bbox: {roi_bbox}")

        # Create detection lookup
        detection_by_frame = {d['frame_idx']: d for d in detections}
        logger.info(f"Detection lookup created for {len(detection_by_frame)} frames")

        # Thread pool for async crop saving
        crop_save_executor = ThreadPoolExecutor(max_workers=6)
        pending_saves = []
        crop_timings = []

        # Iterate through video frames
        frame_iter_start = time()
        frame_count = 0

        for frame_idx, frame in sample_frames_from_video(video_path, sample_rate=1):
            frame_count += 1

            # Check if this frame has a detection
            detection = detection_by_frame.get(frame_idx)
            if detection is None:
                continue

            profile_stats['frames_with_detections'] += 1

            confidence = detection['confidence']
            bbox = detection['bbox']

            # Filter by confidence
            filter_start = time()
            if confidence < conf_threshold:
                profile_stats['frames_skipped_conf'] += 1
                continue

            # Check ROI if configured
            if roi_bbox:
                if bbox is None:
                    continue

                x_center_px = bbox['x_norm'] * frame_dims['width']
                y_center_px = bbox['y_norm'] * frame_dims['height']
                roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi_bbox

                if not (roi_xmin <= x_center_px <= roi_xmax and
                        roi_ymin <= y_center_px <= roi_ymax):
                    profile_stats['frames_filtered_roi'] += 1
                    filtered_count += 1
                    continue
            elif bbox is None:
                continue

            filter_time = time() - filter_start
            profile_stats['roi_filtering_times'].append(filter_time)

            # Generate crop
            if bbox is not None:
                try:
                    crop_start = time()
                    crop = crop_ring_from_frame(frame, bbox, crop_size=224)
                    crop_time = time() - crop_start
                    profile_stats['crop_extract_times'].append(crop_time)

                    if crop is not None:
                        crop_path = output_dir / f"{frame_idx:06d}_ring.png"
                        save_start = time()
                        _save_crop_image(crop, crop_path)
                        save_time = time() - save_start
                        profile_stats['crop_save_times'].append(save_time)
                        crop_timings.append({'extract': crop_time, 'save': save_time})
                        crop_count += 1

                except Exception as e:
                    logger.debug(f"Failed to generate crop for frame {frame_idx}: {e}")

            if frame_count % 100 == 0:
                logger.debug(f"Processed {frame_count} frames, {len(pending_saves)} crops queued...")

        frame_iter_elapsed = time() - frame_iter_start
        profile_stats['frame_iteration_time'] = frame_iter_elapsed

        pass2_elapsed = time() - pass2_start
        profile_stats['total_time'] = pass2_elapsed

        # Summary statistics
        logger.info(f"\nPASS 2 SUMMARY:")
        logger.info(f"  Total time: {pass2_elapsed:.2f}s")
        logger.info(f"  Frames iterated: {frame_count}")
        logger.info(f"  Frames with detections: {profile_stats['frames_with_detections']}")
        logger.info(f"  Frames skipped (low confidence): {profile_stats['frames_skipped_conf']}")
        logger.info(f"  Frames filtered by ROI: {profile_stats['frames_filtered_roi']}")
        logger.info(f"  Crops generated: {crop_count}")

        if profile_stats['crop_extract_times']:
            logger.info(f"  Avg crop extraction time: {np.mean(profile_stats['crop_extract_times']):.4f}s")
        if profile_stats['crop_save_times']:
            logger.info(f"  Avg crop save time: {np.mean(profile_stats['crop_save_times']):.4f}s")
        if profile_stats['roi_filtering_times']:
            logger.info(f"  Avg ROI filtering time: {np.mean(profile_stats['roi_filtering_times']):.6f}s")

        return crop_count, filtered_count, profile_stats

    except Exception as e:
        logger.error(f"Pass 2 profiling failed: {e}")
        logger.error(traceback.format_exc())
        raise


def main(video_path: str, model_path: str, batch_size: int = 16,
         roi_config_path: Optional[str] = None, conf_threshold: float = 0.5,
         device_id: int = 0):
    """Main profiling function."""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    video_stem = video_path.stem
    logger = setup_logging(video_stem)

    try:
        # Load ROI config if provided
        roi_config = None
        if roi_config_path and Path(roi_config_path).exists():
            roi_config = ROIConfig(roi_config_path)
            logger.info(f"Loaded ROI config: {roi_config_path}")

        video_year = _extract_video_year(str(video_path), video_stem)
        frame_sample_rate = 1  # Process all frames for profiling

        # Create output directory
        output_dir = Path(f"debug_output_{video_stem}")
        output_dir.mkdir(exist_ok=True)

        # PASS 1: YOLO Inference
        detections, pass1_stats = profile_yolo_inference_pass(
            str(video_path), model_path, conf_threshold, frame_sample_rate,
            batch_size, device_id, logger
        )

        # PASS 2: Crop Generation
        crop_count, filtered_count, pass2_stats = profile_crop_generation_pass(
            str(video_path), output_dir, video_stem.split('_')[0],
            detections, {'height': 1080, 'width': 1920},
            roi_config, conf_threshold, video_year, logger
        )

        # Overall summary
        overall_time = pass1_stats['total_time'] + pass2_stats['total_time']
        logger.info("=" * 80)
        logger.info("OVERALL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Pass 1 (YOLO): {pass1_stats['total_time']:.2f}s ({pass1_stats['total_time']/overall_time*100:.1f}%)")
        logger.info(f"Pass 2 (Crops): {pass2_stats['total_time']:.2f}s ({pass2_stats['total_time']/overall_time*100:.1f}%)")
        logger.info(f"Total: {overall_time:.2f}s")

        # Save detailed results to JSON
        results = {
            'video': str(video_path),
            'pass1': pass1_stats,
            'pass2': pass2_stats,
            'overall_time': overall_time,
            'crop_count': crop_count,
            'filtered_count': filtered_count,
        }

        results_file = output_dir / "profile_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nDetailed results saved to: {results_file}")

        return 0

    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1 Performance Profiling")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--model-path", default="models/ring_detector.pt", help="Path to YOLO model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for YOLO")
    parser.add_argument("--roi-config", help="Path to ROI config YAML")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID")

    args = parser.parse_args()
    exit_code = main(
        args.video_path,
        args.model_path,
        batch_size=args.batch_size,
        roi_config_path=args.roi_config,
        conf_threshold=args.conf_threshold,
        device_id=args.device_id
    )
    sys.exit(exit_code)
