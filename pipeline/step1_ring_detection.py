"""
Step 1: Ring Detection - Detect metal rings in videos using YOLO.

Simplified architecture: GPU workers process videos directly using YOLO.predict(source=video),
which is more efficient than frame extraction + batching.

Each GPU worker processes one video at a time -> logs results.
"""

import logging
import json
import time
import re
from pathlib import Path
from multiprocessing import Pool, get_context
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2 as cv
import torch
import traceback

from .pipeline_config import PipelineConfig, ROIConfig
from .pipeline_logger import PipelineLogger
from .pipeline_utils import get_video_list, get_video_output_dir, get_video_info, sample_frames_from_video
from .ring_detector import RingDetector


logger = logging.getLogger(__name__)


def _extract_video_year(video_path: str, video_stem: str) -> Optional[int]:
    """Extract a 4-digit year from the video stem or path."""
    year_pattern = re.compile(r'(?<!\d)(20\d{2})(?!\d)')

    for candidate in (video_stem, video_path):
        match = year_pattern.search(candidate)
        if match:
            return int(match.group(1))

    return None


def _process_frame_batch(batch_frames, batch_indices, detector, conf_threshold):
    """
    Process a batch of frames through YOLO inference.

    Returns list of (frame_idx, result, frame) tuples.
    """
    results = []
    try:
        # Convert frames to RGB for YOLO
        frames_rgb = [cv.cvtColor(f, cv.COLOR_BGR2RGB) for f in batch_frames]

        # Run YOLO inference on batch
        batch_results = detector.model.predict(frames_rgb, verbose=False)

        # Pair results with frame indices and original frames
        for frame_idx, result, frame in zip(batch_indices, batch_results, batch_frames):
            results.append((frame_idx, result, frame))

    except Exception as e:
        logger.error(f"Error processing frame batch: {e}")

    return results


def _detection_in_roi(bbox: Dict, frame_height: int, frame_width: int, roi_bbox: Tuple[float, float, float, float]) -> bool:
    """
    Check if a normalized detection bbox overlaps with the ROI bounding box.

    Args:
        bbox: Normalized bbox dict with x_norm, y_norm, w_norm, h_norm
        frame_height: Height of frame in pixels
        frame_width: Width of frame in pixels
        roi_bbox: ROI bounding box as (xmin, ymin, xmax, ymax) in pixel coordinates

    Returns:
        True if detection center is within ROI bbox, False otherwise
    """
    try:
        # Convert normalized coords to pixel coords
        x_center_px = bbox['x_norm'] * frame_width
        y_center_px = bbox['y_norm'] * frame_height

        # Check if center is within ROI bbox
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi_bbox
        return (roi_xmin <= x_center_px <= roi_xmax and
                roi_ymin <= y_center_px <= roi_ymax)

    except Exception as e:
        logger.debug(f"Error checking detection in ROI: {e}")
        return False



def _process_batch_with_crops(batch_frames: list, batch_indices: list, batch_frames_raw: list,
                              detector, conf_threshold: float, output_dir: Path,
                              frame_dims: Dict, roi_bbox: Optional[Tuple],
                              crop_save_executor, pending_saves: list,
                              video_year: Optional[int] = None) -> Tuple[list, int, int]:
    """
    Process one batch: Run YOLO inference and immediately generate/queue crops.

    This interleaves GPU inference (on next batch) with CPU crop generation/saving (current batch).

    Args:
        batch_frames: Preprocessed BGR frames for YOLO
        batch_indices: Frame indices corresponding to batch_frames
        batch_frames_raw: Raw original frames (for crop extraction)
        detector: RingDetector instance
        conf_threshold: Confidence threshold
        output_dir: Output directory for crops
        frame_dims: Dict with frame height/width
        roi_bbox: Optional ROI bounding box for filtering
        crop_save_executor: ThreadPoolExecutor for async saves
        pending_saves: List to accumulate futures for tracking
        video_year: Video year for logging

    Returns:
        Tuple of (detections list, crop_count, filtered_count)
    """
    detections = []
    crop_count = 0
    filtered_count = 0

    # Run YOLO inference on batch
    batch_results = _process_frame_batch(batch_frames, batch_indices, detector, conf_threshold)

    # Immediately process results: generate crops while GPU is ready for next batch
    for (frame_idx_batch, result, _), frame_raw in zip(batch_results, batch_frames_raw):
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

        # Skip filtering if confidence too low
        if confidence < conf_threshold:
            continue

        # Check ROI if configured
        if roi_bbox and bbox:
            x_center_px = bbox['x_norm'] * frame_dims['width']
            y_center_px = bbox['y_norm'] * frame_dims['height']
            roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi_bbox

            if not (roi_xmin <= x_center_px <= roi_xmax and
                    roi_ymin <= y_center_px <= roi_ymax):
                filtered_count += 1
                continue
        elif not bbox:
            continue

        # Generate and queue crop save (async)
        if bbox is not None:
            try:
                crop = crop_ring_from_frame(frame_raw, bbox, crop_size=224)
                if crop is not None:
                    crop_path = output_dir / f"{frame_idx_batch:06d}_ring.png"
                    future = crop_save_executor.submit(_save_crop_image, crop, crop_path)
                    pending_saves.append(future)
                    crop_count += 1
            except Exception as e:
                logger.debug(f"Failed to queue crop save for frame {frame_idx_batch}: {e}")

    return detections, crop_count, filtered_count


def process_video_with_yolo(video_path: str, camera: str, model_path: str,
                            conf_threshold: float, frame_sample_rate: int,
                            output_root: str, batch_size: int = 16, device_id: int = 0,
                            roi_config: Optional[ROIConfig] = None,
                            save_crops: bool = True) -> Dict[str, Any]:
    """
    GPU worker: Interleaved YOLO inference and crop generation.

    Processes video in batches where YOLO inference and crop generation happen concurrently:
    1. Load batch of raw frames
    2. Run YOLO inference on batch
    3. Generate crops and queue async saves for this batch
    4. While I/O is happening, load next batch and repeat

    This parallelizes GPU inference with CPU crop generation + I/O for better throughput.

    Args:
        video_path: Path to video file
        camera: Camera name
        model_path: Path to YOLO model
        conf_threshold: Confidence threshold for detections
        frame_sample_rate: Sample every Nth frame (1 = all frames)
        output_root: Output root directory
        batch_size: Number of frames to process per YOLO batch (default 16)
        device_id: GPU device ID to use (default 0)
        roi_config: ROI configuration for masking detections to specific regions
        save_crops: Whether to save crop images to disk (default True)
    """
    video_stem = Path(video_path).stem
    video_year = _extract_video_year(video_path, video_stem)
    start_time = time.time()

    try:
        # Get video info
        video_info = get_video_info(video_path)
        frame_height = video_info.get('height', 1080)
        frame_width = video_info.get('width', 1920)
        total_frames = video_info.get('frame_count', 0)

        logger.info(f"[{camera}/{video_stem}] Starting interleaved YOLO+crop processing: {total_frames} frames")

        # Create output directory for crops and detections
        output_dir = get_video_output_dir(output_root, camera, video_stem, step=1)
        frame_dims = {'height': frame_height, 'width': frame_width}

        # Initialize detector with device_id
        detector = RingDetector(
            model_path,
            conf=conf_threshold,
            iou=0.7,
            max_det=1,
            device_id=device_id,
        )

        # Get ROI bbox if available
        roi_bbox = None
        if roi_config:
            roi_bbox = roi_config.get_roi_bbox(camera, year=video_year)
            if roi_bbox:
                logger.info(f"Using ROI bbox: {roi_bbox}")

        # Thread pool for async crop saving (6 workers)
        crop_save_executor = ThreadPoolExecutor(max_workers=6)
        pending_saves = []

        # Collect frames in batches with both raw and RGB versions for efficient processing
        all_detections = []
        crop_count = 0
        filtered_count = 0
        batch_frames_rgb = []
        batch_frames_raw = []
        batch_indices = []

        for frame_idx, frame in sample_frames_from_video(video_path, sample_rate=frame_sample_rate):
            # Store both raw (for crop) and RGB (for YOLO)
            batch_frames_raw.append(frame)
            batch_frames_rgb.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            batch_indices.append(frame_idx)

            # Process batch when full or at end of stream
            if len(batch_frames_rgb) >= batch_size:
                # Process this batch: YOLO inference + immediate crop generation (interleaved)
                detections, batch_crop_count, batch_filtered_count = _process_batch_with_crops(
                    batch_frames_rgb, batch_indices, batch_frames_raw,
                    detector, conf_threshold, output_dir,
                    frame_dims, roi_bbox,
                    crop_save_executor, pending_saves,
                    video_year
                )
                all_detections.extend(detections)
                crop_count += batch_crop_count
                filtered_count += batch_filtered_count

                batch_frames_rgb.clear()
                batch_frames_raw.clear()
                batch_indices.clear()
                torch.cuda.empty_cache()

        # Process remaining frames
        if batch_frames_rgb:
            detections, batch_crop_count, batch_filtered_count = _process_batch_with_crops(
                batch_frames_rgb, batch_indices, batch_frames_raw,
                detector, conf_threshold, output_dir,
                frame_dims, roi_bbox,
                crop_save_executor, pending_saves,
                video_year
            )
            all_detections.extend(detections)
            crop_count += batch_crop_count
            filtered_count += batch_filtered_count

            torch.cuda.empty_cache()

        # Wait for all crop saves to complete
        if pending_saves:
            logger.debug(f"Waiting for {len(pending_saves)} crop saves to complete...")
            completed = 0
            failed = 0
            for i, future in enumerate(pending_saves):
                try:
                    future.result(timeout=10)
                    completed += 1
                except Exception as e:
                    logger.debug(f"Crop save task {i+1}/{len(pending_saves)} failed: {e}")
                    failed += 1
                if (i + 1) % 100 == 0:
                    logger.debug(f"  Progress: {i + 1}/{len(pending_saves)} saves completed")

            crop_save_executor.shutdown(wait=True)
            if failed > 0:
                logger.warning(f"  {failed}/{len(pending_saves)} crop saves failed (but processing complete)")

        # Count detections above confidence threshold for stats
        detected_count = sum(1 for d in all_detections if d['detected'] and d['confidence'] >= conf_threshold)
        conf_values = [d['confidence'] for d in all_detections if d['confidence'] > 0]

        elapsed = time.time() - start_time

        # Log summary
        logger.info(f"[{camera}/{video_stem}] Complete in {elapsed:.1f}s")
        logger.info(f"[{camera}/{video_stem}] Results: {detected_count} detections, {crop_count} crops saved ({filtered_count} filtered by ROI)")

        # Save filtered metadata for compatibility
        filtered_detections = [d for d in all_detections if d['detected'] and d['confidence'] >= conf_threshold]
        if roi_bbox:
            # If ROI filtering was applied, also save that info
            filtered_detections = [
                d for d in filtered_detections
                if d['bbox'] and _detection_in_roi(d['bbox'], frame_dims['height'], frame_dims['width'], roi_bbox)
            ]

        metadata = {
            'video_path': video_path,
            'camera': camera,
            'detections': filtered_detections,
            'ring_detect_count': detected_count,
            'frames_total': total_frames,
            'frames_processed': len(all_detections),
        }
        metadata_path = output_dir / "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            'video_path': video_path,
            'video_stem': video_stem,
            'camera': camera,
            'status': 'success',
            'error': None,
            'detections': all_detections,
            'frames_total': total_frames,
            'frames_processed': len(all_detections),
            'ring_detect_count': detected_count,
            'ring_crop_count': crop_count,
            'ring_filtered_count': filtered_count,
            'ring_conf_mean': float(np.mean(conf_values)) if conf_values else 0.0,
            'ring_conf_max': float(np.max(conf_values)) if conf_values else 0.0,
            'processing_seconds': elapsed,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{camera}/{video_stem}] ERROR: {e}")
        logger.error(traceback.format_exc())
        return {
            'video_path': video_path,
            'video_stem': video_stem,
            'camera': camera,
            'status': 'error',
            'error': str(e),
            'detections': [],
            'frames_total': 0,
            'frames_processed': 0,
            'ring_detect_count': 0,
            'ring_crop_count': 0,
            'ring_filtered_count': 0,
            'ring_conf_mean': 0.0,
            'ring_conf_max': 0.0,
            'processing_seconds': elapsed,
        }


def crop_ring_from_frame(frame: np.ndarray, bbox: Dict, crop_size: int = 224) -> Optional[np.ndarray]:
    """
    Crop ring region from frame based on normalized bbox.

    Args:
        frame: Input frame (BGR)
        bbox: Normalized bbox dict with x_norm, y_norm, w_norm, h_norm
        crop_size: Size of crop to extract

    Returns:
        Cropped image or None if crop fails
    """
    try:
        height, width = frame.shape[:2]

        # Convert normalized coords to pixel coords with proper rounding
        # Use round() for consistent center positioning, not int() which truncates
        x_center = round(bbox['x_norm'] * width)
        y_center = round(bbox['y_norm'] * height)

        # Clamp center to frame bounds
        x_center = max(crop_size // 2, min(width - crop_size // 2, x_center))
        y_center = max(crop_size // 2, min(height - crop_size // 2, y_center))

        # Extract square crop centered on ring (now guaranteed to be within bounds)
        x1 = x_center - crop_size // 2
        y1 = y_center - crop_size // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        crop = frame[y1:y2, x1:x2]

        # Validate crop is correct size (should be guaranteed by clamping above)
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            padded = np.zeros((crop_size, crop_size, 3), dtype=frame.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        return crop

    except Exception as e:
        logger.debug(f"Crop extraction failed: {e}")
        return None


def _save_crop_image(crop: np.ndarray, crop_path: Path) -> bool:
    """
    Save a crop image to disk (used by thread pool for async I/O).

    Args:
        crop: Crop image (BGR)
        crop_path: Path to save to

    Returns:
        True if successful, False otherwise
    """
    try:
        cv.imwrite(str(crop_path), crop)
        return True
    except Exception as e:
        logger.debug(f"Failed to save crop {crop_path}: {e}")
        return False


def run_step1_ring_detection(config: PipelineConfig, db_logger: PipelineLogger, force_reprocess: bool = False) -> bool:
    """
    Execute Step 1: Ring Detection on all videos.

    Uses GPU workers to process videos in parallel.
    Each GPU worker processes one video with direct YOLO inference.
    """
    logger.info("Starting Step 1: Ring Detection")
    step_start_time = time.time()

    try:
        # Get list of videos to process
        if not force_reprocess:
            completed = db_logger.get_completed_videos(step="step1")
        else:
            completed = []
            logger.info("Force reprocess mode: ignoring previous completions for Step 1")
        all_videos = get_video_list(config, step=1, completed_videos=completed)

        if not all_videos:
            logger.warning("No videos to process in Step 1")
            return True

        logger.info(f"Processing {len(all_videos)} videos in Step 1 using {len(config.step1.gpu_devices)} GPUs")

        # Use spawn context for CUDA compatibility (fork doesn't work with CUDA)
        ctx = get_context('spawn')
        gpu_pool = ctx.Pool(len(config.step1.gpu_devices))

        # Submit tasks - each video goes to a GPU worker
        model_path = str(config.models.ring_detector.get_model_path())

        tasks = []
        for video_idx, (video_path, camera) in enumerate(all_videos):
            # Assign GPU device in round-robin fashion
            device_id = config.step1.gpu_devices[video_idx % len(config.step1.gpu_devices)]

            task = gpu_pool.apply_async(
                process_video_with_yolo,
                (video_path, camera, model_path, config.step1.conf_threshold,
                 config.step1.frame_sample_rate, config.output_root_path,
                 config.step1.batch_size, device_id, config.roi_config, config.step1.save_crops)
            )
            tasks.append(task)

        logger.info(f"Submitted {len(tasks)} videos to GPU workers")

        # Collect results and log as they complete
        for task_idx, task in enumerate(tasks, 1):
            try:
                result = task.get(timeout=3600)  # 1 hour timeout per video

                video_path = result['video_path']
                video_stem = result['video_stem']
                camera = result['camera']

                if result['status'] == 'success':
                    # Create output directory
                    output_dir = get_video_output_dir(
                        config.output_root_path, camera, video_stem, step=1
                    )

                    # Save metadata JSON
                    metadata = {
                        'video_path': video_path,
                        'camera': camera,
                        'detections': result['detections'],
                        'ring_detect_count': result['ring_detect_count'],
                        'frames_total': result['frames_total'],
                        'frames_processed': result['frames_processed'],
                    }

                    metadata_path = output_dir / "_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    # Log to database
                    db_stats = {
                        'status': 'success',
                        'error_message': None,
                        'processing_seconds': result['processing_seconds'],
                        'frames_total': result['frames_total'],
                        'frames_processed': result['frames_processed'],
                        'ring_detect_count': result['ring_detect_count'],
                        'ring_crop_count': result['ring_crop_count'],
                        'ring_filtered_count': result['ring_filtered_count'],
                        'ring_detect_rate': result['ring_detect_count'] / max(1, result['frames_processed']),
                        'ring_conf_mean': result['ring_conf_mean'],
                        'ring_conf_max': result['ring_conf_max'],
                        'output_path': str(output_dir),
                        'queue_file': str(metadata_path),
                    }

                    db_logger.log_step1_result(video_path, camera, db_stats)

                    # Log with ROI filtering info if applicable
                    if result['ring_filtered_count'] > 0:
                        logger.info(f"✓ [{task_idx}/{len(tasks)}] {video_stem}: {result['ring_crop_count']} crops saved from {result['ring_detect_count']} detections ({result['ring_filtered_count']} filtered by ROI)")
                    else:
                        logger.info(f"✓ [{task_idx}/{len(tasks)}] {video_stem}: {result['ring_crop_count']} crops in {result['frames_processed']} frames")

                else:
                    # Log error to database
                    db_stats = {
                        'status': 'error',
                        'error_message': result['error'],
                        'processing_seconds': result['processing_seconds'],
                        'frames_total': 0,
                        'frames_processed': 0,
                        'ring_detect_count': 0,
                        'ring_detect_rate': 0.0,
                        'ring_conf_mean': 0.0,
                        'ring_conf_max': 0.0,
                        'output_path': '',
                        'queue_file': '',
                    }

                    db_logger.log_step1_result(video_path, camera, db_stats)
                    logger.error(f"✗ [{task_idx}/{len(tasks)}] {video_stem}: {result['error']}")

            except Exception as e:
                logger.error(f"Error processing task {task_idx}: {e}")
                traceback.print_exc()

        # Cleanup
        gpu_pool.close()
        gpu_pool.join()

        elapsed = time.time() - step_start_time
        logger.info(f"✓ Step 1 completed in {elapsed:.1f} seconds")
        return True

    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        logger.error(traceback.format_exc())
        return False
