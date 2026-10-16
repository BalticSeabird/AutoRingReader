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

from pipeline_config import PipelineConfig, ROIConfig
from pipeline_logger import PipelineLogger
from pipeline_utils import get_video_list, get_video_output_dir, get_video_info, sample_frames_from_video
from ring_detector import RingDetector


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


def _crop_to_roi(frame: np.ndarray, roi_bbox: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop frame to ROI bounding box.

    Args:
        frame: Input frame
        roi_bbox: ROI bbox as (xmin, ymin, xmax, ymax) in pixel coordinates

    Returns:
        Tuple of (cropped_frame, transform_matrix) where transform_matrix can be used
        to offset detections from cropped frame back to full frame coordinates
    """
    xmin, ymin, xmax, ymax = [int(x) for x in roi_bbox]
    height, width = frame.shape[:2]

    # Clamp to frame boundaries
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)

    cropped_frame = frame[ymin:ymax, xmin:xmax]

    return cropped_frame, np.array([xmin, ymin])


def _offset_detections_from_crop(result, offset: np.ndarray, crop_shape: Tuple[int, int],
                                  full_frame_shape: Tuple[int, int]):
    """
    Offset YOLO detections from cropped frame back to full frame coordinates.

    Args:
        result: YOLO result object
        offset: [xmin, ymin] offset to apply
        crop_shape: (height, width) of cropped frame
        full_frame_shape: (height, width) of full frame

    Returns:
        Modified result with coordinates transformed to full frame
    """
    if not hasattr(result, 'boxes') or len(result.boxes) == 0:
        return result

    xmin_offset, ymin_offset = offset
    crop_height, crop_width = crop_shape
    full_height, full_width = full_frame_shape

    for box in result.boxes:
        # Get normalized coordinates from cropped frame (xywhn is already normalized to crop)
        x_norm_crop, y_norm_crop, w_norm, h_norm = box.xywhn[0]

        # Convert to pixel coords in cropped frame, then offset to full frame
        x_pixel_crop = x_norm_crop * crop_width
        y_pixel_crop = y_norm_crop * crop_height

        x_pixel_full = x_pixel_crop + xmin_offset
        y_pixel_full = y_pixel_crop + ymin_offset

        # Convert back to normalized coords for full frame
        x_norm_full = x_pixel_full / full_width
        y_norm_full = y_pixel_full / full_height

        # Update the normalized coordinates in-place
        box.xywhn[0][0] = x_norm_full
        box.xywhn[0][1] = y_norm_full

    return result


def _process_frame_batch_with_roi(batch_frames, batch_indices, detector, conf_threshold,
                                   roi_bbox: Optional[Tuple[float, float, float, float]] = None,
                                   full_frame_shape: Tuple[int, int] = (1080, 1920)):
    """
    Process a batch of frames through YOLO inference, optionally cropping to ROI first.

    Args:
        batch_frames: List of frames (BGR)
        batch_indices: Frame indices
        detector: RingDetector instance
        conf_threshold: Confidence threshold
        roi_bbox: Optional ROI bbox as (xmin, ymin, xmax, ymax) to crop before YOLO
        full_frame_shape: (height, width) of full frame for coordinate transformation

    Returns:
        List of (frame_idx, result, frame) tuples where frame is the FULL (uncropped) frame
    """
    results = []
    try:
        if roi_bbox:
            # Crop frames to ROI before YOLO (more efficient - less pixel data to process)
            cropped_frames = []
            crop_offsets = []
            crop_shapes = []

            for frame in batch_frames:
                cropped_frame, offset = _crop_to_roi(frame, roi_bbox)
                cropped_frames.append(cropped_frame)
                crop_offsets.append(offset)
                crop_shapes.append(cropped_frame.shape[:2])

            # Convert to RGB for YOLO
            frames_rgb = [cv.cvtColor(f, cv.COLOR_BGR2RGB) for f in cropped_frames]

            # Run YOLO inference on cropped frames
            batch_results = detector.model.predict(frames_rgb, verbose=False)

            # Offset detections back to full frame coordinates
            for i, (frame_idx, result, offset, crop_shape) in enumerate(
                zip(batch_indices, batch_results, crop_offsets, crop_shapes)
            ):
                result = _offset_detections_from_crop(result, offset, crop_shape, full_frame_shape)
                full_frame = batch_frames[i]
                results.append((frame_idx, result, full_frame))

        else:
            # No ROI - process full frames (original behavior)
            frames_rgb = [cv.cvtColor(f, cv.COLOR_BGR2RGB) for f in batch_frames]
            batch_results = detector.model.predict(frames_rgb, verbose=False)

            for frame_idx, result, frame in zip(batch_indices, batch_results, batch_frames):
                results.append((frame_idx, result, frame))

    except Exception as e:
        logger.error(f"Error processing frame batch: {e}")

    return results


def process_video_with_yolo(video_path: str, camera: str, model_path: str,
                            conf_threshold: float, frame_sample_rate: int,
                            output_root: str, batch_size: int = 16, device_id: int = 0,
                            roi_config: Optional[ROIConfig] = None,
                            save_crops: bool = True) -> Dict[str, Any]:
    """
    GPU worker: Process single video with YOLO inference AND save ring crops (async).

    Extracts and processes frames in batches to manage memory efficiently.
    If ROI is available for the camera, crops frames to ROI before YOLO inference (more efficient).
    Saves crops asynchronously to avoid blocking GPU inference.

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
        save_crops: Whether to save crop images to disk (default True for backward compat)
    """
    video_stem = Path(video_path).stem
    video_year = _extract_video_year(video_path, video_stem)
    start_time = time.time()

    try:
        # Get video info
        video_info = get_video_info(video_path)
        total_frames = video_info.get('frame_count', 0)

        # Load ROI for this camera if available
        roi_polygon = None
        roi_bbox = None
        if roi_config:
            roi_polygon = roi_config.get_roi_for_camera(camera, year=video_year)
            roi_bbox = roi_config.get_roi_bbox(camera, year=video_year)
            if roi_bbox:
                if video_year:
                    logger.info(f"[{camera}/{video_stem}] Using ROI for {video_year}: {roi_bbox}")
                else:
                    logger.info(f"[{camera}/{video_stem}] Using default ROI: {roi_bbox}")
            else:
                logger.info(f"[{camera}/{video_stem}] No ROI configured for year {video_year or 'default'}")

        logger.info(f"[{camera}/{video_stem}] Starting: {total_frames} frames on GPU {device_id}")


        # Initialize detector with device_id
        detector = RingDetector(
            model_path,
            conf=conf_threshold,
            iou=0.7,
            max_det=1,
            device_id=device_id,
        )

        # Create output directory for crops
        output_dir = get_video_output_dir(output_root, camera, video_stem, step=1)

        # Thread pool for async crop saving (don't block GPU inference)
        crop_save_executor = ThreadPoolExecutor(max_workers=2) if save_crops else None
        pending_saves = []

        # Process video: extract frames, run inference, save crops
        detections = []
        conf_values = []
        frame_count = 0
        detected_count = 0
        crop_count = 0

        # Process frames in batches to manage memory
        batch_frames = []
        batch_indices = []

        for frame_idx, frame in sample_frames_from_video(video_path, frame_sample_rate):
            batch_frames.append(frame)
            batch_indices.append(frame_idx)

            # Process batch when it reaches batch_size or at end of video
            if len(batch_frames) >= batch_size:
                batch_results = _process_frame_batch_with_roi(
                    batch_frames, batch_indices, detector, conf_threshold,
                    roi_bbox=roi_bbox, full_frame_shape=(video_info.get('height', 1080), video_info.get('width', 1920))
                )

                # Process and save results IMMEDIATELY (don't accumulate in memory)
                for frame_idx_batch, result, frame in batch_results:
                    frame_count += 1
                    detected = False
                    bbox = None
                    confidence = 0.0

                    # Extract detection from YOLO result
                    if hasattr(result, 'boxes') and len(result.boxes) > 0:
                        box = result.boxes[0]
                        x_norm, y_norm, w_norm, h_norm = box.xywhn[0]
                        confidence = float(box.conf[0])

                        detected = True
                        bbox = {
                            'x_norm': float(x_norm),
                            'y_norm': float(y_norm),
                            'w_norm': float(w_norm),
                            'h_norm': float(h_norm),
                        }

                        if confidence >= conf_threshold:
                            detected_count += 1
                            conf_values.append(confidence)

                            # Save crop image ASYNC (don't block GPU inference)
                            if save_crops:
                                try:
                                    crop = crop_ring_from_frame(frame, bbox, crop_size=224)
                                    if crop is not None:
                                        crop_path = output_dir / f"{frame_idx_batch:06d}_ring.jpg"
                                        # Submit save task to thread pool
                                        future = crop_save_executor.submit(_save_crop_image, crop, crop_path)
                                        pending_saves.append(future)
                                        crop_count += 1
                                except Exception as e:
                                    logger.debug(f"Failed to queue crop save for frame {frame_idx_batch}: {e}")

                    detections.append({
                        'frame_idx': frame_idx_batch,
                        'detected': detected,
                        'bbox': bbox,
                        'confidence': confidence,
                    })

                # Clear batch and GPU memory immediately
                batch_frames.clear()
                batch_indices.clear()
                torch.cuda.empty_cache()

        # Process remaining frames
        if batch_frames:
            batch_results = _process_frame_batch_with_roi(
                batch_frames, batch_indices, detector, conf_threshold,
                roi_bbox=roi_bbox, full_frame_shape=(video_info.get('height', 1080), video_info.get('width', 1920))
            )

            # Process and save remaining results IMMEDIATELY
            for frame_idx_batch, result, frame in batch_results:
                frame_count += 1
                detected = False
                bbox = None
                confidence = 0.0

                # Extract detection from YOLO result
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    box = result.boxes[0]
                    x_norm, y_norm, w_norm, h_norm = box.xywhn[0]
                    confidence = float(box.conf[0])

                    detected = True
                    bbox = {
                        'x_norm': float(x_norm),
                        'y_norm': float(y_norm),
                        'w_norm': float(w_norm),
                        'h_norm': float(h_norm),
                    }

                    if confidence >= conf_threshold:
                        detected_count += 1
                        conf_values.append(confidence)

                        # Save crop image ASYNC
                        if save_crops:
                            try:
                                crop = crop_ring_from_frame(frame, bbox, crop_size=224)
                                if crop is not None:
                                    crop_path = output_dir / f"{frame_idx_batch:06d}_ring.jpg"
                                    future = crop_save_executor.submit(_save_crop_image, crop, crop_path)
                                    pending_saves.append(future)
                                    crop_count += 1
                            except Exception as e:
                                logger.debug(f"Failed to queue crop save for frame {frame_idx_batch}: {e}")

                detections.append({
                    'frame_idx': frame_idx_batch,
                    'detected': detected,
                    'bbox': bbox,
                    'confidence': confidence,
                })

            torch.cuda.empty_cache()

        # Wait for all pending crop saves to complete
        if crop_save_executor and pending_saves:
            logger.debug(f"Waiting for {len(pending_saves)} pending crop saves...")
            for future in pending_saves:
                try:
                    future.result(timeout=30)  # 30s timeout per save
                except Exception as e:
                    logger.debug(f"Crop save task failed: {e}")
            crop_save_executor.shutdown(wait=True)

        elapsed = time.time() - start_time

        logger.info(f"[{camera}/{video_stem}] Complete: {detected_count}/{frame_count} frames, {crop_count} crops saved in {elapsed:.1f}s")

        return {
            'video_path': video_path,
            'video_stem': video_stem,
            'camera': camera,
            'status': 'success',
            'error': None,
            'detections': detections,
            'frames_total': total_frames,
            'frames_processed': frame_count,
            'ring_detect_count': detected_count,
            'ring_conf_mean': float(np.mean(conf_values)) if conf_values else 0.0,
            'ring_conf_max': float(np.max(conf_values)) if conf_values else 0.0,
            'processing_seconds': elapsed,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{camera}/{video_stem}] ERROR: {e}")
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

        # Convert normalized coords to pixel coords
        x_center = int(bbox['x_norm'] * width)
        y_center = int(bbox['y_norm'] * height)

        # Extract square crop centered on ring
        x1 = max(0, x_center - crop_size // 2)
        y1 = max(0, y_center - crop_size // 2)
        x2 = min(width, x1 + crop_size)
        y2 = min(height, y1 + crop_size)

        crop = frame[y1:y2, x1:x2]

        # Pad if necessary
        if crop.shape[0] < crop_size or crop.shape[1] < crop_size:
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
                        'ring_detect_rate': result['ring_detect_count'] / max(1, result['frames_processed']),
                        'ring_conf_mean': result['ring_conf_mean'],
                        'ring_conf_max': result['ring_conf_max'],
                        'output_path': str(output_dir),
                        'queue_file': str(metadata_path),
                    }

                    db_logger.log_step1_result(video_path, camera, db_stats)
                    logger.info(f"✓ [{task_idx}/{len(tasks)}] {video_stem}: {result['ring_detect_count']} rings in {result['frames_processed']} frames")

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
