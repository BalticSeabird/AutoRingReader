"""
Step 2: Tag Detection - Extract and rectify tags from ring crops.

This step:
1. Reads ring crop images generated in Step 1
2. Uses YOLO-seg model to detect and segment tags
3. Applies perspective transform to normalize tag orientation
4. Saves tag crops to disk
5. Logs results to database
"""

import logging
import json
import time
import sys
from pathlib import Path
from multiprocessing import Pool, Process, get_context, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2 as cv
import traceback

from pipeline_config import PipelineConfig
from pipeline_logger import PipelineLogger
from pipeline_utils import (
    get_video_output_dir, ensure_step_output_dir, cuda_device_context,
    load_image_batch, batch_items
)
from tag_detector import TagDetector


logger = logging.getLogger(__name__)


def ensure_console_logging() -> None:
    """Ensure worker processes can emit INFO logs to the console."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    else:
        root_logger.setLevel(logging.INFO)


def load_ring_crops_worker(crop_paths: List[str], queue, logger_: logging.Logger):
    """
    Worker process: Load ring crop images and put them in queue.

    Args:
        crop_paths: List of paths to ring crop images
        queue: Queue to put images in
        logger_: Logger instance
    """
    ensure_console_logging()
    logger_.info(f"Loading {len(crop_paths)} ring crops")
    for crop_path in crop_paths:
        try:
            img = cv.imread(str(crop_path))
            if img is not None:
                item = {
                    'crop_path': crop_path,
                    'crop_name': Path(crop_path).stem,
                    'image': img,
                }
                queue.put(item)
            else:
                logger_.warning(f"Failed to load crop image: {crop_path}")
        except Exception as e:
            logger_.error(f"Error loading crop {crop_path}: {e}")


def gpu_tag_detection_worker(device_id: int, image_queue, result_queue,
                             model_path: str, config: PipelineConfig, logger_: logging.Logger):
    """
    Worker process: YOLO-seg inference on batches of ring crops.

    Args:
        device_id: GPU device ID
        image_queue: Queue to get images from
        result_queue: Queue to put results in
        model_path: Path to YOLO-seg model
        config: Pipeline configuration
        logger_: Logger instance
    """
    try:
        ensure_console_logging()
        with cuda_device_context(device_id):
            detector = TagDetector(model_path, device_id=device_id)
            logger_.info(f"GPU {device_id}: Loaded TagDetector model")

            batch = []
            while True:
                try:
                    item = image_queue.get(timeout=5.0)

                    if item is None:  # Sentinel
                        if batch:
                            process_tag_batch(batch, detector, result_queue, logger_)
                        break

                    batch.append(item)

                    if len(batch) >= config.step2.batch_size:
                        process_tag_batch(batch, detector, result_queue, logger_)
                        batch = []

                except Exception as e:
                    logger_.debug(f"GPU {device_id} timeout or error: {e}")

    except Exception as e:
        logger_.error(f"GPU {device_id} initialization failed: {e}")
        logger_.error(traceback.format_exc())


def process_tag_batch(batch: List[Dict], detector: TagDetector, result_queue,
                      logger_: logging.Logger):
    """Process a batch of ring crops through the tag detector."""
    try:
        for item in batch:
            img = item['image']
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # Use existing detector method which handles YOLO-seg + mask→crop transformation
            tag_results = detector.predict_tag_with_meta(img_rgb)

            if tag_results:
                for tag_idx, (tag_crop, confidence) in enumerate(tag_results):
                    if tag_crop is not None:
                        # Convert PIL Image to numpy array if needed
                        if hasattr(tag_crop, 'shape'):
                            tag_crop_np = tag_crop
                        else:
                            tag_crop_np = np.array(tag_crop)

                        result_queue.put({
                            'crop_path': item['crop_path'],
                            'crop_name': item['crop_name'],
                            'detected': True,
                            'tag_crop': tag_crop_np,
                            'confidence': float(confidence) if confidence is not None else 0.0,
                            'tag_idx': tag_idx,
                        })
            else:
                # No tags detected in this ring crop
                result_queue.put({
                    'crop_path': item['crop_path'],
                    'crop_name': item['crop_name'],
                    'detected': False,
                    'tag_crop': None,
                    'confidence': 0.0,
                    'tag_idx': 0,
                })

    except Exception as e:
        logger_.error(f"Tag batch processing failed: {e}")
        # Put items with error status
        for item in batch:
            result_queue.put({
                'crop_path': item['crop_path'],
                'crop_name': item['crop_name'],
                'detected': False,
                'tags': [],
            })


def extract_tag_detections(yolo_result, ring_item: Dict, detector: TagDetector,
                          ring_image: np.ndarray) -> List[Dict]:
    """
    Extract tag detections from YOLO-seg result using existing detector methods.

    Returns list of tag detection dicts
    """
    tags = []

    try:
        # Use existing predict_tag_with_meta which handles mask→crop transformation
        tag_results = detector.predict_tag_with_meta(ring_image)

        for tag_idx, (tag_crop, confidence) in enumerate(tag_results):
            if tag_crop is not None:
                # Convert PIL Image to numpy array if needed
                if hasattr(tag_crop, 'shape'):
                    tag_crop_np = tag_crop
                else:
                    tag_crop_np = np.array(tag_crop)

                tags.append({
                    'crop_path': ring_item['crop_path'],
                    'crop_name': ring_item['crop_name'],
                    'detected': True,
                    'tag_crop': tag_crop_np,
                    'confidence': float(confidence) if confidence is not None else 0.0,
                    'tag_idx': tag_idx,
                })

    except Exception as e:
        logger.debug(f"Error extracting tag detections: {e}")

    return tags


def save_tag_worker(result_queue, output_root: str, db_logger: PipelineLogger,
                   config: PipelineConfig, logger_: logging.Logger, all_crops_list: List[str]):
    """
    Worker process: Save tag crops and log results per-video as they complete.

    Enables resume support by logging each video to database when its crops finish.

    Args:
        result_queue: Queue to get tag results from
        output_root: Root output directory
        db_logger: Database logger
        config: Pipeline configuration
        logger_: Logger instance
        all_crops_list: List of all crop paths (used to track completion per video)
    """
    ensure_console_logging()
    # Pre-index crops by video to track completion
    video_crops = defaultdict(list)
    for crop_path in all_crops_list:
        # Extract video stem from path: output/step1_ring_crops/{camera}/{video_stem}/{idx}_ring.jpg
        parts = Path(crop_path).parts
        if len(parts) >= 2:
            video_stem = parts[-2]
            video_crops[video_stem].append(crop_path)

    crops_per_video = {v: len(paths) for v, paths in video_crops.items()}
    total_crops = len(all_crops_list)
    logger_.info(f"Pre-indexed {len(crops_per_video)} videos with {total_crops} total crops")

    # Track stats per video
    video_stats = defaultdict(lambda: {
        'rings_processed': 0,
        'tags_detected_count': 0,
        'tags_detected_frames': set(),
        'tag_conf_values': [],
        'camera': "UNKNOWN",
    })
    processed_crops_per_video = defaultdict(int)

    try:
        sentinel_count = 0
        max_sentinels = config.step2.n_processes_gpu
        result_count = 0
        completed_videos = set()
        last_progress_count = 0
        last_progress_time = time.time()
        progress_interval = 10.0  # Report progress every 10 seconds

        while True:
            try:
                result = result_queue.get(timeout=10.0)

                if result is None:
                    sentinel_count += 1
                    logger_.info(f"Received sentinel {sentinel_count}/{max_sentinels}")
                    if sentinel_count >= max_sentinels:
                        logger_.info("All GPU workers finished, stopping result consumer")
                        break
                    continue

                result_count += 1
                crop_path = result['crop_path']
                crop_name = result['crop_name']

                # Extract video info from crop path
                parts = Path(crop_path).parts
                camera = parts[-3] if len(parts) >= 3 else "UNKNOWN"
                video_stem = parts[-2] if len(parts) >= 2 else "unknown"
                video_key = video_stem

                # Track this crop as processed for this video
                processed_crops_per_video[video_key] += 1

                if result['detected']:
                    video_stats[video_key]['tags_detected_count'] += 1
                    video_stats[video_key]['tags_detected_frames'].add(crop_name)
                    video_stats[video_key]['tag_conf_values'].append(result['confidence'])
                    video_stats[video_key]['camera'] = camera

                    # Save tag crop
                    try:
                        output_dir = get_video_output_dir(output_root, camera, video_stem, step=2)
                        tag_path = output_dir / f"{crop_name}_tag_{result['tag_idx']}.jpg"
                        # Convert RGB to BGR for cv.imwrite (which expects BGR)
                        tag_crop_bgr = cv.cvtColor(result['tag_crop'], cv.COLOR_RGB2BGR)
                        cv.imwrite(str(tag_path), tag_crop_bgr)
                    except Exception as e:
                        logger_.debug(f"Failed to save tag crop: {e}")

                # Check if this video is complete (all crops processed)
                if (video_key not in completed_videos and
                    video_key in crops_per_video and
                    processed_crops_per_video[video_key] == crops_per_video[video_key]):

                    stats = video_stats[video_key]
                    camera = stats['camera']

                    logger_.info(f"✓ Video complete: {video_stem} ({stats['tags_detected_count']} tags from {processed_crops_per_video[video_key]} crops)")

                    # Log to database immediately so we can resume from here
                    db_stats = {
                        'status': 'success',
                        'processing_seconds': 0,
                        'rings_processed': processed_crops_per_video[video_key],
                        'tags_detected_count': stats['tags_detected_count'],
                        'tags_detected_frames': len(stats['tags_detected_frames']),
                        'tag_detection_rate': stats['tags_detected_count'] / max(1, processed_crops_per_video[video_key]),
                        'tag_conf_mean': float(np.mean(stats['tag_conf_values'])) if stats['tag_conf_values'] else 0.0,
                        'tag_conf_max': float(np.max(stats['tag_conf_values'])) if stats['tag_conf_values'] else 0.0,
                    }
                    db_logger.log_step2_result(video_stem, camera, db_stats)
                    completed_videos.add(video_key)

                # Progress reporting every N seconds
                current_time = time.time()
                if current_time - last_progress_time >= progress_interval:
                    remaining_crops = total_crops - result_count
                    percent_done = (result_count / total_crops * 100) if total_crops > 0 else 0
                    crops_per_second = (result_count - last_progress_count) / (current_time - last_progress_time)
                    eta_seconds = remaining_crops / crops_per_second if crops_per_second > 0 else 0

                    eta_str = ""
                    if eta_seconds > 0:
                        if eta_seconds < 60:
                            eta_str = f" (ETA: {eta_seconds:.0f}s)"
                        elif eta_seconds < 3600:
                            eta_str = f" (ETA: {eta_seconds/60:.1f}m)"
                        else:
                            eta_str = f" (ETA: {eta_seconds/3600:.1f}h)"

                    logger_.info(f"Progress: {result_count}/{total_crops} crops ({percent_done:.1f}%) | Videos: {len(completed_videos)} complete{eta_str}")
                    last_progress_count = result_count
                    last_progress_time = current_time

            except Exception as e:
                logger_.debug(f"Save worker timeout or error: {e}")

        logger_.info(f"✓ Result consumer processed {result_count}/{total_crops} crops from {len(completed_videos)} completed videos")

    except Exception as e:
        logger_.error(f"Save worker failed: {e}")
        logger_.error(traceback.format_exc())



def discover_ring_crop_events(output_root: str, camera: str) -> List[Tuple[str, List[str]]]:
    """
    Discover Step 1 crop event folders for one camera.

    The expected layout is:
        step1_ring_crops/<camera>/<video_stem>/*_ring.jpg

    This only inspects immediate subfolders under the camera directory and then
    collects crop files with a non-recursive glob, so we do not walk the full tree.

    Args:
        output_root: Root output directory
        camera: Camera name to filter

    Returns:
        List of (video_stem, crop_paths) tuples, one per event folder.
    """
    ring_dir = Path(output_root) / "step1_ring_crops" / camera
    events: List[Tuple[str, List[str]]] = []

    if not ring_dir.exists():
        logger.info(f"No Step 1 crop directory found for camera {camera}: {ring_dir}")
        return events

    for event_dir in sorted(path for path in ring_dir.iterdir() if path.is_dir()):
        crop_paths = sorted(str(crop_file) for crop_file in event_dir.glob("*_ring.png"))
        if not crop_paths:
            # Also check for JPG files for backward compatibility
            crop_paths = sorted(str(crop_file) for crop_file in event_dir.glob("*_ring.jpg"))
        if not crop_paths:
            logger.info(f"Skipping empty event folder for camera {camera}: {event_dir.name}")
            continue

        logger.info(
            f"Discovered event folder for camera {camera}: {event_dir.name} with {len(crop_paths)} ring crops"
        )
        events.append((event_dir.name, crop_paths))

    return events


def process_ring_crop_event(
    camera: str,
    video_stem: str,
    crop_paths: List[str],
    config: PipelineConfig,
    db_logger: PipelineLogger,
    force_reprocess: bool,
    output_root: str,
    logger_: logging.Logger,
) -> bool:
    """
    Process one event folder of ring crops end to end.

    Each event is handled independently so we only load/process/save one folder at a time.
    """
    logger_.info(f"Starting event {camera}/{video_stem} with {len(crop_paths)} crops")

    if not crop_paths:
        logger_.info(f"No crops to process for event {camera}/{video_stem}")
        return True

    completed_videos = set()
    if not force_reprocess:
        completed_videos = set(db_logger.get_completed_videos(step="step2"))
        if video_stem in completed_videos:
            logger_.info(f"Skipping completed event {camera}/{video_stem}")
            return True

    with Manager() as manager:
        image_queue = manager.Queue(maxsize=max(64, len(crop_paths)))
        result_queue = manager.Queue(maxsize=max(64, len(crop_paths)))

        ctx = get_context('spawn')
        cpu_pool = ctx.Pool(config.step2.n_processes_cpu)
        gpu_pool = ctx.Pool(config.step2.n_processes_gpu)

        try:
            crop_batches = list(batch_items(crop_paths, batch_size=10))
            logger_.info(
                f"[{camera}/{video_stem}] Submitting {len(crop_batches)} loading batches for one event folder"
            )

            load_tasks = [
                cpu_pool.apply_async(load_ring_crops_worker, (batch, image_queue, logger_))
                for batch in crop_batches
            ]

            inference_tasks = []
            model_path = str(config.models.tag_detector.get_model_path())
            logger_.info(f"[{camera}/{video_stem}] Using tag detector model: {model_path}")

            for gpu_id in config.step2.gpu_devices:
                logger_.info(f"[{camera}/{video_stem}] Starting GPU worker on device {gpu_id}")
                task = gpu_pool.apply_async(
                    gpu_tag_detection_worker,
                    (gpu_id, image_queue, result_queue, model_path, config, logger_)
                )
                inference_tasks.append(task)

            save_task = Process(
                target=save_tag_worker,
                args=(result_queue, output_root, db_logger, config, logger_, crop_paths)
            )
            logger_.info(f"[{camera}/{video_stem}] Starting result saver process")
            save_task.start()

            logger_.info(f"[{camera}/{video_stem}] Waiting for crop loading to complete")
            for task in load_tasks:
                task.get(timeout=600)

            logger_.info(f"[{camera}/{video_stem}] Crop loading complete, sending sentinels")
            for _ in range(config.step2.n_processes_gpu):
                image_queue.put(None)

            logger_.info(f"[{camera}/{video_stem}] Waiting for GPU workers to complete inference")
            for task in inference_tasks:
                task.get(timeout=3600)

            logger_.info(f"[{camera}/{video_stem}] Tag detection complete, sending sentinels")
            for _ in range(config.step2.n_processes_gpu):
                result_queue.put(None)

            logger_.info(f"[{camera}/{video_stem}] Waiting for result saver to finish")
            save_task.join(timeout=600)
            logger_.info(f"[{camera}/{video_stem}] Result saver process joined")

        finally:
            cpu_pool.close()
            cpu_pool.join()
            gpu_pool.close()
            gpu_pool.join()

    logger_.info(f"Finished event {camera}/{video_stem}")
    return True


def run_step2_tag_detection(config: PipelineConfig, db_logger: PipelineLogger, force_reprocess: bool = False) -> bool:
    """
    Execute Step 2: Tag Detection on all ring crops with parallel event processing.

    Processes multiple event folders in parallel (2-3 at a time) instead of sequentially.

    Args:
        config: Pipeline configuration
        db_logger: Database logger instance
        force_reprocess: If True, ignore resume and reprocess all completed videos

    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting Step 2: Tag Detection (Parallel Events)")
    step_start_time = time.time()

    try:
        # Collect all events from all cameras
        all_events = []
        for camera in config.input.cameras:
            event_folders = discover_ring_crop_events(config.output_root_path, camera)
            if not event_folders:
                logger.info(f"No ring crop event folders found for camera {camera}")
                continue

            logger.info(f"Camera {camera}: {len(event_folders)} event folders discovered")

            for video_stem, crop_paths in event_folders:
                all_events.append((camera, video_stem, crop_paths))

        if not all_events:
            logger.warning("No events to process in Step 2")
            return True

        total_events = len(all_events)
        logger.info(f"Processing {total_events} event folders with parallel workers")

        # Determine max parallel workers (limit to avoid GPU memory contention)
        # With 2 GPUs and multiple processes, allow 2-3 concurrent events
        max_parallel = min(3, len(config.step2.gpu_devices) + 1)
        logger.info(f"Using up to {max_parallel} parallel workers")

        # Process events in parallel using ProcessPoolExecutor
        completed_events = 0
        failed_events = []

        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all tasks
            futures_map = {}
            for camera, video_stem, crop_paths in all_events:
                future = executor.submit(
                    process_ring_crop_event,
                    camera=camera,
                    video_stem=video_stem,
                    crop_paths=crop_paths,
                    config=config,
                    db_logger=db_logger,
                    force_reprocess=force_reprocess,
                    output_root=config.output_root_path,
                    logger_=logger,
                )
                futures_map[future] = (camera, video_stem)

            # Collect results as they complete
            last_progress_time = time.time()
            progress_interval = 10.0

            for future in as_completed(futures_map):
                camera, video_stem = futures_map[future]
                try:
                    result = future.result(timeout=3600)
                    if result:
                        completed_events += 1
                        logger.info(f"✓ Event {completed_events}/{total_events}: {camera}/{video_stem}")
                    else:
                        failed_events.append((camera, video_stem))
                        logger.error(f"✗ Event {completed_events + len(failed_events)}/{total_events}: {camera}/{video_stem} (result=False)")

                except Exception as e:
                    failed_events.append((camera, video_stem))
                    logger.error(f"✗ Event processing failed for {camera}/{video_stem}: {e}")

                # Progress reporting
                current_time = time.time()
                if current_time - last_progress_time >= progress_interval:
                    percent_done = (completed_events + len(failed_events)) / total_events * 100
                    logger.info(f"Event processing: {completed_events + len(failed_events)}/{total_events} complete ({percent_done:.1f}%)")
                    last_progress_time = current_time

        elapsed = time.time() - step_start_time
        logger.info(f"✓ Step 2 completed in {elapsed:.1f} seconds")
        logger.info(f"  Completed: {completed_events}/{total_events} events")

        if failed_events:
            logger.warning(f"  Failed: {len(failed_events)} events")
            for camera, video_stem in failed_events:
                logger.warning(f"    - {camera}/{video_stem}")
            return False

        return True

    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        logger.error(traceback.format_exc())
        return False
