"""
Step 3: Text Recognition - Recognize ring IDs from tag crops.

This step:
1. Discovers tag crop events (one per camera/video_stem folder)
2. Processes each event independently (event-by-event architecture)
3. Groups tag crop images by ring (tag_0, tag_1 together) within each event
4. Uses CLIP4STR model for OCR character recognition
5. Applies pattern matching and majority voting
6. Aggregates to final ring ID per video
7. Logs results to database immediately (enables resume)
8. Saves YAML output

Uses batched processing and optional multi-GPU support via multiprocessing.
Processes ONE EVENT AT A TIME to keep memory bounded (~500MB per event, not 30GB).
"""

import logging
import yaml
import time
import sys
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Manager, get_context
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2 as cv
from PIL import Image
import traceback

from .pipeline_config import PipelineConfig
from .pipeline_logger import PipelineLogger
from .pipeline_utils import get_video_output_dir
from .text_recognition import TextRecognition


logger = logging.getLogger(__name__)


def discover_tag_crop_events(output_root: str, camera: str) -> List[Tuple[str, List[str]]]:
    """
    Discover Step 2 tag crop event folders for one camera.

    Expected layout:
        step2_tag_crops/{camera}/{video_stem}/{frame_id}_ring_tag_{idx}.jpg

    This only inspects immediate subfolders under the camera directory and uses
    non-recursive glob, so we don't walk the full tree (unlike the old group_tag_crops_by_ring).

    Args:
        output_root: Root output directory
        camera: Camera name to filter

    Returns:
        List of (video_stem, crop_paths) tuples, one per event folder.
    """
    tag_dir = Path(output_root) / "step2_tag_crops" / camera
    events: List[Tuple[str, List[str]]] = []

    if not tag_dir.exists():
        logger.info(f"No Step 2 tag crop directory found for camera {camera}: {tag_dir}")
        return events

    # Iterate immediate subdirectories (one per video_stem)
    for event_dir in sorted(path for path in tag_dir.iterdir() if path.is_dir()):
        video_stem = event_dir.name

        # Collect tag crop files for this event
        crop_paths = sorted(str(crop_file) for crop_file in event_dir.glob("*_ring_tag_*.jpg"))

        if not crop_paths:
            logger.debug(f"Skipping empty event folder for camera {camera}: {video_stem}")
            continue

        logger.info(
            f"Discovered tag crop event for camera {camera}: {video_stem} with {len(crop_paths)} tag crops"
        )
        events.append((video_stem, crop_paths))

    return events


def group_tag_crops_by_ring_for_event(crop_paths: List[str]) -> Dict[Tuple[str, str, str], List[str]]:
    """
    Group tag crops by ring for a SINGLE EVENT.

    File structure: {crop_path} = .../step2_tag_crops/{camera}/{video_stem}/{frame_id}_ring_tag_{idx}.jpg
    Groups files by (camera, video_stem, frame_id) so tag_0, tag_1 are together.

    Args:
        crop_paths: List of crop file paths (only from one video_stem)

    Returns:
        Dict mapping (camera, video_stem, frame_id) to list of sorted crop paths
    """
    groups = defaultdict(list)

    for crop_file_str in crop_paths:
        crop_file = Path(crop_file_str)
        parts = crop_file.parts

        # Extract camera and video_stem from path structure
        camera = parts[-3] if len(parts) >= 3 else "UNKNOWN"
        video_stem = parts[-2] if len(parts) >= 2 else "unknown"

        # Parse frame_id and tag_idx from filename: {frame_id}_ring_tag_{idx}.jpg
        stem = crop_file.stem
        if "_ring_tag_" in stem:
            frame_id = stem.split("_ring_tag_")[0]
            key = (camera, video_stem, frame_id)
            groups[key].append(str(crop_file))

    # Sort each group by tag index
    for key in groups:
        groups[key] = sorted(groups[key])

    return dict(groups)


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


def gpu_text_recognition_worker(device_id: int, batch_queue, result_queue,
                                model_path: str, logger_: logging.Logger):
    """
    Worker process: Load TextRecognition model on specific GPU and process batches.

    Args:
        device_id: GPU device ID (0 or 1)
        batch_queue: Queue to get batches from (None is sentinel)
        result_queue: Queue to put results in
        model_path: Path to text recognition model
        logger_: Logger instance
    """
    ensure_console_logging()
    device_str = f"cuda:{device_id}"

    try:
        logger_.info(f"GPU {device_id}: Loading TextRecognition model from {model_path}")
        model = TextRecognition(
            model_path,
            device=device_str
        )
        logger_.info(f"GPU {device_id}: Model loaded successfully")

        while True:
            try:
                batch_item = batch_queue.get(timeout=5.0)

                if batch_item is None:  # Sentinel value
                    logger_.info(f"GPU {device_id}: Received sentinel, shutting down")
                    result_queue.put(None)
                    break

                batch_idx, group_batch_keys, group_batch, group_image_counts = batch_item

                # Flatten all images in batch
                all_images_batch = []
                for image_list in group_batch:
                    all_images_batch.extend(image_list)

                # Process batch on this GPU
                try:
                    tag_predictions = model.predict_tag_batch(all_images_batch)
                    result_queue.put({
                        'batch_idx': batch_idx,
                        'group_keys': group_batch_keys,
                        'group_image_counts': group_image_counts,
                        'predictions': tag_predictions,
                        'status': 'success',
                    })
                    logger_.info(f"GPU {device_id}: Processed batch {batch_idx} ({len(all_images_batch)} images)")

                except Exception as e:
                    logger_.error(f"GPU {device_id}: Batch {batch_idx} processing failed: {e}")
                    result_queue.put({
                        'batch_idx': batch_idx,
                        'status': 'error',
                        'error': str(e),
                    })

            except Exception as e:
                logger_.debug(f"GPU {device_id}: Queue timeout or error (this is normal): {e}")

    except Exception as e:
        logger_.error(f"GPU {device_id}: Initialization failed: {e}")
        logger_.error(traceback.format_exc())
    finally:
        logger_.info(f"GPU {device_id}: Worker process exiting")


def _run_step3_single_gpu(
    config: PipelineConfig,
    camera: str,
    video_stem: str,
    crop_paths: List[str],
    model: TextRecognition
) -> Dict:
    """
    Process one event's tag groups using single GPU with batched inference.

    Args:
        config: Pipeline configuration
        camera: Camera name for this event
        video_stem: Video stem for this event
        crop_paths: List of crop file paths (only from this event)
        model: TextRecognition model instance

    Returns:
        Dict with keys: 'final_tag', 'letters', 'numbers', 'detection_count',
                        'letters_confidence', 'numbers_confidence'
    """
    tag_groups = group_tag_crops_by_ring_for_event(crop_paths)
    total_groups = len(tag_groups)

    if total_groups == 0:
        logger.info(f"No tag groups found for {camera}/{video_stem}")
        return {}

    logger.info(f"Processing {total_groups} tag groups for {camera}/{video_stem} with single GPU")

    video_results = {
        'frames': [],
        'letters': [],
        'numbers': [],
        'letters_confs': [],
        'numbers_confs': [],
    }

    batch_size = config.step3.batch_size
    group_batch = []
    group_batch_keys = []
    group_batch_images = []
    group_image_counts = []

    processed_batches = 0
    last_progress_time = time.time()
    progress_interval = 5.0

    for i, (group_key, group_crop_paths) in enumerate(tag_groups.items()):
        frame_id = group_key[2]

        # Load images for this group
        images = []
        try:
            for crop_path in group_crop_paths:
                img = cv.imread(str(crop_path))
                if img is not None:
                    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    images.append(img_pil)
        except Exception as e:
            logger.debug(f"Error loading crops for {frame_id}: {e}")
            continue

        if not images:
            continue

        # Accumulate this group
        group_batch.append(images)
        group_batch_keys.append(group_key)
        group_image_counts.append(len(images))
        for img in images:
            group_batch_images.append(img)

        # Process batch when reaching batch_size or at end
        should_process = (len(group_batch) >= batch_size) or (i == total_groups - 1)

        if should_process:
            # Run batched OCR
            try:
                tag_predictions = model.predict_tag_batch(group_batch_images)

                # Map predictions back to groups, combining letters+numbers per ring
                pred_idx = 0
                for group_idx, group_key in enumerate(group_batch_keys):
                    cam, vstem, fid = group_key
                    images_in_group = group_image_counts[group_idx]

                    # Collect all predictions for this group
                    group_predictions = {}
                    for img_idx in range(images_in_group):
                        pred_dict = tag_predictions[pred_idx]
                        pred_idx += 1

                        # Collect both letters and numbers from this group
                        if pred_dict.get('letters'):
                            group_predictions['letters'] = pred_dict['letters']
                        if pred_dict.get('numbers'):
                            group_predictions['numbers'] = pred_dict['numbers']

                    # Only add result if we have BOTH letters and numbers for this ring
                    if group_predictions.get('letters') and group_predictions.get('numbers'):
                        letters = group_predictions['letters']['str']
                        numbers = group_predictions['numbers']['str']
                        letters_conf = float(np.mean(group_predictions['letters']['pred']))
                        numbers_conf = float(np.mean(group_predictions['numbers']['pred']))

                        video_results['frames'].append(fid)
                        video_results['letters'].append(letters)
                        video_results['numbers'].append(numbers)
                        video_results['letters_confs'].append(letters_conf)
                        video_results['numbers_confs'].append(numbers_conf)

            except Exception as e:
                logger.error(f"Batch OCR failed for batch {processed_batches}: {e}")

            processed_batches += 1

            # Progress reporting
            current_time = time.time()
            if current_time - last_progress_time >= progress_interval:
                percent_done = (i + 1) / total_groups * 100
                logger.info(f"  Progress: {i + 1}/{total_groups} groups ({percent_done:.1f}%) | {processed_batches} batches")
                last_progress_time = current_time

            # Reset batch
            group_batch = []
            group_batch_keys = []
            group_batch_images = []
            group_image_counts = []

    logger.info(f"✓ OCR complete for {camera}/{video_stem}: {len(video_results['letters'])} results from {processed_batches} batches")

    # Aggregate per video via majority voting
    if not video_results['letters']:
        return {}

    letters_counter = Counter(video_results['letters'])
    numbers_counter = Counter(video_results['numbers'])

    final_letters = letters_counter.most_common(1)[0][0]
    final_numbers = numbers_counter.most_common(1)[0][0]
    final_tag = f"{final_letters}{final_numbers}"

    return {
        'final_tag': final_tag,
        'letters': final_letters,
        'numbers': final_numbers,
        'detection_count': len(video_results['frames']),
        'letters_confidence': float(np.mean(video_results['letters_confs'])),
        'numbers_confidence': float(np.mean(video_results['numbers_confs'])),
    }


def _run_step3_multi_gpu(
    config: PipelineConfig,
    camera: str,
    video_stem: str,
    crop_paths: List[str],
    model_path: str
) -> Dict:
    """
    Process one event's tag groups using multiple GPUs with multiprocessing.

    Args:
        config: Pipeline configuration
        camera: Camera name for this event
        video_stem: Video stem for this event
        crop_paths: List of crop file paths (only from this event)
        model_path: Path to text recognition model

    Returns:
        Dict with keys: 'final_tag', 'letters', 'numbers', 'detection_count',
                        'letters_confidence', 'numbers_confidence'
    """
    tag_groups = group_tag_crops_by_ring_for_event(crop_paths)
    total_groups = len(tag_groups)

    if total_groups == 0:
        logger.info(f"No tag groups found for {camera}/{video_stem}")
        return {}

    logger.info(f"Processing {total_groups} tag groups for {camera}/{video_stem} with multi-GPU")

    video_results = {
        'frames': [],
        'letters': [],
        'numbers': [],
        'letters_confs': [],
        'numbers_confs': [],
    }

    batch_size = config.step3.batch_size
    num_gpus = len(config.step3.gpu_devices)

    logger.info(f"Starting multi-GPU processing with {num_gpus} GPUs (batch_size={batch_size})")

    with Manager() as manager:
        # Queue sizes based on groups in this event (not millions)
        queue_size = max(64, total_groups // 5)
        batch_queue = manager.Queue(maxsize=queue_size)
        result_queue = manager.Queue(maxsize=queue_size)

        ctx = get_context('spawn')
        gpu_pool = ctx.Pool(num_gpus)

        try:
            # Start GPU workers
            gpu_workers = []
            for gpu_id in config.step3.gpu_devices:
                task = gpu_pool.apply_async(
                    gpu_text_recognition_worker,
                    (gpu_id, batch_queue, result_queue, model_path, logger)
                )
                gpu_workers.append(task)

            logger.info(f"Started {num_gpus} GPU workers for {camera}/{video_stem}")

            # Producer: Accumulate and submit batches
            batch_idx = 0
            group_batch = []
            group_batch_keys = []
            group_image_counts = []

            for i, (group_key, group_crop_paths) in enumerate(tag_groups.items()):
                frame_id = group_key[2]

                # Load images for this group
                images = []
                try:
                    for crop_path in group_crop_paths:
                        img = cv.imread(str(crop_path))
                        if img is not None:
                            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                            img_pil = Image.fromarray(img_rgb)
                            images.append(img_pil)
                except Exception as e:
                    logger.debug(f"Error loading crops for {frame_id}: {e}")
                    continue

                if not images:
                    continue

                # Accumulate
                group_batch.append(images)
                group_batch_keys.append(group_key)
                group_image_counts.append(len(images))

                # Submit batch when ready or at end
                should_submit = (len(group_batch) >= batch_size) or (i == total_groups - 1)
                if should_submit:
                    batch_queue.put((batch_idx, group_batch_keys, group_batch, group_image_counts))
                    batch_idx += 1
                    group_batch = []
                    group_batch_keys = []
                    group_image_counts = []

            # Send sentinels
            logger.info("All batches submitted, sending sentinels to GPU workers")
            for _ in range(num_gpus):
                batch_queue.put(None)

            # Consumer: Collect results
            sentinel_count = 0
            results_received = {}
            last_result_time = time.time()
            result_interval = 5.0
            timeout_count = 0
            max_consecutive_timeouts = 120

            logger.info("Collecting results from GPU workers")
            while sentinel_count < num_gpus:
                try:
                    result = result_queue.get(timeout=10.0)
                    timeout_count = 0

                    if result is None:
                        sentinel_count += 1
                        logger.info(f"✓ Sentinel {sentinel_count}/{num_gpus} received")
                        continue

                    if result['status'] == 'success':
                        results_received[result['batch_idx']] = result
                        timeout_count = 0

                        # Progress reporting
                        current_time = time.time()
                        if current_time - last_result_time >= result_interval:
                            if batch_idx > 0:
                                percent_done = len(results_received) / batch_idx * 100
                            else:
                                percent_done = 0
                            logger.info(f"  Results: {len(results_received)}/{batch_idx} batches ({percent_done:.1f}%)")
                            last_result_time = current_time

                    else:
                        logger.error(f"Batch {result['batch_idx']} failed: {result.get('error')}")
                        timeout_count = 0

                except Exception as e:
                    timeout_count += 1
                    if timeout_count % 10 == 0:
                        logger.info(f"  Waiting for results ({timeout_count}x timeout, {len(results_received)} results so far)...")
                    if timeout_count > max_consecutive_timeouts:
                        logger.error(f"Too many consecutive timeouts ({timeout_count}), assuming workers are stuck")
                        raise

            logger.info(f"✓ Collected results from {len(results_received)} batches")

            # Process results in order
            for batch_idx_key in sorted(results_received.keys()):
                result = results_received[batch_idx_key]
                group_keys = result['group_keys']
                group_image_counts = result['group_image_counts']
                tag_predictions = result['predictions']

                # Map predictions back to groups, combining letters+numbers per ring
                pred_idx = 0
                for group_idx, group_key in enumerate(group_keys):
                    cam, vstem, fid = group_key
                    images_in_group = group_image_counts[group_idx]

                    # Collect all predictions for this group
                    group_predictions = {}
                    for img_idx in range(images_in_group):
                        pred_dict = tag_predictions[pred_idx]
                        pred_idx += 1

                        # Collect both letters and numbers from this group
                        if pred_dict.get('letters'):
                            group_predictions['letters'] = pred_dict['letters']
                        if pred_dict.get('numbers'):
                            group_predictions['numbers'] = pred_dict['numbers']

                    # Only add result if we have BOTH letters and numbers for this ring
                    if group_predictions.get('letters') and group_predictions.get('numbers'):
                        letters = group_predictions['letters']['str']
                        numbers = group_predictions['numbers']['str']
                        letters_conf = float(np.mean(group_predictions['letters']['pred']))
                        numbers_conf = float(np.mean(group_predictions['numbers']['pred']))

                        video_results['frames'].append(fid)
                        video_results['letters'].append(letters)
                        video_results['numbers'].append(numbers)
                        video_results['letters_confs'].append(letters_conf)
                        video_results['numbers_confs'].append(numbers_conf)

            # Wait for GPU workers to finish
            logger.info("Waiting for GPU workers to complete")
            for task in gpu_workers:
                task.get(timeout=3600)

            logger.info(f"✓ OCR complete for {camera}/{video_stem}: {len(video_results['letters'])} results")

        finally:
            gpu_pool.close()
            gpu_pool.join()

    # Aggregate per video via majority voting
    if not video_results['letters']:
        return {}

    letters_counter = Counter(video_results['letters'])
    numbers_counter = Counter(video_results['numbers'])

    final_letters = letters_counter.most_common(1)[0][0]
    final_numbers = numbers_counter.most_common(1)[0][0]
    final_tag = f"{final_letters}{final_numbers}"

    return {
        'final_tag': final_tag,
        'letters': final_letters,
        'numbers': final_numbers,
        'detection_count': len(video_results['frames']),
        'letters_confidence': float(np.mean(video_results['letters_confs'])),
        'numbers_confidence': float(np.mean(video_results['numbers_confs'])),
    }


def run_step3_text_recognition(config: PipelineConfig, db_logger: PipelineLogger, force_reprocess: bool = False) -> bool:
    """
    Execute Step 3: Text Recognition on all tag crops with EVENT-BY-EVENT processing.

    This processes tag crops one event (camera/video_stem folder) at a time, keeping
    memory bounded (~500MB per event, not 30GB). Each event is logged to the database
    immediately after completion, enabling resume support.

    Args:
        config: Pipeline configuration
        db_logger: Database logger instance
        force_reprocess: If True, ignore resume and reprocess all completed videos

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("Starting Step 3: Text Recognition (EVENT-BY-EVENT PROCESSING)")
    logger.info("=" * 80)
    step_start_time = time.time()
    total_events_processed = 0
    total_events_failed = 0

    try:
        num_gpus = len(config.step3.gpu_devices)
        use_multi_gpu = num_gpus > 1
        logger.info(f"Using {num_gpus} GPU(s), multi_gpu={use_multi_gpu}")

        # Load model once (reuse across all events)
        try:
            if use_multi_gpu:
                model = None
                model_path = str(config.models.text_recognition.get_model_path())
            else:
                logger.info("Loading TextRecognition model for single-GPU mode...")
                model = TextRecognition(
                    config.models.text_recognition.get_model_path(),
                    device="cuda:0"
                )
                logger.info("✓ Model loaded")
                model_path = None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

        # Get completed videos for resume
        completed_videos = set()
        if not force_reprocess:
            completed_videos = set(db_logger.get_completed_videos(step="step3"))
            logger.info(f"Resuming: {len(completed_videos)} videos already completed in step3")
        else:
            logger.info("Force reprocess: Reprocessing all videos")

        # Iterate through each camera
        for camera in config.input.cameras:
            logger.info(f"\nProcessing camera: {camera}")

            # Discover events for this camera
            events = discover_tag_crop_events(config.output_root_path, camera)

            if not events:
                logger.info(f"No tag crops found for camera {camera}")
                continue

            logger.info(f"Found {len(events)} events for camera {camera}")

            # Process each event independently
            for event_idx, (video_stem, crop_paths) in enumerate(events):
                # Check if already completed
                if video_stem in completed_videos:
                    logger.info(f"[{event_idx + 1}/{len(events)}] Skipping completed: {camera}/{video_stem}")
                    continue

                event_start_time = time.time()
                logger.info(f"[{event_idx + 1}/{len(events)}] Processing: {camera}/{video_stem} ({len(crop_paths)} crops)")

                try:
                    # Process this event
                    if use_multi_gpu:
                        video_result = _run_step3_multi_gpu(
                            config, camera, video_stem, crop_paths, model_path
                        )
                    else:
                        video_result = _run_step3_single_gpu(
                            config, camera, video_stem, crop_paths, model
                        )

                    # Handle case where no results were produced
                    if not video_result:
                        logger.warning(f"No OCR results for {camera}/{video_stem}")
                        db_logger.log_step3_result(video_stem, camera, {
                            'status': 'success',
                            'processing_seconds': time.time() - event_start_time,
                            'tags_processed': 0,
                            'ocr_matches_count': 0,
                            'ocr_success_rate': 0.0,
                            'letters_conf_min_mean': 0.0,
                            'numbers_conf_min_mean': 0.0,
                            'output_path': '',
                        })
                        total_events_processed += 1
                        continue

                    # Save results for this video
                    try:
                        output_dir = get_video_output_dir(config.output_root_path, camera, video_stem, step=3)
                        output_dir.mkdir(parents=True, exist_ok=True)

                        summary_path = output_dir / f"{video_stem}.yaml"
                        with open(summary_path, 'w') as f:
                            yaml.dump({
                                'video_stem': video_stem,
                                'final_tag': video_result['final_tag'],
                                'letters': video_result['letters'],
                                'numbers': video_result['numbers'],
                                'detection_count': video_result['detection_count'],
                                'letters_confidence': video_result['letters_confidence'],
                                'numbers_confidence': video_result['numbers_confidence'],
                            }, f)

                        logger.info(f"✓ Saved: {video_stem} → {video_result['final_tag']}")

                    except Exception as e:
                        logger.error(f"Error saving results for {video_stem}: {e}")
                        db_logger.log_step3_result(video_stem, camera, {
                            'status': 'error',
                            'error_message': f"Save error: {str(e)}",
                        })
                        total_events_failed += 1
                        continue

                    # Log to database IMMEDIATELY (enables resume)
                    elapsed = time.time() - event_start_time
                    try:
                        db_logger.log_step3_result(video_stem, camera, {
                            'status': 'success',
                            'processing_seconds': elapsed,
                            'tags_processed': video_result['detection_count'],
                            'ocr_matches_count': 1,
                            'ocr_success_rate': 1.0,
                            'letters_conf_min_mean': video_result['letters_confidence'],
                            'numbers_conf_min_mean': video_result['numbers_confidence'],
                            'letters_ema_min_mean': video_result['letters_confidence'],
                            'numbers_ema_min_mean': video_result['numbers_confidence'],
                            'output_path': str(summary_path),
                        })
                        logger.info(f"✓ Logged to database: {camera}/{video_stem}")
                    except Exception as e:
                        logger.error(f"Error logging to database for {video_stem}: {e}")
                        total_events_failed += 1
                        continue

                    total_events_processed += 1

                except Exception as e:
                    logger.error(f"✗ Failed to process {camera}/{video_stem}: {e}")
                    logger.error(traceback.format_exc())

                    # Log error to database
                    try:
                        db_logger.log_step3_result(video_stem, camera, {
                            'status': 'error',
                            'error_message': str(e),
                            'processing_seconds': time.time() - event_start_time,
                        })
                    except Exception as log_e:
                        logger.error(f"Failed to log error to database: {log_e}")

                    total_events_failed += 1

        elapsed = time.time() - step_start_time
        logger.info("\n" + "=" * 80)
        logger.info(f"✓ Step 3 completed in {elapsed:.1f} seconds")
        logger.info(f"  Events processed: {total_events_processed}")
        logger.info(f"  Events failed: {total_events_failed}")
        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        logger.error(traceback.format_exc())
        return False
