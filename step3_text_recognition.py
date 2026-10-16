"""
Step 3: Text Recognition - Recognize ring IDs from tag crops.

This step:
1. Groups tag crop images by ring (tag_0, tag_1 together)
2. Uses CLIP4STR model for OCR character recognition
3. Applies pattern matching and majority voting
4. Aggregates to final ring ID per video
5. Logs results to database and saves YAML output

Uses batched processing and optional multi-GPU support via multiprocessing.
"""

import logging
import yaml
import time
import sys
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Manager, get_context, Process
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2 as cv
from PIL import Image
import traceback

from pipeline_config import PipelineConfig
from pipeline_logger import PipelineLogger
from pipeline_utils import get_video_output_dir
from text_recognition import TextRecognition


logger = logging.getLogger(__name__)


def group_tag_crops_by_ring(output_root: str) -> Dict[Tuple[str, str, str], List[str]]:
    """
    Group tag crops by ring (camera, video_stem, frame_id).

    File structure: step2_tag_crops/{camera}/{video_stem}/{frame_id}_ring_tag_{idx}.jpg
    Groups files by (camera, video_stem, frame_id) so tag_0, tag_1 are together.

    Args:
        output_root: Root output directory

    Returns:
        Dict mapping (camera, video_stem, frame_id) to list of sorted crop paths
    """
    tag_dir = Path(output_root) / "step2_tag_crops"
    groups = defaultdict(list)

    if not tag_dir.exists():
        return groups

    for crop_file in sorted(tag_dir.rglob("*.jpg")):
        parts = crop_file.parts
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
                    # Send sentinel to result_queue to signal main process
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


def _run_step3_single_gpu(config: PipelineConfig, db_logger: PipelineLogger,
                         tag_groups: Dict, model: TextRecognition) -> Tuple[List[Dict], Dict]:
    """
    Process tag groups using single GPU with batched inference.

    Returns:
        Tuple of (all_results, video_results)
    """
    total_groups = len(tag_groups)
    all_results = []
    video_results = defaultdict(lambda: {
        'frames': [],
        'letters': [],
        'numbers': [],
        'letters_confs': [],
        'numbers_confs': [],
    })

    batch_size = 32
    group_batch = []
    group_batch_keys = []
    group_batch_images = []
    group_image_counts = []

    processed_batches = 0
    last_progress_time = time.time()
    progress_interval = 5.0

    for i, (group_key, crop_paths) in enumerate(tag_groups.items()):
        camera, video_stem, frame_id = group_key

        # Load images for this group
        images = []
        try:
            for crop_path in crop_paths:
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
                    camera, video_stem, frame_id = group_key
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

                        all_results.append({
                            'group_key': group_key,
                            'camera': camera,
                            'video_stem': video_stem,
                            'frame_id': frame_id,
                            'letters': letters,
                            'numbers': numbers,
                            'letters_conf': letters_conf,
                            'numbers_conf': numbers_conf,
                        })

                        video_results[video_stem]['frames'].append(frame_id)
                        video_results[video_stem]['letters'].append(letters)
                        video_results[video_stem]['numbers'].append(numbers)
                        video_results[video_stem]['letters_confs'].append(letters_conf)
                        video_results[video_stem]['numbers_confs'].append(numbers_conf)

            except Exception as e:
                logger.error(f"Batch OCR failed for batch {processed_batches}: {e}")

            processed_batches += 1

            # Progress reporting
            current_time = time.time()
            if current_time - last_progress_time >= progress_interval:
                percent_done = (i + 1) / total_groups * 100
                logger.info(f"Progress: {i + 1}/{total_groups} groups ({percent_done:.1f}%) | {processed_batches} batches")
                last_progress_time = current_time

            # Reset batch
            group_batch = []
            group_batch_keys = []
            group_batch_images = []
            group_image_counts = []

    logger.info(f"✓ OCR complete with single GPU: {len(all_results)} results from {processed_batches} batches")
    return all_results, dict(video_results)


def _run_step3_multi_gpu(config: PipelineConfig, db_logger: PipelineLogger,
                        tag_groups: Dict, model_path: str) -> Tuple[List[Dict], Dict]:
    """
    Process tag groups using multiple GPUs with multiprocessing.

    Returns:
        Tuple of (all_results, video_results)
    """
    total_groups = len(tag_groups)
    all_results = []
    video_results = defaultdict(lambda: {
        'frames': [],
        'letters': [],
        'numbers': [],
        'letters_confs': [],
        'numbers_confs': [],
    })

    batch_size = 32
    num_gpus = len(config.step3.gpu_devices)

    logger.info(f"Starting multi-GPU processing with {num_gpus} GPUs")

    with Manager() as manager:
        # Larger queue sizes to avoid producer blocking
        # With 10K+ groups, need room for many batches in queue
        queue_size = max(256, len(tag_groups) // 10)  # Generous buffer
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

            logger.info(f"Started {num_gpus} GPU workers")

            # Producer: Accumulate and submit batches
            batch_idx = 0
            group_batch = []
            group_batch_keys = []
            group_image_counts = []

            for i, (group_key, crop_paths) in enumerate(tag_groups.items()):
                camera, video_stem, frame_id = group_key

                # Load images for this group
                images = []
                try:
                    for crop_path in crop_paths:
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
            max_consecutive_timeouts = 120  # 20 minutes of timeouts = stuck

            logger.info("Collecting results from GPU workers")
            while sentinel_count < num_gpus:
                try:
                    result = result_queue.get(timeout=10.0)
                    timeout_count = 0  # Reset timeout counter on success

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
                            logger.info(f"Results: {len(results_received)}/{batch_idx} batches ({percent_done:.1f}%)")
                            last_result_time = current_time

                    else:
                        logger.error(f"Batch {result['batch_idx']} failed: {result.get('error')}")
                        timeout_count = 0

                except Exception as e:
                    timeout_count += 1
                    if timeout_count % 10 == 0:  # Log every 10 timeouts (100 seconds)
                        logger.info(f"Waiting for results ({timeout_count}x timeout, {len(results_received)} results so far)...")
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
                    camera, video_stem, frame_id = group_key
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

                        all_results.append({
                            'group_key': group_key,
                            'camera': camera,
                            'video_stem': video_stem,
                            'frame_id': frame_id,
                            'letters': letters,
                            'numbers': numbers,
                            'letters_conf': letters_conf,
                            'numbers_conf': numbers_conf,
                        })

                        video_results[video_stem]['frames'].append(frame_id)
                        video_results[video_stem]['letters'].append(letters)
                        video_results[video_stem]['numbers'].append(numbers)
                        video_results[video_stem]['letters_confs'].append(letters_conf)
                        video_results[video_stem]['numbers_confs'].append(numbers_conf)

            # Wait for GPU workers to finish
            logger.info("Waiting for GPU workers to complete")
            for task in gpu_workers:
                task.get(timeout=3600)

            logger.info(f"✓ OCR complete with {num_gpus} GPUs: {len(all_results)} results")
            return all_results, dict(video_results)

        finally:
            gpu_pool.close()
            gpu_pool.join()


def run_step3_text_recognition(config: PipelineConfig, db_logger: PipelineLogger, force_reprocess: bool = False) -> bool:
    """
    Execute Step 3: Text Recognition on all tag crops with batched OCR processing.

    Supports both single-GPU (faster for setup) and multi-GPU modes.
    Uses batches of 32 tag groups for optimized inference.

    Args:
        config: Pipeline configuration
        db_logger: Database logger instance
        force_reprocess: If True, ignore resume and reprocess all completed videos

    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting Step 3: Text Recognition")
    step_start_time = time.time()

    try:
        # Group crops by ring tags
        tag_groups = group_tag_crops_by_ring(config.output_root_path)

        if not tag_groups:
            logger.warning("No tag crop groups found from Step 2")
            return True

        total_groups = len(tag_groups)
        num_gpus = len(config.step3.gpu_devices)
        logger.info(f"Processing {total_groups} ring tag groups with {num_gpus} GPU(s)")

        # Decide whether to use single or multi-GPU
        use_multi_gpu = num_gpus > 1

        if use_multi_gpu:
            logger.info("Using multi-GPU mode with Manager and Pools")
            try:
                model_path = str(config.models.text_recognition.get_model_path())
                all_results, video_results = _run_step3_multi_gpu(config, db_logger, tag_groups, model_path)
            except Exception as e:
                logger.warning(f"Multi-GPU mode failed, falling back to single-GPU: {e}")
                logger.info("Loading model for single-GPU fallback...")
                try:
                    model = TextRecognition(
                        config.models.text_recognition.get_model_path(),
                        device="cuda:0"
                    )
                    all_results, video_results = _run_step3_single_gpu(config, db_logger, tag_groups, model)
                except Exception as e2:
                    logger.error(f"Single-GPU fallback also failed: {e2}")
                    return False
        else:
            logger.info("Using single-GPU mode with batched inference")
            try:
                model = TextRecognition(
                    config.models.text_recognition.get_model_path(),
                    device="cuda:0"
                )
                logger.info("✓ Model loaded")
                all_results, video_results = _run_step3_single_gpu(config, db_logger, tag_groups, model)
            except Exception as e:
                logger.error(f"Single-GPU mode failed: {e}")
                return False

        # Aggregate per video via majority voting
        final_tags = {}
        for video_stem, data in video_results.items():
            if not data['letters']:
                continue

            letters_counter = Counter(data['letters'])
            numbers_counter = Counter(data['numbers'])

            final_letters = letters_counter.most_common(1)[0][0]
            final_numbers = numbers_counter.most_common(1)[0][0]
            final_tag = f"{final_letters}{final_numbers}"

            final_tags[video_stem] = {
                'tag': final_tag,
                'letters': final_letters,
                'numbers': final_numbers,
                'detection_count': len(data['frames']),
                'letters_confidence': float(np.mean(data['letters_confs'])),
                'numbers_confidence': float(np.mean(data['numbers_confs'])),
            }

        # Save results
        for video_stem, tag_data in final_tags.items():
            try:
                # Extract camera from all_results for this video
                camera = "UNKNOWN"
                for result in all_results:
                    if result['video_stem'] == video_stem:
                        camera = result['camera']
                        break

                output_dir = get_video_output_dir(config.output_root_path, camera, video_stem, step=3)

                summary_path = output_dir / f"{video_stem}.yaml"
                with open(summary_path, 'w') as f:
                    yaml.dump({
                        'video_stem': video_stem,
                        'final_tag': tag_data['tag'],
                        'letters': tag_data['letters'],
                        'numbers': tag_data['numbers'],
                        'detection_count': tag_data['detection_count'],
                        'letters_confidence': tag_data['letters_confidence'],
                        'numbers_confidence': tag_data['numbers_confidence'],
                    }, f)

                # Log to database if available
                if db_logger:
                    db_stats = {
                        'status': 'success',
                        'processing_seconds': 0,
                        'tags_processed': tag_data['detection_count'],
                        'ocr_matches_count': 1,
                        'ocr_success_rate': 1.0,
                        'letters_conf_min_mean': tag_data['letters_confidence'],
                        'numbers_conf_min_mean': tag_data['numbers_confidence'],
                        'letters_ema_min_mean': tag_data['letters_confidence'],
                        'numbers_ema_min_mean': tag_data['numbers_confidence'],
                        'output_path': str(summary_path),
                    }
                    db_logger.log_step3_result(video_stem, camera, db_stats)

                logger.info(f"✓ Saved OCR result: {video_stem} → {tag_data['tag']} (camera: {camera})")

            except Exception as e:
                logger.error(f"Error saving OCR results for {video_stem}: {e}")

        elapsed = time.time() - step_start_time
        logger.info(f"✓ Step 3 completed in {elapsed:.1f} seconds ({len(final_tags)} videos processed)")
        return True

    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        logger.error(traceback.format_exc())
        return False
