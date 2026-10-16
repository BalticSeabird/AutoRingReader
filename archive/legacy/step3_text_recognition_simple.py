#!/usr/bin/env python3
"""
Step 3 Alternative: TEXT RECOGNITION - Simpler thread-based implementation.
Avoids complex multiprocessing synchronization issues.
"""

import logging
import yaml
import time
from pathlib import Path
from threading import Thread
from queue import Queue, Empty
from collections import defaultdict
from typing import List, Dict, Tuple, Any
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
    """Group tag crops by ring (camera, video_stem, frame_id)."""
    tag_dir =Path(output_root) / "step2_tag_crops"
    groups = defaultdict(list)

    if not tag_dir.exists():
        return groups

    for crop_file in sorted(tag_dir.rglob("*.jpg")):
        parts = crop_file.parts
        camera = parts[-3] if len(parts) >= 3 else "UNKNOWN"
        video_stem = parts[-2] if len(parts) >= 2 else "unknown"

        stem = crop_file.stem
        if "_ring_tag_" in stem:
            frame_id = stem.split("_ring_tag_")[0]
            key = (camera, video_stem, frame_id)
            groups[key].append(str(crop_file))

    for key in groups:
        groups[key] = sorted(groups[key])

    return dict(groups)


def run_step3_text_recognition_simple(config: PipelineConfig, db_logger: PipelineLogger, force_reprocess: bool = False) -> bool:
    """
    Step 3 using simple threading (avoids multiprocessing complexities).
    Process all crops sequentially on GPU using text recognition model.
    """
    logger.info("Starting Step 3: Text Recognition (threaded)")
    step_start_time = time.time()

    try:
        #Group crops by ring
        tag_groups = group_tag_crops_by_ring(config.output_root_path)

        if not tag_groups:
            logger.warning("No tag crop groups found from Step 2")
            return True

        total_groups = len(tag_groups)
        logger.info(f"Processing {total_groups} ring tag groups")

        # Load model once
        logger.info("Loading TextRecognition model...")
        try:
            model = TextRecognition(
                config.models.text_recognition.get_model_path(),
                device="cuda:0"
            )
            logger.info("✓ Model loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

        # Process groups
        all_results = []
        video_results = defaultdict(lambda: {'frames': [], 'letters': [], 'numbers': [], 'letters_confs': [], 'numbers_confs': []})

        for i, (group_key, crop_paths) in enumerate(tag_groups.items()):
            camera, video_stem, frame_id = group_key

            # Load images
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

            # Run OCR
            try:
                tag_dict = model.predict_tag(images)

                if tag_dict.get('letters') and tag_dict.get('numbers'):
                    letters = tag_dict['letters']['str']
                    numbers = tag_dict['numbers']['str']
                    letters_conf = float(np.mean(tag_dict['letters']['pred']))
                    numbers_conf = float(np.mean(tag_dict['numbers']['pred']))

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
                logger.debug(f"OCR failed for {frame_id}: {e}")

            if (i + 1) % 200 == 0:
                logger.info(f"Progress: {i + 1}/{total_groups} groups ({(i+1)*100//total_groups}%)")

        logger.info(f"✓ OCR complete, got {len(all_results)} results")

        # Aggregate per video
        final_tags = {}
        from collections import Counter
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
                camera = "UNKNOWN"
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

                logger.info(f"✓ {video_stem} → {tag_data['tag']}")

            except Exception as e:
                logger.error(f"Error saving {video_stem}: {e}")

        elapsed = time.time() - step_start_time
        logger.info(f"✓ Step 3 completed in {elapsed:.1f} seconds ({len(final_tags)} videos processed)")
        return True

    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        logger.error(traceback.format_exc())
        return False
