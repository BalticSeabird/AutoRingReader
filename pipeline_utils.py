"""Shared utilities for step-based processing pipeline."""

import os
import logging
from pathlib import Path
from typing import List, Iterator, Tuple, Generator, Any, Optional
import numpy as np
import torch
import cv2 as cv
from contextlib import contextmanager
from multiprocessing import Queue

from pipeline_config import PipelineConfig


def get_video_list(config: PipelineConfig, step: int = 1,
                   completed_videos: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Recursively find all video files matching camera filters.

    Args:
        config: Pipeline configuration
        step: Which step number to check for resume (1, 2, or 3)
        completed_videos: List of already-completed video paths for resume

    Returns:
        List of (video_path, camera) tuples
    """
    logger = logging.getLogger(__name__)
    video_root = Path(config.input.video_root_path)

    if not video_root.exists():
        logger.error(f"Video root path does not exist: {video_root}")
        return []

    video_files = []

    # Supported video formats
    video_extensions = {'.mp4', '.avi', '.ts', '.mov', '.flv', '.mkv', '.webm'}

    # Find all video files
    for video_path in video_root.rglob('*'):
        if video_path.suffix.lower() in video_extensions:
            # Check if camera matches
            camera_match = None
            for camera in config.input.cameras:
                if camera.lower() in video_path.name.lower():
                    camera_match = camera
                    break

            if camera_match:
                video_files.append((str(video_path), camera_match))

    logger.info(f"Found {len(video_files)} videos matching cameras: {config.input.cameras}")

    # Filter by completed videos if resume is enabled
    if config.input.resume and completed_videos:
        completed_set = set(completed_videos)
        video_files = [(v, c) for v, c in video_files if v not in completed_set]
        logger.info(f"After resume filtering: {len(video_files)} videos pending for step {step}")

    return video_files


def batch_items(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """
    Yield batches of items from a list.

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def batch_frames_from_queue(frame_queue: Queue, batch_size: int,
                            queue_timeout: float = 1.0) -> Generator[List[Tuple[Any, Any]], None, None]:
    """
    Yield batches of frames from a queue without blocking indefinitely.

    Args:
        frame_queue: Queue containing frames
        batch_size: Batch size
        queue_timeout: Timeout for queue.get()

    Yields:
        Batches of frames
    """
    batch = []
    while True:
        try:
            item = frame_queue.get(timeout=queue_timeout)
            if item is None:  # Sentinel value indicating end of stream
                if batch:
                    yield batch
                break
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        except:
            # Timeout - yield what we have if non-empty
            if batch:
                yield batch
                batch = []


@contextmanager
def cuda_device_context(device_id: int):
    """
    Context manager for CUDA device selection.

    Args:
        device_id: CUDA device ID (e.g., 0, 1)
    """
    if torch.cuda.is_available():
        original_device = torch.cuda.current_device()
        torch.cuda.set_device(device_id)
        try:
            yield
        finally:
            torch.cuda.set_device(original_device)
    else:
        yield


def get_available_gpus() -> List[int]:
    """Get list of available GPU device IDs."""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def load_image_batch(image_paths: List[str], target_size: Optional[Tuple[int, int]] = None,
                     normalize: bool = True) -> np.ndarray:
    """
    Load a batch of images from paths.

    Args:
        image_paths: List of image file paths
        target_size: Optional (height, width) to resize
        normalize: Whether to normalize to [0, 1]

    Returns:
        Numpy array of shape (batch_size, height, width, 3)
    """
    images = []
    for path in image_paths:
        img = cv.imread(str(path))
        if img is None:
            logging.warning(f"Failed to load image: {path}")
            if target_size:
                img = np.zeros((*target_size, 3), dtype=np.uint8)
            else:
                img = np.zeros((224, 224, 3), dtype=np.uint8)

        if target_size:
            img = cv.resize(img, (target_size[1], target_size[0]))

        # Convert BGR to RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if normalize:
            img = img.astype(np.float32) / 255.0

        images.append(img)

    return np.array(images)


def sample_frames_from_video(video_path: str, sample_rate: int = 1,
                             max_frames: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yield frames from video file with optional sampling.

    Args:
        video_path: Path to video file
        sample_rate: Yield every Nth frame (1 = every frame, 2 = every other frame, etc)
        max_frames: Maximum frames to yield (None = all)

    Yields:
        (frame_index, frame_array) tuples
    """
    logger = logging.getLogger(__name__)

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    frame_idx = 0
    yielded_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames based on rate
            if frame_idx % sample_rate == 0:
                yield frame_idx, frame
                yielded_count += 1

                if max_frames and yielded_count >= max_frames:
                    break

            frame_idx += 1
    finally:
        cap.release()

    logger.debug(f"Read {frame_idx} frames from {video_path}, yielded {yielded_count}")


def get_video_info(video_path: str) -> dict:
    """
    Get metadata about a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with keys: frame_count, width, height, fps, duration_seconds
    """
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    try:
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        return {
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'fps': fps,
            'duration_seconds': duration,
        }
    finally:
        cap.release()


def ensure_output_dir(output_root: str) -> Path:
    """Ensure output directory exists."""
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def ensure_step_output_dir(output_root: str, step: int) -> Path:
    """Ensure step-specific output directory exists."""
    step_names = {
        1: "step1_ring_crops",
        2: "step2_tag_crops",
        3: "step3_ocr_results",
    }

    step_dir = Path(output_root) / step_names[step]
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir


def get_camera_output_dir(output_root: str, camera: str, step: int) -> Path:
    """Get output directory for a specific camera and step."""
    step_dir = ensure_step_output_dir(output_root, step)
    camera_dir = step_dir / camera
    camera_dir.mkdir(parents=True, exist_ok=True)
    return camera_dir


def get_video_output_dir(output_root: str, camera: str, video_stem: str, step: int) -> Path:
    """Get output directory for a specific video at a step."""
    camera_dir = get_camera_output_dir(output_root, camera, step)
    video_dir = camera_dir / video_stem
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir
