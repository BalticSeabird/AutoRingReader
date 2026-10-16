#!/usr/bin/env python3
"""Parallel runner for Step 2 only, using GPU 1."""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

from pipeline_config import PipelineConfig
from pipeline_logger import PipelineLogger
from pipeline_utils import ensure_output_dir


def setup_logging() -> logging.Logger:
    """Set up logging for the runner."""
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = log_dir / f"{timestamp}_step2_parallel.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    return logging.getLogger(__name__)


def main(batch_yaml_path: str, gpu_id: int = 1, force_reprocess: bool = False):
    """Run Step 2 only."""
    logger = setup_logging()

    try:
        logger.info("Starting Step 2 (Tag Detection) runner")
        logger.info(f"Batch YAML: {batch_yaml_path}")
        logger.info("Console log: log/*_step2_parallel.txt")
        logger.info(f"Requested GPU: {gpu_id}")
        if force_reprocess:
            logger.info("⚠️  FORCE REPROCESS MODE ENABLED")

        # Load configuration
        if not Path(batch_yaml_path).exists():
            logger.error(f"Batch YAML not found: {batch_yaml_path}")
            return 1

        config = PipelineConfig.from_yaml(batch_yaml_path)
        logger.info(f"Output root: {config.output_root_path}")
        logger.info(f"Cameras: {config.input.cameras}")

        # Override GPU device for Step 2
        config.step2.gpu_devices = [gpu_id]
        logger.info(f"Step 2 GPU devices set to: {config.step2.gpu_devices}")
        logger.info(
            f"Step 2 workers: CPU={config.step2.n_processes_cpu}, GPU={config.step2.n_processes_gpu}, batch_size={config.step2.batch_size}"
        )

        # Ensure output directories
        ensure_output_dir(config.output_root_path)
        logger.info("Output directory checked/created")

        # Initialize database logger
        db_path = config.get_db_path()
        logger.info(f"Database path: {db_path}")
        pipeline_logger = PipelineLogger(str(db_path), logger)

        # Run Step 2
        from step2_tag_detection import run_step2_tag_detection

        logger.info("Launching Step 2 processing")
        success = run_step2_tag_detection(config, pipeline_logger, force_reprocess=force_reprocess)

        pipeline_logger.close()

        if success:
            logger.info("✓ Step 2 completed successfully")
            return 0
        else:
            logger.error("✗ Step 2 failed")
            return 1

    except Exception as e:
        logger.error(f"Step 2 parallel runner failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Step 2 (Tag Detection) in parallel")
    parser.add_argument("batch_yaml", help="Path to batch configuration YAML file")
    parser.add_argument("--gpu", type=int, default=1, help="GPU device ID (default: 1)")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all videos")
    args = parser.parse_args()

    exit_code = main(args.batch_yaml, gpu_id=args.gpu, force_reprocess=args.force)
    sys.exit(exit_code)
