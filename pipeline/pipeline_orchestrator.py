"""Main orchestrator for the step-based processing pipeline."""


import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from time import time

from .pipeline_config import PipelineConfig
from .pipeline_logger import PipelineLogger
from .pipeline_diagnostics import PipelineDiagnostics
from .pipeline_utils import ensure_output_dir, get_video_list, get_available_gpus


def setup_logging() -> logging.Logger:
    """Set up logging for the orchestrator."""
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = log_dir / f"{timestamp}_orchestrator.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )

    return logging.getLogger(__name__)


def print_config_summary(config: PipelineConfig, logger: logging.Logger):
    """Print a summary of the configuration."""
    logger.info("=" * 80)
    logger.info("PIPELINE CONFIGURATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Input Videos: {config.input.video_root_path}")
    logger.info(f"Cameras: {config.input.cameras}")
    logger.info(f"Output Root: {config.output_root_path}")
    logger.info(f"ROI Groups Loaded: {len(config.roi_config.rois_data) if config.roi_config else 0}")
    logger.info(f"Resume Enabled: {config.input.resume}")
    logger.info("")
    logger.info(f"Step 1 (Ring Detection):")
    logger.info(f"  - CPU workers: {config.step1.n_processes_cpu}")
    logger.info(f"  - GPU workers: {config.step1.n_processes_gpu} on devices {config.step1.gpu_devices}")
    logger.info(f"  - Batch size: {config.step1.batch_size}, Frame sample rate: {config.step1.frame_sample_rate}")
    logger.info(f"  - Confidence threshold: {config.step1.conf_threshold}")
    logger.info("")
    logger.info(f"Step 2 (Tag Detection):")
    logger.info(f"  - CPU workers: {config.step2.n_processes_cpu}")
    logger.info(f"  - GPU workers: {config.step2.n_processes_gpu} on devices {config.step2.gpu_devices}")
    logger.info(f"  - Batch size: {config.step2.batch_size}")
    logger.info(f"  - Confidence threshold: {config.step2.conf_threshold}")
    logger.info("")
    logger.info(f"Step 3 (Text Recognition):")
    logger.info(f"  - CPU workers: {config.step3.n_processes_cpu}")
    logger.info(f"  - GPU workers: {config.step3.n_processes_gpu} on devices {config.step3.gpu_devices}")
    logger.info(f"  - Batch size: {config.step3.batch_size}")
    logger.info(f"  - Confidence threshold: {config.step3.conf_threshold}")
    logger.info("=" * 80)


def validate_config(config: PipelineConfig, logger: logging.Logger) -> bool:
    """Validate configuration before processing."""
    errors = []

    if not Path(config.input.video_root_path).exists():
        errors.append(f"Video root path does not exist: {config.input.video_root_path}")

    if not config.input.cameras:
        errors.append("No cameras specified in configuration")

    available_gpus = get_available_gpus()
    all_gpu_devices = set(config.step1.gpu_devices + config.step2.gpu_devices + config.step3.gpu_devices)
    for gpu_id in all_gpu_devices:
        if gpu_id not in available_gpus:
            errors.append(f"GPU device {gpu_id} not available (available: {available_gpus})")

    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    logger.info("✓ Configuration validation passed")
    return True


def run_step1(config: PipelineConfig, logger: PipelineLogger, pipeline_logger: logging.Logger, diagnostics: PipelineDiagnostics, force_reprocess: bool = False) -> bool:
    """Run Step 1: Ring Detection."""
    pipeline_logger.info("")
    pipeline_logger.info("=" * 80)
    pipeline_logger.info("STARTING STEP 1: RING DETECTION")
    pipeline_logger.info("=" * 80)

    start_time = time()

    try:
        from .step1_ring_detection import run_step1_ring_detection
        success = run_step1_ring_detection(config, logger, force_reprocess=force_reprocess)

        if success:
            diagnostics.print_step_report("step1")

        return success

    except Exception as e:
        pipeline_logger.error(f"Step 1 failed with error: {e}", exc_info=True)
        return False
    finally:
        elapsed = time() - start_time
        pipeline_logger.info(f"Step 1 completed in {elapsed:.1f} seconds")


def run_step2(config: PipelineConfig, logger: PipelineLogger, pipeline_logger: logging.Logger, diagnostics: PipelineDiagnostics, force_reprocess: bool = False) -> bool:
    """Run Step 2: Tag Detection."""
    pipeline_logger.info("")
    pipeline_logger.info("=" * 80)
    pipeline_logger.info("STARTING STEP 2: TAG DETECTION")
    pipeline_logger.info("=" * 80)

    start_time = time()

    try:
        from .step2_tag_detection import run_step2_tag_detection
        success = run_step2_tag_detection(config, logger, force_reprocess=force_reprocess)

        if success:
            diagnostics.print_step_report("step2")

        return success

    except Exception as e:
        pipeline_logger.error(f"Step 2 failed with error: {e}", exc_info=True)
        return False
    finally:
        elapsed = time() - start_time
        pipeline_logger.info(f"Step 2 completed in {elapsed:.1f} seconds")


def run_step3(config: PipelineConfig, logger: PipelineLogger, pipeline_logger: logging.Logger, diagnostics: PipelineDiagnostics, force_reprocess: bool = False) -> bool:
    """Run Step 3: Text Recognition."""
    pipeline_logger.info("")
    pipeline_logger.info("=" * 80)
    pipeline_logger.info("STARTING STEP 3: TEXT RECOGNITION")
    pipeline_logger.info("=" * 80)

    start_time = time()

    try:
        from .step3_text_recognition import run_step3_text_recognition
        success = run_step3_text_recognition(config, logger, force_reprocess=force_reprocess)

        if success:
            diagnostics.print_step_report("step3")

        return success

    except Exception as e:
        pipeline_logger.error(f"Step 3 failed with error: {e}", exc_info=True)
        return False
    finally:
        elapsed = time() - start_time
        pipeline_logger.info(f"Step 3 completed in {elapsed:.1f} seconds")


def main(batch_yaml_path: str, force_reprocess: bool = False):
    """Main orchestrator entry point."""
    # Setup logging
    logger = setup_logging()

    try:
        logger.info("AutoRingReader Pipeline Orchestrator Starting")
        logger.info(f"Batch configuration: {batch_yaml_path}")
        if force_reprocess:
            logger.info("⚠️  FORCE REPROCESS MODE ENABLED - will reprocess all videos")

        # Load configuration
        if not Path(batch_yaml_path).exists():
            logger.error(f"Batch YAML file not found: {batch_yaml_path}")
            return 1

        config = PipelineConfig.from_yaml(batch_yaml_path)
        print_config_summary(config, logger)

        # Validate configuration
        if not validate_config(config, logger):
            return 1

        # Ensure output directories
        ensure_output_dir(config.output_root_path)

        # Initialize database logger and diagnostics
        db_path = config.get_db_path()
        pipeline_logger = PipelineLogger(str(db_path), logger)
        diagnostics = PipelineDiagnostics(config.output_root_path, str(db_path), logger)

        # Check available videos
        all_videos = get_video_list(config, step=1)
        logger.info(f"Total videos to process: {len(all_videos)}")

        if not all_videos:
            logger.warning("No videos found to process")
            return 0

        # Run pipelines
        overall_start = time()

        # Step 1: Ring Detection
        logger.info("Running Step 1: Ring Detection")
        if not run_step1(config, pipeline_logger, logger, diagnostics, force_reprocess):
            logger.error("Step 1 failed")
            return 1

        # Step 2: Tag Detection
        logger.info("Running Step 2: Tag Detection")
        if not run_step2(config, pipeline_logger, logger, diagnostics, force_reprocess):
            logger.error("Step 2 failed")
            return 1

        # Step 3: Text Recognition
        logger.info("Running Step 3: Text Recognition")
        if not run_step3(config, pipeline_logger, logger, diagnostics, force_reprocess):
            logger.error("Step 3 failed")
            return 1

        # Print combined diagnostics
        diagnostics.print_combined_report()

        if not diagnostics.save_final_summary(pipeline_logger):
            logger.error("Failed to save final pipeline summary")
            return 1

        overall_elapsed = time() - overall_start
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"✓ PIPELINE COMPLETED SUCCESSFULLY in {overall_elapsed:.1f} seconds")
        logger.info("=" * 80)

        # Finalize logger
        pipeline_logger.close()

        return 0

    except Exception as e:
        logger.error(f"Pipeline orchestrator failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoRingReader Step-Based Pipeline Orchestrator")
    parser.add_argument("batch_yaml", help="Path to batch configuration YAML file")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all videos (ignore resume)")
    args = parser.parse_args()

    exit_code = main(args.batch_yaml, force_reprocess=args.force)
    sys.exit(exit_code)
