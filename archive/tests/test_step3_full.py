#!/usr/bin/env python3
"""Quick test of full Step 3 with improved diagnostics."""

import sys
import logging
from pathlib import Path

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path('.')))

from pipeline_config import PipelineConfig
from pipeline_logger import PipelineLogger
from step3_text_recognition import run_step3_text_recognition

def main():
    # Load config
    config = PipelineConfig.from_yaml("batch/batch_pipeline_test.yaml")

    # Create logger (with proper directory)
    output_dir = Path(config.output_root_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    try:
        db_logger = PipelineLogger(config.output_root_path)
        logger.info("✓ Database logger initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database logger: {e}")
        logger.info("Continuing anyway...")
        db_logger = None

    # Run step 3 with small batch for testing
    original_batch_size = config.step3.batch_size
    config.step3.batch_size = 50  # Smaller batch size for testing
    config.step3.n_processes_cpu = 4  # Fewer CPUs for testing

    logger.info(f"Using batch_size={config.step3.batch_size}, n_processes_cpu={config.step3.n_processes_cpu}")

    try:
        success = run_step3_text_recognition(config, db_logger, force_reprocess=False)
        if success:
            logger.info("✓ Step 3 completed successfully!")
            sys.exit(0)
        else:
            logger.error("✗ Step 3 failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Step 3 interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Step 3 crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

