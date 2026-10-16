#!/usr/bin/env python3
"""Quick test script for Step 1 only."""

import sys
import logging
from pathlib import Path
from pipeline_config import PipelineConfig
from pipeline_logger import PipelineLogger
from step1_ring_detection import run_step1_ring_detection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_step1_only.py <batch_yaml>")
        print("\nExample:")
        print("  python3 test_step1_only.py batch/batch_pipeline_test.yaml")
        sys.exit(1)

    batch_yaml = sys.argv[1]

    if not Path(batch_yaml).exists():
        print(f"Error: {batch_yaml} not found")
        sys.exit(1)

    print("=" * 80)
    print("Testing Step 1 (Ring Detection) Only")
    print("=" * 80)

    # Load config
    try:
        config = PipelineConfig.from_yaml(batch_yaml)
        print(f"✓ Config loaded from {batch_yaml}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        sys.exit(1)

    # Initialize logger
    db_path = config.get_db_path()
    print(f"Database: {db_path}")

    try:
        db_logger = PipelineLogger(str(db_path))
        print("✓ Database logger initialized")
    except Exception as e:
        print(f"✗ Failed to initialize logger: {e}")
        sys.exit(1)

    # Run Step 1
    print("\n" + "=" * 80)
    print("Running Step 1...")
    print("=" * 80 + "\n")

    success = run_step1_ring_detection(config, db_logger)

    print("\n" + "=" * 80)
    if success:
        print("✓ Step 1 completed successfully!")

        # Show database stats
        stats = db_logger.get_step_stats("step1")
        print(f"\nDatabase stats:")
        print(f"  Total videos: {stats['total']}")
        print(f"  Successful: {stats['success']}")
        print(f"  Errors: {stats['error']}")
        print(f"  Pending: {stats['pending']}")
    else:
        print("✗ Step 1 failed")
        sys.exit(1)

    print("=" * 80)
