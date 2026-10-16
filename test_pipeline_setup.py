#!/usr/bin/env python3
"""Quick test to validate pipeline setup before running full processing."""

import sys
from pathlib import Path

print("=" * 80)
print("PIPELINE SETUP VALIDATION TEST")
print("=" * 80)

# Test 1: Import all modules
print("\n[1/5] Testing module imports...")
try:
    from pipeline_config import PipelineConfig
    from pipeline_logger import PipelineLogger
    from pipeline_utils import get_video_list
    from pipeline_orchestrator import main
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load batch config
print("\n[2/5] Testing batch configuration loading...")
try:
    if not Path("batch/batch_pipeline_test.yaml").exists():
        print("✗ batch/batch_pipeline_test.yaml not found")
        sys.exit(1)
    config = PipelineConfig.from_yaml("batch/batch_pipeline_test.yaml")
    print(f"✓ Config loaded: {len(config.input.cameras)} cameras")
    print(f"  - Ring detector model: {config.models.ring_detector.model_name}")
    print(f"  - Tag detector model: {config.models.tag_detector.model_name}")
    print(f"  - Text recognition model: {config.models.text_recognition.model_name}")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    sys.exit(1)

# Test 3: Validate paths
print("\n[3/5] Testing path validation...")
try:
    video_root = Path(config.input.video_root_path)
    if not video_root.exists():
        print(f"✗ Video root path doesn't exist: {video_root}")
        sys.exit(1)

    output_root = Path(config.output_root_path)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"✓ Paths valid:")
    print(f"  - Input: {video_root}")
    print(f"  - Output: {output_root}")
except Exception as e:
    print(f"✗ Path validation failed: {e}")
    sys.exit(1)

# Test 4: Database initialization
print("\n[4/5] Testing database initialization...")
try:
    db_path = config.get_db_path()
    logger = PipelineLogger(str(db_path))
    stats = logger.get_step_stats("step1")
    print(f"✓ Database initialized:")
    print(f"  - Path: {db_path}")
    print(f"  - Step1 status: {stats}")
    logger.close()
except Exception as e:
    print(f"✗ Database init failed: {e}")
    sys.exit(1)

# Test 5: Video discovery
print("\n[5/5] Testing video discovery...")
try:
    videos = get_video_list(config, step=1)
    print(f"✓ Found {len(videos)} videos to process:")
    for video_path, camera in videos[:3]:
        print(f"  - {Path(video_path).name} ({camera})")
    if len(videos) > 3:
        print(f"  ... and {len(videos) - 3} more")
except Exception as e:
    print(f"✗ Video discovery failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED - Pipeline is ready!")
print("=" * 80)
print("\nNext step: python3 pipeline_orchestrator.py batch/batch_pipeline_test.yaml")
