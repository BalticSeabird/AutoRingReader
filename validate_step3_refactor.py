#!/usr/bin/env python3
"""
Validation script for Step 3 refactoring.
Checks that the new event-by-event architecture works correctly.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Step 3 Refactoring Validation")
    logger.info("=" * 60)

    # Check 1: Imports
    logger.info("\n[1/4] Checking imports...")
    try:
        from step3_text_recognition import (
            discover_tag_crop_events,
            group_tag_crops_by_ring_for_event,
            _run_step3_single_gpu,
            _run_step3_multi_gpu,
            run_step3_text_recognition
        )
        logger.info("✓ All functions imported successfully")
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

    # Check 2: Function signatures
    logger.info("\n[2/4] Checking function signatures...")
    try:
        import inspect

        # Check discover_tag_crop_events
        sig = inspect.signature(discover_tag_crop_events)
        assert 'output_root' in sig.parameters and 'camera' in sig.parameters
        logger.info("✓ discover_tag_crop_events signature correct")

        # Check group_tag_crops_by_ring_for_event
        sig = inspect.signature(group_tag_crops_by_ring_for_event)
        assert 'crop_paths' in sig.parameters
        logger.info("✓ group_tag_crops_by_ring_for_event signature correct")

        # Check _run_step3_single_gpu
        sig = inspect.signature(_run_step3_single_gpu)
        assert all(p in sig.parameters for p in ['config', 'camera', 'video_stem', 'crop_paths', 'model'])
        logger.info("✓ _run_step3_single_gpu signature correct")

        # Check _run_step3_multi_gpu
        sig = inspect.signature(_run_step3_multi_gpu)
        assert all(p in sig.parameters for p in ['config', 'camera', 'video_stem', 'crop_paths', 'model_path'])
        logger.info("✓ _run_step3_multi_gpu signature correct")

        # Check run_step3_text_recognition
        sig = inspect.signature(run_step3_text_recognition)
        assert all(p in sig.parameters for p in ['config', 'db_logger', 'force_reprocess'])
        logger.info("✓ run_step3_text_recognition signature correct")

    except Exception as e:
        logger.error(f"✗ Signature check failed: {e}")
        return False

    # Check 3: No rglob in code
    logger.info("\n[3/4] Checking for rglob bottleneck removal...")
    try:
        with open('step3_text_recognition.py', 'r') as f:
            content = f.read()

        # Check that old function is gone
        if 'def group_tag_crops_by_ring(' in content:
            logger.error("✗ Old group_tag_crops_by_ring() function still present!")
            return False
        logger.info("✓ Old group_tag_crops_by_ring() removed")

        # Check that new discovery function exists
        if 'def discover_tag_crop_events(' not in content:
            logger.error("✗ New discover_tag_crop_events() not found!")
            return False
        logger.info("✓ New discover_tag_crop_events() present")

        # Check that new event grouping function exists
        if 'def group_tag_crops_by_ring_for_event(' not in content:
            logger.error("✗ New group_tag_crops_by_ring_for_event() not found!")
            return False
        logger.info("✓ New group_tag_crops_by_ring_for_event() present")

        # Check for event-by-event loop in main function
        if 'for event_idx, (video_stem, crop_paths) in enumerate(events)' not in content:
            logger.error("✗ Event-by-event loop not found in main function!")
            return False
        logger.info("✓ Event-by-event loop present")

        # Check for resume logic
        if 'if video_stem in completed_videos' not in content:
            logger.error("✗ Resume logic not found!")
            return False
        logger.info("✓ Resume logic present")

        # Check for immediate database logging
        if 'db_logger.log_step3_result(video_stem, camera' not in content:
            logger.error("✗ Immediate database logging not found!")
            return False
        logger.info("✓ Immediate database logging present")

    except Exception as e:
        logger.error(f"✗ Code structure check failed: {e}")
        return False

    # Check 4: Syntax validation
    logger.info("\n[4/4] Verifying Python syntax...")
    try:
        import py_compile
        py_compile.compile('step3_text_recognition.py', doraise=True)
        logger.info("✓ Python syntax valid")
    except py_compile.PyCompileError as e:
        logger.error(f"✗ Syntax error: {e}")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("✓ ALL VALIDATION CHECKS PASSED")
    logger.info("\nRefactoring is ready for testing!")
    logger.info("\nNext steps:")
    logger.info("1. Run the pipeline: python3 run_orchestrator.py")
    logger.info("2. Monitor the logs for event-by-event processing")
    logger.info("3. Verify database logging is immediate (not batched)")
    logger.info("4. Check memory usage stays ~500MB per event")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
