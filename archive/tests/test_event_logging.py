#!/usr/bin/env python3
"""
Test strategy to diagnose event logging issues.
Run this script to isolate where logging is failing.
"""

import sqlite3
from pathlib import Path
import sys
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s'
)

def test_1_database_file_creation():
    """Test 1: Can we create the database file?"""
    print("\n" + "="*60)
    print("TEST 1: Database File Creation")
    print("="*60)

    db_path = Path.cwd().joinpath("log", "events_test.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        if db_path.exists():
            print(f"✓ Database file created successfully: {db_path}")
            print(f"  File size: {db_path.stat().st_size} bytes")
            db_path.unlink()  # Clean up
            return True
        else:
            print(f"✗ Database file was not created")
            return False
    except Exception as e:
        print(f"✗ Failed to create database: {e}")
        return False


def test_2_event_logger_initialization():
    """Test 2: Can EventLogger initialize without errors?"""
    print("\n" + "="*60)
    print("TEST 2: EventLogger Initialization")
    print("="*60)

    try:
        from event_logger import EventLogger

        db_path = Path.cwd().joinpath("log", "events_test2.db")
        logger = EventLogger(db_path)

        print(f"✓ EventLogger initialized successfully")
        print(f"  Database path: {db_path}")
        print(f"  Database exists: {db_path.exists()}")

        logger.close()
        db_path.unlink()  # Clean up
        return True
    except Exception as e:
        print(f"✗ EventLogger initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_single_log_write():
    """Test 3: Can we write a single video summary?"""
    print("\n" + "="*60)
    print("TEST 3: Single Log Write")
    print("="*60)

    try:
        from event_logger import EventLogger

        db_path = Path.cwd().joinpath("log", "events_test3.db")
        logger = EventLogger(db_path)

        # Try to log a video summary
        logger.log_video_summary(
            video_path="/test/video1.mp4",
            camera="FAR6_SCALE",
            processing_seconds=10.5,
            frames_total=100,
            ring_detect_total=50,
            ring_detect_roi=40,
            tag_detect_total=100,
            tag_detect_frames=40,
            ocr_final_accepts=35,
            final_tag="ABC123",
            final_tag_count=5,
            status="success"
        )

        logger.close()

        # Verify the write
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM video_summary")
        count = cursor.fetchone()[0]
        conn.close()

        if count == 1:
            print(f"✓ Successfully wrote and verified 1 row to database")
            db_path.unlink()  # Clean up
            return True
        else:
            print(f"✗ Database write failed or data not committed. Found {count} rows")
            return False
    except Exception as e:
        print(f"✗ Single log write failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_multiprocessing_simulation():
    """Test 4: Can multiprocessing workers log data?"""
    print("\n" + "="*60)
    print("TEST 4: Multiprocessing Simulation")
    print("="*60)

    from multiprocessing import get_context
    from functools import partial

    def worker(video_num):
        """Simulated worker that returns stats"""
        return {
            "video_path": f"/test/video{video_num}.mp4",
            "camera": "FAR6_SCALE",
            "processing_seconds": 10.0 + video_num,
            "frames_total": 100 + video_num * 10,
            "ring_detect_total": 50 + video_num,
            "ring_detect_roi": 40 + video_num,
            "tag_detect_total": 100 + video_num,
            "tag_detect_frames": 40 + video_num,
            "ocr_final_accepts": 35 + video_num,
            "final_tag": f"ABC12{video_num}",
            "final_tag_count": 5 + video_num,
            "status": "success"
        }

    try:
        from event_logger import EventLogger

        db_path = Path.cwd().joinpath("log", "events_test4.db")
        event_logger = EventLogger(db_path)

        # Simulate processing with multiprocessing
        with get_context('spawn').Pool(2, maxtasksperchild=1) as p:
            for stats_row in p.imap_unordered(worker, range(5)):
                print(f"  Processing result for: {stats_row['video_path']}")

                try:
                    event_logger.log_video_summary(
                        video_path=stats_row['video_path'],
                        camera=stats_row['camera'],
                        processing_seconds=stats_row['processing_seconds'],
                        frames_total=stats_row['frames_total'],
                        ring_detect_total=stats_row['ring_detect_total'],
                        ring_detect_roi=stats_row['ring_detect_roi'],
                        tag_detect_total=stats_row['tag_detect_total'],
                        tag_detect_frames=stats_row['tag_detect_frames'],
                        ocr_final_accepts=stats_row['ocr_final_accepts'],
                        final_tag=stats_row['final_tag'],
                        final_tag_count=stats_row['final_tag_count'],
                        status=stats_row['status'],
                    )
                    print(f"    ✓ Logged")
                except Exception as e:
                    print(f"    ✗ Failed to log: {e}")

        event_logger.close()

        # Verify all writes
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM video_summary")
        count = cursor.fetchone()[0]
        conn.close()

        if count == 5:
            print(f"✓ Successfully logged all 5 rows in multiprocessing simulation")
            db_path.unlink()  # Clean up
            return True
        else:
            print(f"✗ Only found {count} rows instead of 5")
            print(f"  Database file: {db_path}")
            return False
    except Exception as e:
        print(f"✗ Multiprocessing simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_check_actual_database():
    """Test 5: Check the actual events.db for data"""
    print("\n" + "="*60)
    print("TEST 5: Check Actual Database")
    print("="*60)

    db_path = Path.cwd().joinpath("log", "events.db")

    if not db_path.exists():
        print(f"✗ Database file doesn't exist: {db_path}")
        return False

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check video_summary table
        cursor.execute("SELECT COUNT(*) FROM video_summary")
        count = cursor.fetchone()[0]
        print(f"  Rows in video_summary: {count}")

        if count > 0:
            cursor.execute("SELECT video_path, camera, status FROM video_summary LIMIT 5")
            rows = cursor.fetchall()
            for row in rows:
                print(f"    - {row[0]} ({row[1]}) status={row[2]}")

        # Check other tables
        cursor.execute("SELECT COUNT(*) FROM events")
        print(f"  Rows in events: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM ring_detections")
        print(f"  Rows in ring_detections: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM tag_detections")
        print(f"  Rows in tag_detections: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM text_recognition")
        print(f"  Rows in text_recognition: {cursor.fetchone()[0]}")

        conn.close()

        if count == 0:
            print(f"\n✗ Database exists but is empty (no rows in video_summary)")
            return False
        else:
            print(f"\n✓ Database contains {count} rows in video_summary")
            return True
    except Exception as e:
        print(f"✗ Failed to query actual database: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("EVENT LOGGING DIAGNOSTIC TEST SUITE")
    print("="*60)

    results = {}

    results['test_1_db_creation'] = test_1_database_file_creation()
    results['test_2_logger_init'] = test_2_event_logger_initialization()
    results['test_3_single_write'] = test_3_single_log_write()
    results['test_4_multiprocessing'] = test_4_multiprocessing_simulation()
    results['test_5_actual_db'] = test_5_check_actual_database()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests PASSED - event logging infrastructure is working")
        print("  The issue likely lies in:")
        print("  1. The actual pipeline not calling log_video_summary()")
        print("  2. Worker function returning None or invalid stats_row")
        print("  3. Conditional logic preventing logging (line 771-798)")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"✗ {len(failed)} tests FAILED:")
        for test_name in failed:
            print(f"  - {test_name}")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
