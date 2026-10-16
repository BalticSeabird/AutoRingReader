"""Real-time pipeline monitoring script."""

import sqlite3
import time
import sys
import os
from pathlib import Path
from collections import defaultdict


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name == 'posix' else 'cls')


def get_step_stats(db_path: str, step: str = "step1") -> dict:
    """Get statistics for a processing step."""
    try:
        with sqlite3.connect(str(db_path), timeout=5.0) as conn:
            cursor = conn.cursor()

            table_name = f"{step}_ring_detections" if step == "step1" else \
                         f"{step}_tag_detections" if step == "step2" else \
                         f"{step}_text_recognition"

            cursor.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as success_count,
                    SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as error_count,
                    SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) as pending_count
                FROM {table_name}
            """)

            row = cursor.fetchone()
            if row:
                return {
                    'total': row[0] or 0,
                    'success': row[1] or 0,
                    'error': row[2] or 0,
                    'pending': row[3] or 0,
                }
    except Exception as e:
        pass

    return {'total': 0, 'success': 0, 'error': 0, 'pending': 0}


def get_latest_results(db_path: str, step: str = "step1", limit: int = 3) -> list:
    """Get latest results from a processing step."""
    try:
        with sqlite3.connect(str(db_path), timeout=5.0) as conn:
            cursor = conn.cursor()

            table_name = f"{step}_ring_detections" if step == "step1" else \
                         f"{step}_tag_detections" if step == "step2" else \
                         f"{step}_text_recognition"

            cursor.execute(f"""
                SELECT video_path, status, timestamp
                FROM {table_name}
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))

            return cursor.fetchall()
    except:
        return []


def draw_progress_bar(current: int, total: int, width: int = 40) -> str:
    """Draw a simple text progress bar."""
    if total == 0:
        percentage = 0
        filled = 0
    else:
        percentage = int((current / total) * 100)
        filled = int((current / total) * width)

    bar = '█' * filled + '░' * (width - filled)
    return f"{bar} {percentage:3d}%  ({current}/{total})"


def monitor_pipeline(db_path: str, interval: float = 5.0):
    """Monitor pipeline progress."""
    db_path = Path(db_path)

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Waiting for pipeline to start...")

        while not db_path.exists():
            time.sleep(interval)
            print(".", end="", flush=True)

        print("\nDatabase found! Starting monitoring...\n")

    prev_results = {}
    iteration = 0

    while True:
        iteration += 1
        clear_screen()

        print("=" * 80)
        print("AutoRingReader Pipeline Monitor")
        print("=" * 80)
        print(f"Database: {db_path}")
        print(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        stats = {}
        for step_num, step_name in [(1, "step1"), (2, "step2"), (3, "step3")]:
            stats[step_name] = get_step_stats(str(db_path), step_name)

        # Display summary
        print("PIPELINE PROGRESS:")
        print("-" * 80)

        for step_num, step_name in [(1, "Ring Detection (step1)"),
                                    (2, "Tag Detection (step2)"),
                                    (3, "Text Recognition (step3)")]:
            step_key = f"step{step_num}"
            s = stats[step_key]

            if s['total'] == 0:
                status_str = "Not started"
            else:
                pct = int((s['success'] / s['total']) * 100) if s['total'] > 0 else 0
                status_str = f"{s['success']}/{s['total']} success"

            print(f"{step_name:<25}", end="")

            if s['total'] > 0:
                bar = draw_progress_bar(s['success'], s['total'])
                print(bar, end="")
                if s['error'] > 0:
                    print(f"  {s['error']} errors", end="")
                print()
            else:
                print("Pending...")

        print()
        print("RECENT RESULTS:")
        print("-" * 80)

        for step_num, step_name in [(1, "Step 1"), (2, "Step 2"), (3, "Step 3")]:
            step_key = f"step{step_num}"
            print(f"\n{step_name}:")
            latest = get_latest_results(str(db_path), step_key, limit=3)

            if latest:
                for path, status, timestamp in latest:
                    video_name = Path(path).stem
                    status_icon = "✓" if status == "success" else "✗" if status == "error" else "⏳"
                    print(f"  {status_icon} {video_name:<40} [{timestamp}]")
            else:
                print("  No results yet")

        print()
        print("=" * 80)
        print(f"Press Ctrl+C to exit | Refresh interval: {interval}s")
        print()

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor AutoRingReader pipeline progress")
    parser.add_argument("--db", default="log/events.db", help="Path to events database")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    try:
        monitor_pipeline(args.db, args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        sys.exit(0)
