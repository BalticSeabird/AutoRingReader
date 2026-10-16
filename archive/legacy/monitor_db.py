#!/usr/bin/env python3
"""Real-time database monitoring script for AutoRingReader event logging"""

import sqlite3
import time
import sys
from pathlib import Path

DB_PATH = Path("./log/events.db")

def query_db():
    """Query database and show current state"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get counts
        cursor.execute("SELECT COUNT(*) as total FROM video_summary")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as success FROM video_summary WHERE status = 'success'")
        success = cursor.fetchone()["success"]

        cursor.execute("SELECT COUNT(*) as error FROM video_summary WHERE status = 'error'")
        error = cursor.fetchone()["error"]

        # Get latest entries
        cursor.execute("""
            SELECT video_path, status, processing_seconds, final_tag_count
            FROM video_summary
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        latest = cursor.fetchall()

        conn.close()
        return {
            "total": total,
            "success": success,
            "error": error,
            "latest": [dict(row) for row in latest]
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        sys.exit(1)

    print(f"Monitoring database: {DB_PATH}")
    print("Press Ctrl+C to stop\n")

    last_total = 0
    while True:
        result = query_db()

        if "error" in result:
            print(f"❌ Database error: {result['error']}")
        else:
            total = result["total"]
            success = result["success"]
            error = result["error"]
            latest = result["latest"]

            # Show summary
            print(f"\n[{time.strftime('%H:%M:%S')}] Total: {total} | ✓Success: {success} | ✗Error: {error}", end="")

            if total > last_total:
                print(f" | NEW: +{total - last_total}")
                last_total = total
            else:
                print()

            # Show latest 3 entries
            if latest:
                print("└─ Latest entries:")
                for row in latest[:3]:
                    video_name = Path(row["video_path"]).name if row["video_path"] else "?"
                    status_icon = "✓" if row["status"] == "success" else "✗"
                    proc_time = f"{row['processing_seconds']:.1f}s" if row["processing_seconds"] else "?"
                    tags = f"tags={row['final_tag_count']}" if row["final_tag_count"] else "no_tags"
                    print(f"   {status_icon} {video_name[:50]:50} {proc_time:8} {tags}")

        time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
