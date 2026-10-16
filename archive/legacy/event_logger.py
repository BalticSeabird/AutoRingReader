import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import atexit


class EventLogger:
    """SQLite logger for ring detection pipeline events"""

    def __init__(self, db_path: Path):
        """Initialize the event logger with a SQLite database

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        # Register cleanup on exit
        atexit.register(self.close)

    def _initialize_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Video summary table (one entry per video) - stores all stats from video_stats.csv
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    video_path TEXT UNIQUE NOT NULL,
                    camera TEXT NOT NULL,
                    output_rel_path TEXT,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    processing_seconds REAL,
                    frames_total INTEGER,
                    ring_detect_total INTEGER,
                    ring_detect_roi INTEGER,
                    ring_detection_rate REAL,
                    ring_conf_mean REAL,
                    ring_conf_max REAL,
                    tag_detect_total INTEGER,
                    tag_detect_frames INTEGER,
                    tag_detection_rate REAL,
                    tag_conf_mean REAL,
                    tag_conf_max REAL,
                    ocr_candidates INTEGER,
                    ocr_candidate_letters_min_mean REAL,
                    ocr_candidate_numbers_min_mean REAL,
                    ocr_final_accepts INTEGER,
                    ocr_success_rate REAL,
                    ocr_final_letters_min_mean REAL,
                    ocr_final_numbers_min_mean REAL,
                    ocr_final_letters_ema_min_mean REAL,
                    ocr_final_numbers_ema_min_mean REAL,
                    final_tag TEXT,
                    final_tag_count INTEGER,
                    final_letters_conf_min REAL,
                    final_numbers_conf_min REAL,
                    final_letters_ema_min REAL,
                    final_numbers_ema_min REAL
                )
            """)

            conn.commit()


    def log_video_stats_row(self, stats_row: Dict[str, Any]):
        """Log complete video statistics from stats_row dict (all 34 fields from video_stats.csv)
        Data is logged row-by-row immediately after text reading step 3 completes.

        Args:
            stats_row: Dictionary with all video statistics fields
        """
        if not stats_row or not isinstance(stats_row, dict):
            logging.error(f"Invalid stats_row: {type(stats_row)}")
            return

        try:
            # Convert empty strings to None for REAL columns
            def to_float_or_none(val):
                if val == "" or val is None:
                    return None
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None

            # Prepare values tuple
            values = (
                stats_row.get("video_path", ""),
                stats_row.get("camera", ""),
                stats_row.get("output_rel_path", ""),
                stats_row.get("status", "success"),
                stats_row.get("error_message", ""),
                to_float_or_none(stats_row.get("processing_seconds")),
                stats_row.get("frames_total", 0),
                stats_row.get("ring_detect_total", 0),
                stats_row.get("ring_detect_roi", 0),
                to_float_or_none(stats_row.get("ring_detection_rate")),
                to_float_or_none(stats_row.get("ring_conf_mean")),
                to_float_or_none(stats_row.get("ring_conf_max")),
                stats_row.get("tag_detect_total", 0),
                stats_row.get("tag_detect_frames", 0),
                to_float_or_none(stats_row.get("tag_detection_rate")),
                to_float_or_none(stats_row.get("tag_conf_mean")),
                to_float_or_none(stats_row.get("tag_conf_max")),
                stats_row.get("ocr_candidates", 0),
                to_float_or_none(stats_row.get("ocr_candidate_letters_min_mean")),
                to_float_or_none(stats_row.get("ocr_candidate_numbers_min_mean")),
                stats_row.get("ocr_final_accepts", 0),
                to_float_or_none(stats_row.get("ocr_success_rate")),
                to_float_or_none(stats_row.get("ocr_final_letters_min_mean")),
                to_float_or_none(stats_row.get("ocr_final_numbers_min_mean")),
                to_float_or_none(stats_row.get("ocr_final_letters_ema_min_mean")),
                to_float_or_none(stats_row.get("ocr_final_numbers_ema_min_mean")),
                stats_row.get("final_tag", ""),
                stats_row.get("final_tag_count", 0),
                to_float_or_none(stats_row.get("final_letters_conf_min")),
                to_float_or_none(stats_row.get("final_numbers_conf_min")),
                to_float_or_none(stats_row.get("final_letters_ema_min")),
                to_float_or_none(stats_row.get("final_numbers_ema_min")),
            )

            # Single connection for insert + commit + optimize (atomic)
            conn = sqlite3.connect(self.db_path, timeout=10.0)  # 10s timeout for lock wait
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO video_summary
                    (video_path, camera, output_rel_path, status, error_message,
                     processing_seconds, frames_total, ring_detect_total, ring_detect_roi,
                     ring_detection_rate, ring_conf_mean, ring_conf_max,
                     tag_detect_total, tag_detect_frames, tag_detection_rate,
                     tag_conf_mean, tag_conf_max, ocr_candidates,
                     ocr_candidate_letters_min_mean, ocr_candidate_numbers_min_mean,
                     ocr_final_accepts, ocr_success_rate, ocr_final_letters_min_mean,
                     ocr_final_numbers_min_mean, ocr_final_letters_ema_min_mean,
                     ocr_final_numbers_ema_min_mean, final_tag, final_tag_count,
                     final_letters_conf_min, final_numbers_conf_min,
                     final_letters_ema_min, final_numbers_ema_min)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                conn.commit()
                logging.info(f"✓ Logged: {Path(stats_row.get('video_path', '')).name} | status={stats_row.get('status')} | {stats_row.get('processing_seconds')}s")
            finally:
                conn.close()

        except Exception as e:
            logging.error(f"✗ DB insert failed for {stats_row.get('video_path', 'unknown')}: {e}")



    def get_video_summary(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific video

        Args:
            video_path: Path to the video

        Returns:
            Video summary dictionary or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM video_summary WHERE video_path = ?", (video_path,))
            row = cursor.fetchone()

            return dict(row) if row else None

    def get_processed_videos(self, status: str = "success") -> set:
        """Get all processed video paths

        Args:
            status: Filter by status (default: 'success'). Use None to get all.

        Returns:
            Set of video paths that have been processed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if status:
                    cursor.execute("SELECT video_path FROM video_summary WHERE status = ?", (status,))
                else:
                    cursor.execute("SELECT video_path FROM video_summary")
                results = {row[0] for row in cursor.fetchall()}
                logging.debug(f"Found {len(results)} processed videos" + (f" with status={status}" if status else ""))
                return results
        except Exception as e:
            logging.warning(f"Failed to query processed videos: {e}")
            return set()

    def close(self):
        """Close the database and ensure all data is flushed"""
        try:
            # Force WAL checkpoint to ensure all data is written to disk
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA optimize")
                conn.commit()
            logging.debug(f"Event logger database closed and optimized: {self.db_path}")
        except Exception as e:
            logging.error(f"Error closing event logger: {e}")
