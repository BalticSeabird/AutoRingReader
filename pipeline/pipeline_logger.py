"""Extended event logging system for step-based pipeline."""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


class PipelineLogger:
    """Extended logger for pipeline stage results with per-step database tables."""

    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None):
        """Initialize pipeline logger with database schema."""
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)

        # Ensure log directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_schema()

    def _init_schema(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()

            # Step 1: Ring Detection Results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS step1_ring_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    video_path TEXT UNIQUE NOT NULL,
                    camera TEXT,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    processing_seconds REAL,
                    frames_total INTEGER,
                    frames_processed INTEGER,
                    ring_detect_count INTEGER,
                    ring_detect_rate REAL,
                    ring_conf_mean REAL,
                    ring_conf_max REAL,
                    output_path TEXT,
                    queue_file TEXT
                )
            """)

            # Step 2: Tag Detection Results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS step2_tag_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    video_path TEXT UNIQUE NOT NULL,
                    camera TEXT,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    processing_seconds REAL,
                    rings_processed INTEGER,
                    tags_detected_count INTEGER,
                    tags_detected_frames INTEGER,
                    tag_detection_rate REAL,
                    tag_conf_mean REAL,
                    tag_conf_max REAL,
                    output_path TEXT,
                    queue_file TEXT
                )
            """)

            # Step 3: Text Recognition Results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS step3_text_recognition (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    video_path TEXT UNIQUE NOT NULL,
                    camera TEXT,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    processing_seconds REAL,
                    tags_processed INTEGER,
                    ocr_matches_count INTEGER,
                    ocr_success_rate REAL,
                    letters_conf_min_mean REAL,
                    numbers_conf_min_mean REAL,
                    letters_ema_min_mean REAL,
                    numbers_ema_min_mean REAL,
                    output_path TEXT
                )
            """)

            # Final summary (aggregated across all steps)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS final_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    video_path TEXT UNIQUE NOT NULL,
                    camera TEXT,
                    video_stem TEXT,
                    step1_status TEXT,
                    step1_frames_total INTEGER,
                    step1_frames_processed INTEGER,
                    step1_ring_detections INTEGER,
                    step1_ring_conf_mean REAL,
                    step1_ring_conf_max REAL,
                    step1_time REAL,
                    step1_error TEXT,
                    step2_status TEXT,
                    step2_ring_crops_count INTEGER,
                    step2_tags_detected INTEGER,
                    step2_tag_conf_mean REAL,
                    step2_tag_conf_max REAL,
                    step2_time REAL,
                    step2_error TEXT,
                    step3_status TEXT,
                    step3_final_tag TEXT,
                    step3_ocr_matches INTEGER,
                    step3_letters_conf REAL,
                    step3_numbers_conf REAL,
                    step3_time REAL,
                    step3_error TEXT
                )
            """)

            # Remove schemas from older versions that are no longer used.
            cursor.execute("DROP TABLE IF EXISTS final_results")
            cursor.execute("DROP TABLE IF EXISTS video_summary")

            conn.commit()

    def log_step1_result(self, video_path: str, camera: str, stats: Dict[str, Any]) -> bool:
        """Log Step 1 (Ring Detection) results."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO step1_ring_detections
                    (video_path, camera, status, error_message, processing_seconds,
                     frames_total, frames_processed, ring_detect_count, ring_detect_rate,
                     ring_conf_mean, ring_conf_max, output_path, queue_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_path, camera,
                    stats.get('status', 'success'),
                    stats.get('error_message'),
                    stats.get('processing_seconds'),
                    stats.get('frames_total'),
                    stats.get('frames_processed'),
                    stats.get('ring_detect_count'),
                    stats.get('ring_detect_rate'),
                    stats.get('ring_conf_mean'),
                    stats.get('ring_conf_max'),
                    stats.get('output_path'),
                    stats.get('queue_file'),
                ))

                conn.commit()
                self.logger.info(f"✓ Logged Step 1: {video_path}")
                return True
        except Exception as e:
            self.logger.error(f"✗ Failed to log Step 1 result for {video_path}: {e}")
            return False

    def log_step2_result(self, video_path: str, camera: str, stats: Dict[str, Any]) -> bool:
        """Log Step 2 (Tag Detection) results."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO step2_tag_detections
                    (video_path, camera, status, error_message, processing_seconds,
                     rings_processed, tags_detected_count, tags_detected_frames,
                     tag_detection_rate, tag_conf_mean, tag_conf_max, output_path, queue_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_path, camera,
                    stats.get('status', 'success'),
                    stats.get('error_message'),
                    stats.get('processing_seconds'),
                    stats.get('rings_processed'),
                    stats.get('tags_detected_count'),
                    stats.get('tags_detected_frames'),
                    stats.get('tag_detection_rate'),
                    stats.get('tag_conf_mean'),
                    stats.get('tag_conf_max'),
                    stats.get('output_path'),
                    stats.get('queue_file'),
                ))

                conn.commit()
                self.logger.info(f"✓ Logged Step 2: {video_path}")
                return True
        except Exception as e:
            self.logger.error(f"✗ Failed to log Step 2 result for {video_path}: {e}")
            return False

    def log_step3_result(self, video_path: str, camera: str, stats: Dict[str, Any]) -> bool:
        """Log Step 3 (Text Recognition) results."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO step3_text_recognition
                    (video_path, camera, status, error_message, processing_seconds,
                     tags_processed, ocr_matches_count, ocr_success_rate,
                     letters_conf_min_mean, numbers_conf_min_mean,
                     letters_ema_min_mean, numbers_ema_min_mean, output_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_path, camera,
                    stats.get('status', 'success'),
                    stats.get('error_message'),
                    stats.get('processing_seconds'),
                    stats.get('tags_processed'),
                    stats.get('ocr_matches_count'),
                    stats.get('ocr_success_rate'),
                    stats.get('letters_conf_min_mean'),
                    stats.get('numbers_conf_min_mean'),
                    stats.get('letters_ema_min_mean'),
                    stats.get('numbers_ema_min_mean'),
                    stats.get('output_path'),
                ))

                conn.commit()
                self.logger.info(f"✓ Logged Step 3: {video_path}")
                return True
        except Exception as e:
            self.logger.error(f"✗ Failed to log Step 3 result for {video_path}: {e}")
            return False

    def replace_final_summary(self, summaries: List[Dict[str, Any]]) -> bool:
        """Replace the final per-video summary with the current pipeline results."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM final_summary")
                columns = (
                    "video_path, camera, video_stem, "
                    "step1_status, step1_frames_total, step1_frames_processed, "
                    "step1_ring_detections, step1_ring_conf_mean, step1_ring_conf_max, "
                    "step1_time, step1_error, step2_status, step2_ring_crops_count, "
                    "step2_tags_detected, step2_tag_conf_mean, step2_tag_conf_max, "
                    "step2_time, step2_error, step3_status, step3_final_tag, "
                    "step3_ocr_matches, step3_letters_conf, step3_numbers_conf, "
                    "step3_time, step3_error"
                )
                placeholders = ", ".join("?" for _ in columns.split(", "))
                values = [
                    tuple(summary.get(column) for column in columns.split(", "))
                    for summary in summaries
                ]
                if values:
                    cursor.executemany(
                        f"INSERT INTO final_summary ({columns}) VALUES ({placeholders})",
                        values,
                    )

                conn.commit()
                self.logger.info(f"✓ Saved final summary for {len(summaries)} videos")
                return True
        except Exception as e:
            self.logger.error(f"✗ Failed to save final summary: {e}")
            return False

    def get_pending_videos(self, step: str = "step1", status: str = "pending") -> List[str]:
        """Get list of pending videos for a specific step."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                table_name = f"{step}_ring_detections" if step == "step1" else \
                             f"{step}_tag_detections" if step == "step2" else \
                             f"{step}_text_recognition"

                cursor.execute(f"""
                    SELECT video_path FROM {table_name}
                    WHERE status = ?
                    ORDER BY video_path
                """, (status,))

                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"✗ Failed to get pending videos for {step}: {e}")
            return []

    def get_completed_videos(self, step: str = "step1") -> List[str]:
        """Get list of videos completed for a specific step."""
        return self.get_pending_videos(step, "success")

    def get_step_stats(self, step: str = "step1") -> Dict[str, Any]:
        """Get summary statistics for a processing step."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                table_name = f"{step}_ring_detections" if step == "step1" else \
                             f"{step}_tag_detections" if step == "step2" else \
                             f"{step}_text_recognition"

                cursor.execute(f"""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as success_count,
                        SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as error_count,
                        SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) as pending_count,
                        AVG(processing_seconds) as avg_time
                    FROM {table_name}
                """)

                row = cursor.fetchone()
                if row:
                    return {
                        'total': row[0] or 0,
                        'success': row[1] or 0,
                        'error': row[2] or 0,
                        'pending': row[3] or 0,
                        'avg_time': row[4] or 0.0,
                    }
                return {'total': 0, 'success': 0, 'error': 0, 'pending': 0, 'avg_time': 0.0}
        except Exception as e:
            self.logger.error(f"✗ Failed to get step stats for {step}: {e}")
            return {}

    def close(self):
        """Finalize database and optimize."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA optimize;")
                conn.commit()
            self.logger.info("✓ Database optimized and closed")
        except Exception as e:
            self.logger.error(f"✗ Failed to close database: {e}")
