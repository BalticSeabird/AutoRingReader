"""Pipeline diagnostics - comprehensive performance and quality metrics tracking."""

import sqlite3
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


@dataclass
class VideoMetrics:
    """Metrics for a single video across all steps."""
    camera: str
    video_stem: str
    video_path: str = ""

    # Step 1: Ring Detection
    step1_status: str = "pending"
    step1_frames_total: int = 0
    step1_frames_processed: int = 0
    step1_ring_detections: int = 0
    step1_ring_conf_mean: Optional[float] = None
    step1_ring_conf_max: Optional[float] = None
    step1_time: Optional[float] = None
    step1_error: Optional[str] = None

    # Step 2: Tag Detection
    step2_status: str = "pending"
    step2_ring_crops_count: int = 0
    step2_tags_detected: int = 0
    step2_tag_conf_mean: Optional[float] = None
    step2_tag_conf_max: Optional[float] = None
    step2_time: Optional[float] = None
    step2_error: Optional[str] = None

    # Step 3: Text Recognition
    step3_status: str = "pending"
    step3_final_tag: Optional[str] = None
    step3_ocr_matches: int = 0
    step3_letters_conf: Optional[float] = None
    step3_numbers_conf: Optional[float] = None
    step3_time: Optional[float] = None
    step3_error: Optional[str] = None


class PipelineDiagnostics:
    """Generates and reports diagnostics by scanning output folders."""

    def __init__(self, output_root: str, db_path: str, logger: Optional[logging.Logger] = None):
        """Initialize diagnostics."""
        self.output_root = Path(output_root)
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)

    def scan_all_videos(self) -> Dict[str, VideoMetrics]:
        """
        Scan output directories to get metrics for all videos.
        Returns dict mapping video_stem to VideoMetrics.
        """
        videos = {}

        # Scan step 1 output (ring crops)
        step1_videos = self._scan_step1()
        for stem, metrics in step1_videos.items():
            if stem not in videos:
                videos[stem] = metrics
            else:
                videos[stem].step1_ring_detections = metrics.step1_ring_detections
                videos[stem].step1_ring_conf_mean = metrics.step1_ring_conf_mean

        # Scan step 2 output (tag crops)
        step2_videos = self._scan_step2()
        for stem, metrics in step2_videos.items():
            if stem not in videos:
                videos[stem] = metrics
            else:
                v = videos[stem]
                v.step2_ring_crops_count = metrics.step2_ring_crops_count
                v.step2_tags_detected = metrics.step2_tags_detected
                v.step2_tag_conf_mean = metrics.step2_tag_conf_mean

        # Scan step 3 output (OCR results)
        step3_videos = self._scan_step3()
        for stem, metrics in step3_videos.items():
            if stem not in videos:
                videos[stem] = metrics
            else:
                v = videos[stem]
                v.step3_final_tag = metrics.step3_final_tag
                v.step3_ocr_matches = metrics.step3_ocr_matches
                v.step3_letters_conf = metrics.step3_letters_conf
                v.step3_numbers_conf = metrics.step3_numbers_conf

        # Get database status
        db_status = self._get_database_status()
        for stem, status_info in db_status.items():
            if stem in videos:
                videos[stem].step1_status = status_info.get('step1_status', 'pending')
                videos[stem].step2_status = status_info.get('step2_status', 'pending')
                videos[stem].step3_status = status_info.get('step3_status', 'pending')
                videos[stem].step1_time = status_info.get('step1_time')
                videos[stem].step2_time = status_info.get('step2_time')
                videos[stem].step3_time = status_info.get('step3_time')

        return videos

    def _scan_step1(self) -> Dict[str, VideoMetrics]:
        """Scan step1_ring_crops directory for ring detection metrics."""
        videos = {}
        step1_dir = self.output_root / "step1_ring_crops"

        if not step1_dir.exists():
            return videos

        # Find all ring crops grouped by camera/video
        for camera_dir in step1_dir.iterdir():
            if not camera_dir.is_dir():
                continue
            camera = camera_dir.name

            for video_dir in camera_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                video_stem = video_dir.name

                # Count ring crops
                ring_crops = list(video_dir.glob("*.png"))
                crop_count = len(ring_crops)

                metrics = VideoMetrics(
                    camera=camera,
                    video_stem=video_stem,
                )
                metrics.step1_ring_detections = crop_count
                videos[video_stem] = metrics

        return videos

    def _scan_step2(self) -> Dict[str, VideoMetrics]:
        """Scan step2_tag_crops directory for tag detection metrics."""
        videos = {}
        step2_dir = self.output_root / "step2_tag_crops"

        if not step2_dir.exists():
            return videos

        # Find all tag crops grouped by camera/video
        for camera_dir in step2_dir.iterdir():
            if not camera_dir.is_dir():
                continue
            camera = camera_dir.name

            for video_dir in camera_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                video_stem = video_dir.name

                # Find all tag crops
                tag_crops = list(video_dir.glob("*.jpg"))

                # Parse confidence and count unique rings/tags
                tag_confidences = []
                ring_ids = set()
                tag_count = 0

                for crop_file in tag_crops:
                    # Filename format: {frame_id}_ring_tag_{idx}.jpg
                    stem = crop_file.stem
                    if "_ring_tag_" in stem:
                        ring_ids.add(stem.split("_ring_tag_")[0])
                        tag_count += 1

                ring_crop_count = len(ring_ids)

                metrics = VideoMetrics(
                    camera=camera,
                    video_stem=video_stem,
                )
                metrics.step2_ring_crops_count = ring_crop_count
                metrics.step2_tags_detected = tag_count

                videos[video_stem] = metrics

        return videos

    def _scan_step3(self) -> Dict[str, VideoMetrics]:
        """Scan step3_ocr_results directory for OCR metrics."""
        videos = {}
        step3_dir = self.output_root / "step3_ocr_results"

        if not step3_dir.exists():
            return videos

        # Find all YAML results grouped by camera/video
        for camera_dir in step3_dir.iterdir():
            if not camera_dir.is_dir():
                continue
            camera = camera_dir.name

            for video_dir in camera_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                video_stem = video_dir.name

                # Find YAML result file
                yaml_files = list(video_dir.glob("*.yaml"))
                if not yaml_files:
                    continue

                final_tag = None
                letters_conf = None
                numbers_conf = None

                try:
                    yaml_file = yaml_files[0]
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f)
                        if data:
                            final_tag = data.get('final_tag')
                            letters_conf = data.get('letters_confidence')
                            numbers_conf = data.get('numbers_confidence')
                except Exception as e:
                    self.logger.debug(f"Error reading {yaml_file}: {e}")
                    continue

                metrics = VideoMetrics(
                    camera=camera,
                    video_stem=video_stem,
                )
                metrics.step3_final_tag = final_tag
                metrics.step3_ocr_matches = 1 if final_tag else 0
                metrics.step3_letters_conf = letters_conf
                metrics.step3_numbers_conf = numbers_conf

                videos[video_stem] = metrics

        return videos

    def _get_database_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status and timing info from database."""
        status = {}

        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get all videos from step1
                cursor.execute("SELECT video_path, camera, status, processing_seconds FROM step1_ring_detections")
                for row in cursor.fetchall():
                    parts = Path(row['video_path']).stem
                    if parts not in status:
                        status[parts] = {}
                    status[parts]['step1_status'] = row['status']
                    status[parts]['step1_time'] = row['processing_seconds']

                # Get step2 status
                cursor.execute("SELECT video_path, status, processing_seconds FROM step2_tag_detections")
                for row in cursor.fetchall():
                    parts = Path(row['video_path']).stem
                    if parts not in status:
                        status[parts] = {}
                    status[parts]['step2_status'] = row['status']
                    status[parts]['step2_time'] = row['processing_seconds']

                # Get step3 status
                cursor.execute("SELECT video_path, status, processing_seconds FROM step3_text_recognition")
                for row in cursor.fetchall():
                    parts = Path(row['video_path']).stem
                    if parts not in status:
                        status[parts] = {}
                    status[parts]['step3_status'] = row['status']
                    status[parts]['step3_time'] = row['processing_seconds']

        except Exception as e:
            self.logger.debug(f"Error reading database status: {e}")

        return status

    def print_step_report(self, step: str) -> None:
        """Print formatted report for a step."""
        videos = self.scan_all_videos()

        self.logger.info("")
        self.logger.info("=" * 120)
        self.logger.info(f"DIAGNOSTICS: {step.upper()} - POST-PROCESSING REPORT")
        self.logger.info("=" * 120)

        # Filter to videos with data for this step
        if step == "step1":
            step_videos = {s: v for s, v in videos.items() if v.step1_ring_detections > 0}
            if not step_videos:
                self.logger.info("No Step 1 data found")
                return

            self.logger.info(f"\n🎯 STEP 1: RING DETECTION")
            self.logger.info(f"  Total Videos: {len(step_videos)}")
            total_detections = sum(v.step1_ring_detections for v in step_videos.values())
            self.logger.info(f"  Total Ring Detections: {total_detections}")
            avg_detections = total_detections / len(step_videos) if step_videos else 0
            self.logger.info(f"  Avg Detections per Video: {avg_detections:.1f}")

            self.logger.info(f"\n📝 TOP 15 VIDEOS BY DETECTIONS")
            sorted_videos = sorted(step_videos.items(),
                                 key=lambda x: x[1].step1_ring_detections, reverse=True)
            for i, (stem, v) in enumerate(sorted_videos[:15], 1):
                status = v.step1_status if v.step1_status else "?"
                time_s = f"{v.step1_time:.1f}s" if v.step1_time else "?"
                self.logger.info(
                    f"  {i:2d}. {v.camera:15s} {stem:30s} | "
                    f"Detections: {v.step1_ring_detections:4d} [{status:7s}] {time_s}"
                )

        elif step == "step2":
            step_videos = {s: v for s, v in videos.items() if v.step2_tags_detected > 0}
            if not step_videos:
                self.logger.info("No Step 2 data found")
                return

            self.logger.info(f"\n🎯 STEP 2: TAG DETECTION")
            self.logger.info(f"  Total Videos: {len(step_videos)}")
            total_rings = sum(v.step2_ring_crops_count for v in step_videos.values())
            total_tags = sum(v.step2_tags_detected for v in step_videos.values())
            self.logger.info(f"  Total Ring Crops: {total_rings}")
            self.logger.info(f"  Total Tags Detected: {total_tags}")
            avg_tags_per_ring = total_tags / total_rings if total_rings > 0 else 0
            self.logger.info(f"  Avg Tags per Ring: {avg_tags_per_ring:.2f}")

            self.logger.info(f"\n📝 TOP 15 VIDEOS BY TAG DETECTIONS")
            sorted_videos = sorted(step_videos.items(),
                                 key=lambda x: x[1].step2_tags_detected, reverse=True)
            for i, (stem, v) in enumerate(sorted_videos[:15], 1):
                status = v.step2_status if v.step2_status else "?"
                time_s = f"{v.step2_time:.1f}s" if v.step2_time else "?"
                self.logger.info(
                    f"  {i:2d}. {v.camera:15s} {stem:30s} | "
                    f"Rings: {v.step2_ring_crops_count:4d}, Tags: {v.step2_tags_detected:4d} [{status:7s}] {time_s}"
                )

        elif step == "step3":
            step_videos = {s: v for s, v in videos.items() if v.step3_final_tag}
            if not step_videos:
                self.logger.info("No Step 3 data found")
                return

            self.logger.info(f"\n🎯 STEP 3: TEXT RECOGNITION (OCR)")
            self.logger.info(f"  Total Videos with Results: {len(step_videos)}")
            self.logger.info(f"  Total OCR Matches: {sum(v.step3_ocr_matches for v in step_videos.values())}")

            self.logger.info(f"\n📝 OCR RESULTS")
            sorted_videos = sorted(step_videos.items(),
                                 key=lambda x: x[0])
            for i, (stem, v) in enumerate(sorted_videos[:20], 1):
                status = v.step3_status if v.step3_status else "?"
                time_s = f"{v.step3_time:.1f}s" if v.step3_time else "?"
                letters_c = f"{v.step3_letters_conf:.3f}" if v.step3_letters_conf else "?"
                numbers_c = f"{v.step3_numbers_conf:.3f}" if v.step3_numbers_conf else "?"
                self.logger.info(
                    f"  {i:2d}. {v.camera:15s} | Final Tag: {v.step3_final_tag:20s} | "
                    f"L:{letters_c} N:{numbers_c} [{status:7s}] {time_s}"
                )
            if len(step_videos) > 20:
                self.logger.info(f"  ... and {len(step_videos) - 20} more")

        self.logger.info("=" * 120)

    def print_combined_report(self) -> None:
        """Print comprehensive combined report."""
        videos = self.scan_all_videos()

        self.logger.info("")
        self.logger.info("=" * 120)
        self.logger.info("COMPREHENSIVE DIAGNOSTICS: ALL STEPS COMBINED")
        self.logger.info("=" * 120)

        self.logger.info(f"\n📊 OVERALL SUMMARY")
        self.logger.info(f"  Total Videos: {len(videos)}")

        # Count completions by status
        step1_complete = sum(1 for v in videos.values() if v.step1_ring_detections > 0)
        step2_complete = sum(1 for v in videos.values() if v.step2_tags_detected > 0)
        step3_complete = sum(1 for v in videos.values() if v.step3_final_tag)
        all_complete = sum(1 for v in videos.values() if
                          v.step1_ring_detections > 0 and
                          v.step2_tags_detected > 0 and
                          v.step3_final_tag)

        self.logger.info(f"  Step 1 Complete: {step1_complete}/{len(videos)} ({step1_complete/len(videos)*100:.1f}%)")
        self.logger.info(f"  Step 2 Complete: {step2_complete}/{len(videos)} ({step2_complete/len(videos)*100:.1f}%)")
        self.logger.info(f"  Step 3 Complete: {step3_complete}/{len(videos)} ({step3_complete/len(videos)*100:.1f}%)")
        self.logger.info(f"  All Steps Complete: {all_complete}/{len(videos)} ({all_complete/len(videos)*100:.1f}%)")

        # Aggregate metrics
        self.logger.info(f"\n📈 AGGREGATE METRICS")
        total_detections = sum(v.step1_ring_detections for v in videos.values())
        total_tags = sum(v.step2_tags_detected for v in videos.values())
        total_ocr = sum(v.step3_ocr_matches for v in videos.values())

        self.logger.info(f"  Step 1 - Total Ring Detections: {total_detections}")
        self.logger.info(f"  Step 2 - Total Tag Detections: {total_tags}")
        self.logger.info(f"  Step 3 - Total OCR Matches: {total_ocr}")

        # Processing times
        times_s1 = [v.step1_time for v in videos.values() if v.step1_time]
        times_s2 = [v.step2_time for v in videos.values() if v.step2_time]
        times_s3 = [v.step3_time for v in videos.values() if v.step3_time]

        self.logger.info(f"\n⏱️  PROCESSING TIMES")
        if times_s1:
            self.logger.info(f"  Step 1: {sum(times_s1):.1f}s total, {sum(times_s1)/len(times_s1):.1f}s avg")
        if times_s2:
            self.logger.info(f"  Step 2: {sum(times_s2):.1f}s total, {sum(times_s2)/len(times_s2):.1f}s avg")
        if times_s3:
            self.logger.info(f"  Step 3: {sum(times_s3):.1f}s total, {sum(times_s3)/len(times_s3):.1f}s avg")

        # Top performers
        if videos:
            self.logger.info(f"\n🏆 TOP PERFORMERS")
            by_detections = sorted(videos.items(),
                                 key=lambda x: x[1].step1_ring_detections, reverse=True)
            self.logger.info(f"  Most Detections: {by_detections[0][1].camera} - {by_detections[0][0]} "
                           f"({by_detections[0][1].step1_ring_detections} detections)")

            by_tags = sorted(videos.items(),
                           key=lambda x: x[1].step2_tags_detected, reverse=True)
            self.logger.info(f"  Most Tags: {by_tags[0][1].camera} - {by_tags[0][0]} "
                           f"({by_tags[0][1].step2_tags_detected} tags)")

            with_ocr = [v for v in videos.values() if v.step3_final_tag]
            if with_ocr:
                self.logger.info(f"  OCR Results: {len(with_ocr)} videos with final tags")

        self.logger.info("=" * 120)

    def save_final_summary(self, pipeline_logger) -> bool:
        """Persist the combined per-video metrics in the pipeline database."""
        summaries = []
        for metrics in self.scan_all_videos().values():
            summary = asdict(metrics)
            if not summary["video_path"]:
                summary["video_path"] = f"{metrics.camera}/{metrics.video_stem}"
            summaries.append(summary)
        return pipeline_logger.replace_final_summary(summaries)
