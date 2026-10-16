import argparse
import csv
import cv2 as cv

import ring_detector
import tag_detector
import text_recognition


from dataclasses import astuple
from datetime import timedelta
import errno
from functools import partial
import logging
from multiprocessing import get_context
import numpy as np
import os
from pathlib import Path
import yaml
import metrics
import shutil


import sys
import time

import torch
from ultralytics import YOLO

from PIL import Image

module_path = Path(__file__).resolve().parent.joinpath("CLIP4STR")

sys.path.append(str(module_path))

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint

from annotations import open_yaml
from logger import Logger

import config

import re

import pandas as pd


from shapely import geometry

from dataclasses import dataclass, asdict

from event_logger import EventLogger

save_tag_image = True

@dataclass(frozen=True)
class ROIS:
    FAR3_CUTE: geometry.Polygon = geometry.Polygon(
        [
            (1318, 726),
            (1562.167, 840.833),
            (830.167, 1036.833),
            (828.833, 707.500),
        ]
    )
    FAR3_SCALE: geometry.Polygon = geometry.Polygon(
        [
            (388, 782),
            (1419.500, 1315.500),
            (1855.500, 1083.500),
            (1281.500, 277.500),
        ]
    )
    FAR6_SCALE: geometry.Polygon = geometry.Polygon(
        [
            (218, 1090),
            (1049.500, 1251.500),
            (1483.500, 637.500),
            (471.500, 177.500),
        ]
    )
    ROST2_SCALE: geometry.Polygon = geometry.Polygon(
        [
            (248, 1144),
            (733.500, 1435.500),
            (1979.500, 1273.500),
            (1051.500, 265.500),
        ]
    )


# print(asdict(ROIS()))
_rois = asdict(ROIS())

_wc = 224
_hc = 224


_VIDEO_STATS_FIELDS = [
    "video_path",
    "camera",
    "output_rel_path",
    "status",
    "error_message",
    "processing_seconds",
    "frames_total",
    "ring_detect_total",
    "ring_detect_roi",
    "ring_detection_rate",
    "ring_conf_mean",
    "ring_conf_max",
    "tag_detect_total",
    "tag_detect_frames",
    "tag_detection_rate",
    "tag_conf_mean",
    "tag_conf_max",
    "ocr_candidates",
    "ocr_candidate_letters_min_mean",
    "ocr_candidate_numbers_min_mean",
    "ocr_final_accepts",
    "ocr_success_rate",
    "ocr_final_letters_min_mean",
    "ocr_final_numbers_min_mean",
    "ocr_final_letters_ema_min_mean",
    "ocr_final_numbers_ema_min_mean",
    "final_tag",
    "final_tag_count",
    "final_letters_conf_min",
    "final_numbers_conf_min",
    "final_letters_ema_min",
    "final_numbers_ema_min",
]


def _safe_avg(total: float, count: int):
    return total / count if count else None


def _safe_ratio(num: float, denom: float):
    return num / denom if denom else None


def _format_metric(value, digits: int = 4):
    if value is None:
        return ""
    return round(value, digits)


def _init_video_stats(video_path: Path, camera: str, output_rel_path: Path):
    return {
        "video_path": str(video_path),
        "camera": camera,
        "output_rel_path": str(output_rel_path),
        "status": "success",
        "error_message": "",
        "processing_seconds": 0.0,
        "frames_total": 0,
        "ring_detect_total": 0,
        "ring_detect_roi": 0,
        "ring_conf_sum": 0.0,
        "ring_conf_count": 0,
        "ring_conf_max": None,
        "tag_detect_total": 0,
        "tag_detect_frames": 0,
        "tag_conf_sum": 0.0,
        "tag_conf_count": 0,
        "tag_conf_max": None,
        "ocr_candidates": 0,
        "ocr_candidate_letters_min_sum": 0.0,
        "ocr_candidate_numbers_min_sum": 0.0,
        "ocr_final_accepts": 0,
        "ocr_final_letters_min_sum": 0.0,
        "ocr_final_numbers_min_sum": 0.0,
        "ocr_final_letters_ema_min_sum": 0.0,
        "ocr_final_numbers_ema_min_sum": 0.0,
        "final_tag": "",
        "final_tag_count": 0,
        "final_letters_conf_min": None,
        "final_numbers_conf_min": None,
        "final_letters_ema_min": None,
        "final_numbers_ema_min": None,
    }


def _build_stats_row(stats: dict):
    ring_conf_mean = _safe_avg(stats.get("ring_conf_sum", 0.0), stats.get("ring_conf_count", 0))
    tag_conf_mean = _safe_avg(stats.get("tag_conf_sum", 0.0), stats.get("tag_conf_count", 0))
    ocr_candidate_letters_min_mean = _safe_avg(
        stats.get("ocr_candidate_letters_min_sum", 0.0), stats.get("ocr_candidates", 0)
    )
    ocr_candidate_numbers_min_mean = _safe_avg(
        stats.get("ocr_candidate_numbers_min_sum", 0.0), stats.get("ocr_candidates", 0)
    )
    ocr_final_letters_min_mean = _safe_avg(
        stats.get("ocr_final_letters_min_sum", 0.0), stats.get("ocr_final_accepts", 0)
    )
    ocr_final_numbers_min_mean = _safe_avg(
        stats.get("ocr_final_numbers_min_sum", 0.0), stats.get("ocr_final_accepts", 0)
    )
    ocr_final_letters_ema_min_mean = _safe_avg(
        stats.get("ocr_final_letters_ema_min_sum", 0.0), stats.get("ocr_final_accepts", 0)
    )
    ocr_final_numbers_ema_min_mean = _safe_avg(
        stats.get("ocr_final_numbers_ema_min_sum", 0.0), stats.get("ocr_final_accepts", 0)
    )

    row = {
        "video_path": stats.get("video_path", ""),
        "camera": stats.get("camera", ""),
        "output_rel_path": stats.get("output_rel_path", ""),
        "status": stats.get("status", "unknown"),
        "error_message": stats.get("error_message", ""),
        "processing_seconds": _format_metric(stats.get("processing_seconds", None), digits=3),
        "frames_total": stats.get("frames_total", 0),
        "ring_detect_total": stats.get("ring_detect_total", 0),
        "ring_detect_roi": stats.get("ring_detect_roi", 0),
        "ring_detection_rate": _format_metric(
            _safe_ratio(stats.get("ring_detect_roi", 0), stats.get("frames_total", 0))
        ),
        "ring_conf_mean": _format_metric(ring_conf_mean),
        "ring_conf_max": _format_metric(stats.get("ring_conf_max")),
        "tag_detect_total": stats.get("tag_detect_total", 0),
        "tag_detect_frames": stats.get("tag_detect_frames", 0),
        "tag_detection_rate": _format_metric(
            _safe_ratio(stats.get("tag_detect_frames", 0), stats.get("ring_detect_roi", 0))
        ),
        "tag_conf_mean": _format_metric(tag_conf_mean),
        "tag_conf_max": _format_metric(stats.get("tag_conf_max")),
        "ocr_candidates": stats.get("ocr_candidates", 0),
        "ocr_candidate_letters_min_mean": _format_metric(ocr_candidate_letters_min_mean),
        "ocr_candidate_numbers_min_mean": _format_metric(ocr_candidate_numbers_min_mean),
        "ocr_final_accepts": stats.get("ocr_final_accepts", 0),
        "ocr_success_rate": _format_metric(
            _safe_ratio(stats.get("ocr_final_accepts", 0), stats.get("tag_detect_frames", 0))
        ),
        "ocr_final_letters_min_mean": _format_metric(ocr_final_letters_min_mean),
        "ocr_final_numbers_min_mean": _format_metric(ocr_final_numbers_min_mean),
        "ocr_final_letters_ema_min_mean": _format_metric(ocr_final_letters_ema_min_mean),
        "ocr_final_numbers_ema_min_mean": _format_metric(ocr_final_numbers_ema_min_mean),
        "final_tag": stats.get("final_tag", ""),
        "final_tag_count": stats.get("final_tag_count", 0),
        "final_letters_conf_min": _format_metric(stats.get("final_letters_conf_min")),
        "final_numbers_conf_min": _format_metric(stats.get("final_numbers_conf_min")),
        "final_letters_ema_min": _format_metric(stats.get("final_letters_ema_min")),
        "final_numbers_ema_min": _format_metric(stats.get("final_numbers_ema_min")),
    }

    return row


def _write_video_stats_csv(output_root_path: Path, rows: list):
    rows_to_write = [row for row in rows if row]
    if not rows_to_write:
        return

    csv_path = output_root_path.joinpath("video_stats.csv")
    with csv_path.open(mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_VIDEO_STATS_FIELDS)
        writer.writeheader()
        for row in rows_to_write:
            writer.writerow(row)

def remove_and_make_dir(dir_path: Path, remove: bool = False):
    if dir_path.exists() and remove:
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print(f"Error: remove {str(dir_path)}, {e.strerror}")
            sys.exit(1)
    try:
        dir_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as e:
        print(f"Error: mkdir {str(dir_path)}, {e.strerror}")
        sys.exit(1)


def open_yaml(yaml_path: Path) -> dict:
    try:
        with open(yaml_path, "r") as yaml_file:
            yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        raise e
    return yaml_dict


def save_yaml(yaml_dict: dict, yaml_path: Path, default_flow_style=False):
    try:
        with open(yaml_path, mode="wt", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_flow_style=default_flow_style,
            )

    except yaml.YAMLError as e:
        raise e


def _crop_image(
    image: np.ndarray,
    bb: np.ndarray,
    wc: int = _wc,
    hc: int = _hc,
):
    assert bb.ndim == 1
    assert bb.shape[0] == 4

    image_pil = Image.fromarray(image)
    w, h = image_pil.size
    xc = np.clip(bb[0] * w - 0.5 * wc, 0, w - wc - 1)
    yc = np.clip(bb[1] * h - 0.5 * hc, 0, h - hc - 1)

    return image_pil, image_pil.crop((xc, yc, xc + _wc, yc + _hc))


def parse_args():
    argparser = argparse.ArgumentParser(description="A script to ...")
    argparser.add_argument(
        "batch_file",
        default=None,
        nargs="?",
        type=Path,
        help="Batch yaml-file.",
    )

    return argparser.parse_args()


def _ring_tag_read(
    video_path: Path,
    camera: str,
    output_root_path: Path,
    ring_detection_model_root_path: Path,
    ring_detection_model_name: str,
    tag_detection_model_root_path: Path,
    tag_detection_model_name: str,
    text_recognition_model_root_path: Path,
    text_recognition_model_name: str,
    ring_detection_conf: float = 0.7,
    ring_detection_iou: float = 0.7,
    ring_detection_max_det: int = 1,
    tag_detection_conf: float = 0.7,
    tag_detection_iou: float = 0.7,
    tag_detection_max_det: int = 1,
    text_recognition_conf: float = 0.9,
):

    time_begin = time.time()
    logging.info(f"pid: {os.getpid()} | WORKER START: {video_path}")
    logging.info(f"pid: {os.getpid()} | Start prediction on: {video_path}")

    # Initialize to None so they're accessible in finally block
    cap = None
    _ring_detector = None
    _tag_detector = None
    _text_recognition = None
    tmp_path = None

    cap = cv.VideoCapture(str(video_path))
    _ring_detector = ring_detector.RingDetector(
        model_path=ring_detection_model_root_path.joinpath(ring_detection_model_name),
        stream=True,
        conf=ring_detection_conf,
        iou=ring_detection_iou,
        max_det=ring_detection_max_det,
    )
    _tag_detector = tag_detector.TagDetector(
        model_path=tag_detection_model_root_path.joinpath(tag_detection_model_name),
        stream=True,
        conf=tag_detection_conf,
        iou=tag_detection_iou,
        max_det=tag_detection_max_det,
    )
    #_text_recognition = text_recognition2.TextRecognition(
    _text_recognition = text_recognition.TextRecognition(
        model_path=text_recognition_model_root_path.joinpath(
            text_recognition_model_name
        ),
        conf=text_recognition_conf,
    )

    idx = -1
    _ring_types = config.RingTypes()

    output_rel_path = Path(camera).joinpath(video_path.stem)
    tmp_path = output_root_path.joinpath("tmp", output_rel_path)
    if not tmp_path.exists():
        tmp_path.mkdir(parents=True, exist_ok=False)

    stats = _init_video_stats(video_path, camera, output_rel_path)
    bbs = []
    tags = []

    pred_letters_ema = np.zeros(4)
    pred_numbers_ema = np.zeros(4)
    alpha = 0.9

    try:
        roi = None
        if camera in ["FAR6_SCALE", "FAR3_SCALE", "ROST2_SCALE", "FAR3_CUTE"]:
            roi = _rois[camera]

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            stats["frames_total"] += 1
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            idx += 1

            ring_pred = _ring_detector.predict_cls(
                image, _ring_types.id(_ring_types.METAL_RING)
            )
            if ring_pred is None:
                continue

            bb, ring_conf = ring_pred
            stats["ring_detect_total"] += 1
            if ring_conf is not None:
                stats["ring_conf_sum"] += ring_conf
                stats["ring_conf_count"] += 1
                stats["ring_conf_max"] = (
                    ring_conf
                    if stats["ring_conf_max"] is None
                    else max(stats["ring_conf_max"], ring_conf)
                )

            if bb.ndim == 1:
                bb = np.expand_dims(bb, axis=0)

            if roi is None:
                bb = np.squeeze(bb[0])
                stats["ring_detect_roi"] += 1
            else:
                (height, width, _) = image.shape
                ring_in_roi = False
                for bbi in bb:
                    xmin = (bbi[0] - 0.5 * bbi[2]) * width
                    ymin = (bbi[1] - 0.5 * bbi[3]) * height
                    xmax = (bbi[0] + 0.5 * bbi[2]) * width
                    ymax = (bbi[1] + 0.5 * bbi[3]) * height

                    bbp = geometry.Polygon(
                        [
                            (xmin, ymin),
                            (xmax, ymin),
                            (xmax, ymax),
                            (xmin, ymax),
                        ]
                    )
                    if roi.contains(bbp):
                        ring_in_roi = True
                        bb = np.squeeze(bbi)
                        break
                if not ring_in_roi:
                    continue
                stats["ring_detect_roi"] += 1

            image_pil, image_pil_crop = _crop_image(image, bb)

            tag_entries = _tag_detector.predict_tag_with_meta(image_pil_crop)
            tag_images = [entry[0] for entry in tag_entries]
            tag_confidences = [entry[1] for entry in tag_entries if entry[1] is not None]

            stats["tag_detect_total"] += len(tag_images)
            if tag_images:
                stats["tag_detect_frames"] += 1
            if tag_confidences:
                stats["tag_conf_sum"] += sum(tag_confidences)
                stats["tag_conf_count"] += len(tag_confidences)
                tag_conf_max = max(tag_confidences)
                stats["tag_conf_max"] = (
                    tag_conf_max
                    if stats["tag_conf_max"] is None
                    else max(stats["tag_conf_max"], tag_conf_max)
                )

            tag = _text_recognition.predict_tag(tag_images)

            if len(tag) != 2:
                pred_letters_ema *= (1 - alpha)
                pred_numbers_ema *= (1 - alpha)
                continue

            tag_images_r180 = [ti.transpose(Image.ROTATE_180) for ti in tag_images]
            tag_r180 = _text_recognition.predict_tag(tag_images_r180)

            if len(tag_r180) == 2:
                tag_min = np.min([tag["letters"]["pred"], tag["numbers"]["pred"]])
                tag_rot180_min = np.min(
                    [tag_r180["letters"]["pred"], tag_r180["numbers"]["pred"]]
                )
                if tag_min < tag_rot180_min:
                    tag = tag_r180

            stats["ocr_candidates"] += 1
            letters_pred_min = float(np.min(tag["letters"]["pred"]))
            numbers_pred_min = float(np.min(tag["numbers"]["pred"]))
            stats["ocr_candidate_letters_min_sum"] += letters_pred_min
            stats["ocr_candidate_numbers_min_sum"] += numbers_pred_min

            pred_letters_ema = alpha * tag["letters"]["pred"] + (1 - alpha) * pred_letters_ema
            pred_numbers_ema = alpha * tag["numbers"]["pred"] + (1 - alpha) * pred_numbers_ema

            if (np.min(pred_letters_ema) < text_recognition_conf) or (
                np.min(pred_numbers_ema) < text_recognition_conf
            ):
                continue

            letters_ema_min = float(np.min(pred_letters_ema))
            numbers_ema_min = float(np.min(pred_numbers_ema))

            use_bb = True
            bbl = [
                bb[0],
                bb[1],
                bb[0] + bb[2],
                bb[1] + bb[3],
            ]

            for ii, bbi in enumerate(bbs):
                if metrics.IOU(bbl, bbi) > 0.9:
                    if (
                        tags[ii]["letters"]["str"] == tag["letters"]["str"]
                        and tags[ii]["numbers"]["str"] == tag["numbers"]["str"]
                    ):
                        use_bb = False
                        break
            if not use_bb:
                continue

            stats["ocr_final_accepts"] += 1
            stats["ocr_final_letters_min_sum"] += letters_pred_min
            stats["ocr_final_numbers_min_sum"] += numbers_pred_min
            stats["ocr_final_letters_ema_min_sum"] += letters_ema_min
            stats["ocr_final_numbers_ema_min_sum"] += numbers_ema_min

            tag["letters"]["ema_min"] = letters_ema_min
            tag["numbers"]["ema_min"] = numbers_ema_min

            bbs.append(bbl)
            tags.append(tag)

            name_stem = video_path.stem

            image_pil.save(
                tmp_path.joinpath(name_stem + "_" + str(idx) + ".jpg"),
                quality=100,
                subsampling=0,
            )

            image_pil_crop.save(
                tmp_path.joinpath(name_stem + "_" + str(idx) + "_crop.jpg"),
                quality=100,
                subsampling=0,
            )

            output_yaml_dict = {
                "video_path": str(video_path),
                "image_idx": idx,
                "tag": {
                    "str1": str(tag["letters"]["str"]),
                    "pred1": tag["letters"]["pred"].tolist(),
                    "str2": str(tag["numbers"]["str"]),
                    "pred2": tag["numbers"]["pred"].tolist(),
                },
            }

            save_yaml(
                output_yaml_dict,
                tmp_path.joinpath(name_stem + "_" + str(idx) + ".yaml"),
            )

        tag_strs = [ti["letters"]["str"] + ti["numbers"]["str"] for ti in tags]

        if len(tag_strs) > 0:
            tag_strs_unique, tag_strs_index, _, tag_strs_count = np.unique(
                tag_strs,
                return_index=True,
                return_inverse=True,
                return_counts=True,
            )

            idx_max = np.argmax(tag_strs_count)

            stats["final_tag"] = tag_strs_unique[idx_max]
            stats["final_tag_count"] = int(tag_strs_count[idx_max])

            representative_tag = tags[tag_strs_index[idx_max]]
            stats["final_letters_conf_min"] = float(
                np.min(representative_tag["letters"]["pred"])
            )
            stats["final_numbers_conf_min"] = float(
                np.min(representative_tag["numbers"]["pred"])
            )
            stats["final_letters_ema_min"] = representative_tag["letters"].get(
                "ema_min"
            )
            stats["final_numbers_ema_min"] = representative_tag["numbers"].get(
                "ema_min"
            )

            output_yaml_dict = {
                "video_rel_path": str(video_path),
                "number_of_detections": int(tag_strs_count[idx_max]),
                "str1": tag_strs_unique[idx_max][0:3],
                "str2": tag_strs_unique[idx_max][3:6],
            }

            name_stem = video_path.stem
            save_yaml(
                output_yaml_dict,
                tmp_path.joinpath(name_stem + ".yaml"),
            )

            output_path = output_root_path.joinpath(output_rel_path)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=False)

            tmp_path.rename(output_path)
        else:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)

    except Exception as err:
        stats["status"] = "error"
        stats["error_message"] = str(err)
        logging.critical(
            f"pid: {os.getpid()} | _ring_tag_read : Exception: TODO: implement this specific exception: {err.__class__.__name__}, error: {err}, video path {video_path}"
        )
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        if tmp_path is not None and tmp_path.exists():
            shutil.rmtree(tmp_path)

    time_end = time.time()
    td = timedelta(seconds=time_end - time_begin)
    stats["processing_seconds"] = td.total_seconds()
    logging.info(f"pid: {os.getpid()} | Done predict on: {video_path}, time: {td}")

    # Build stats_row - wrapped in try-catch to guarantee return value
    try:
        stats_row = _build_stats_row(stats)
    except Exception as build_err:
        logging.error(f"pid: {os.getpid()} | Error building stats_row: {build_err}", exc_info=True)
        # Fallback: return minimal stats_row
        stats_row = {
            "video_path": str(video_path),
            "camera": stats.get("camera", "unknown"),
            "status": "error",
            "error_message": f"Build stats failed: {str(build_err)}",
            "processing_seconds": stats["processing_seconds"],
        }

    logging.info(f"pid: {os.getpid()} | WORKER RETURNING: video_path={stats_row.get('video_path')}, status={stats_row.get('status')}")
    return stats_row


def ring_tag_read(
    video_root_path: Path,
    #video_rel_paths: list,
    cameras: list,
    output_root_path: Path,
    ring_detection_model_root_path: Path,
    ring_detection_model_name: Path,
    tag_detection_model_root_path: Path,
    tag_detection_model_name: Path,
    text_recognition_model_root_path: Path,
    text_recognition_model_name: Path,
    #video_coder_dump_level: int = config.Defaults.VIDEO_CODER_DUMP_LEVEL,
    ring_detection_conf: float = 0.7,
    ring_detection_iou: float = 0.7,
    ring_detection_max_det: int = 1,
    tag_detection_conf: float = 0.7,
    tag_detection_iou: float = 0.7,
    tag_detection_max_det: int = 1,
    text_recognition_conf: float = 0.7,
    multiprocessing_n_processes: int = config.Defaults.N_PROCESSES,
):

    # Create output directory only if it doesn't exist (to allow resuming)
    output_root_path.mkdir(parents=True, exist_ok=True)

    time_begin = time.time()

    logger = Logger(
        root_path=Path.cwd().joinpath("log"),
        name="ring_tag_read_log",
        log_to_stream=True,
    )
    logger.start()

    # Initialize event logger for SQLite database logging
    db_path = Path.cwd().joinpath("log").joinpath("events.db")
    event_logger = EventLogger(db_path)

    logging.info(f"Start ring tag reading")
    logging.info(f"Video root path: {video_root_path}")
    logging.info(f"Cameras: {cameras}")
    logging.info(f"Output root path: {output_root_path}")
    logging.info(f"Ring detection model root path: {ring_detection_model_root_path}")
    logging.info(f"Ring detection model name: {ring_detection_model_name}")
    logging.info(f"Tag detection model root path: {tag_detection_model_root_path}")
    logging.info(f"Tag detection model name: {tag_detection_model_name}")
    logging.info(
        f"Text recognition model root path: {text_recognition_model_root_path}"
    )
    logging.info(f"Text recognition model name: {text_recognition_model_name}")
    # TODO: log all input variables

    if not video_root_path.exists():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), str(video_root_path)
        )

    try:

        video_path_list = []

        for video_path in video_root_path.glob("**/*.mp4"):
            video_path_stem = video_path.stem
            camera = None
            for ci in cameras: #this is utterly d
                if video_path_stem.find(ci)>=0:
                    camera = ci
                    break
            if camera is None:
                logging.warning(f"No camera match for: {video_path}")
                continue


            video_path_list.append((video_path, camera))

        # Get already processed videos to skip them
        logging.info(f"DEBUG: Total videos after camera matching: {len(video_path_list)}")
        if video_path_list:
            logging.debug(f"DEBUG: First 3 videos: {[str(vp) for vp, _ in video_path_list[:3]]}")

        processed_videos = event_logger.get_processed_videos(status="success")
        logging.info(f"DEBUG: Previously processed videos in DB: {len(processed_videos)}")
        if processed_videos:
            logging.debug(f"DEBUG: Sample processed videos (first 3): {list(processed_videos)[:3]}")

        original_count = len(video_path_list)

        # Filter out previously processed videos
        video_path_list = [
            (vp, cam) for vp, cam in video_path_list
            if str(vp) not in processed_videos
        ]

        skipped_count = original_count - len(video_path_list)
        logging.info(f"DEBUG: Videos to process after filtering: {len(video_path_list)} (skipped {skipped_count})")
        if skipped_count > 0:
            logging.info(f"Skipping {skipped_count} previously processed videos")

        if not video_path_list:
            logging.info("No new videos to process")
            time_end = time.time()
            td = timedelta(seconds=time_end - time_begin)
            logging.info(f"Done: time: {td}")
            logger.stop()
            return
        tmp_root_path = output_root_path.joinpath("tmp")
        if tmp_root_path.exists():
            shutil.rmtree(tmp_root_path)

        logging.info(f"Start multiprocessing with {len(video_path_list)} videos to process")
        with get_context('spawn').Pool(multiprocessing_n_processes, maxtasksperchild=1) as p:
            ret = []
            video_path_by_row = {str(video_path): (video_path, camera) for video_path, camera in video_path_list}
            logging.info(f"DEBUG: video_path_by_row keys: {list(video_path_by_row.keys())[:3]}... ({len(video_path_by_row)} total)")

            # Use imap_unordered to process results as they complete
            worker = partial(
                _ring_tag_read,
                output_root_path=output_root_path,
                ring_detection_model_root_path=ring_detection_model_root_path,
                ring_detection_model_name=ring_detection_model_name,
                tag_detection_model_root_path=tag_detection_model_root_path,
                tag_detection_model_name=tag_detection_model_name,
                text_recognition_model_root_path=text_recognition_model_root_path,
                text_recognition_model_name=text_recognition_model_name,
                ring_detection_conf=ring_detection_conf,
                ring_detection_iou=ring_detection_iou,
                ring_detection_max_det=ring_detection_max_det,
                tag_detection_conf=tag_detection_conf,
                tag_detection_iou=tag_detection_iou,
                tag_detection_max_det=tag_detection_max_det,
                text_recognition_conf=text_recognition_conf,
            )

            result_count = 0
            logging.info(f"Starting to iterate over pool results...")

            try:
                # Use starmap to pass tuple arguments (video_path, camera)
                logging.info(f"[POOL] Calling p.starmap with {len(video_path_list)} tasks...")
                result_iter = p.starmap(worker, video_path_list)
                logging.info(f"[POOL] starmap iterator created, waiting for results...")

                for stats_row in result_iter:
                    result_count += 1
                    ret.append(stats_row)

                    # Log immediately as each video completes - THIS MUST NOT FAIL
                    try:
                        if stats_row and isinstance(stats_row, dict) and stats_row.get("video_path"):
                            video_path_str = stats_row.get("video_path")
                            logging.info(f"[RESULT {result_count}/{len(video_path_list)}] video={Path(video_path_str).name} status={stats_row.get('status')} time={stats_row.get('processing_seconds')}s")

                            # Attempt database logging - single responsibility
                            try:
                                event_logger.log_video_stats_row(stats_row)
                                logging.info(f"[RESULT {result_count}] ✓ Logged to database")
                            except Exception as log_error:
                                logging.error(f"[RESULT {result_count}] ✗ DB logging failed: {log_error}", exc_info=True)
                                # Don't fail entirely - try to continue
                        else:
                            logging.warning(f"[RESULT {result_count}] Invalid stats_row: {type(stats_row)} - {stats_row if not isinstance(stats_row, dict) else 'missing video_path'}")
                    except Exception as process_error:
                        logging.error(f"[RESULT {result_count}] Error processing result: {process_error}", exc_info=True)

                logging.info(f"[POOL] Iteration completed. Processed {result_count}/{len(video_path_list)} results total.")

            except Exception as pool_error:
                logging.critical(f"[POOL] ERROR during pool iteration: {pool_error.__class__.__name__}: {pool_error}", exc_info=True)
                logging.critical(f"[POOL] Results received before error: {result_count}")
                # Cleanup pool
                try:
                    p.terminate()
                    p.join()
                except Exception as cleanup_error:
                    logging.error(f"[POOL] Error during cleanup: {cleanup_error}")

            _write_video_stats_csv(output_root_path, ret)

    # KeyboardInterrupt
    except FileNotFoundError:
        pass
    except Exception as err:
        logging.critical(f"Exception: ring_tag_read, {err.__class__.__name__}, Error: {err}")
    finally:
        # Ensure event logger is properly closed before shutting down
        event_logger.close()

    time_end = time.time()
    td = timedelta(seconds=time_end - time_begin)
    logging.info(f"Done: time: {td}")
    logger.stop()


def main() -> int:
    args = parse_args()
    args_dict = vars(args)

    try:
        batch_dict = open_yaml(args_dict["batch_file"])
        # print(annotations.pretty_print_dict(batch_dict))
        # print(batch_dict)

        ring_detection_dict = batch_dict["models"]["ring_detection"]
        tag_detection_dict = batch_dict["models"]["tag_detection"]
        text_recognition_dict = batch_dict["models"]["text_recognition"]

        ring_tag_read(
            video_root_path=Path(batch_dict["input"]["video_root_path"]),
            #video_rel_paths=batch_dict["input"]["video_rel_paths"],
            cameras=batch_dict["input"]["cameras"],
            output_root_path=Path(batch_dict["output_root_path"]),
            ring_detection_model_root_path=Path(ring_detection_dict["model_root_path"]),
            ring_detection_model_name=ring_detection_dict["model_name"],
            tag_detection_model_root_path=Path(tag_detection_dict["model_root_path"]),
            tag_detection_model_name=tag_detection_dict["model_name"],
            text_recognition_model_root_path=Path(
                text_recognition_dict["model_root_path"]
            ),
            text_recognition_model_name=text_recognition_dict["model_name"],
            #video_coder_dump_level=batch_dict["video_coder"]["dump_level"],
            ring_detection_conf=ring_detection_dict["conf"],
            ring_detection_iou=ring_detection_dict["iou"],
            ring_detection_max_det=ring_detection_dict["max_det"],
            tag_detection_conf=tag_detection_dict["conf"],
            tag_detection_iou=tag_detection_dict["iou"],
            tag_detection_max_det=tag_detection_dict["max_det"],
            text_recognition_conf=text_recognition_dict["conf"],
            multiprocessing_n_processes=batch_dict["multiprocessing"]["n_processes"],
        )

    except KeyError as e:
        print(f"KeyError: {e}")

    return 0


if __name__ == "__main__":

    
    sys.exit(main())
