import argparse
import cv2 as cv

import ring_detector
import tag_detector
import text_recognition


from dataclasses import astuple
from datetime import timedelta
import errno
from functools import partial
import logging
from multiprocessing import Pool
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
    #output_rel_path: Path,
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
    logging.info(f"pid: {os.getpid()} | Start prediction on: {video_path}")

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

    # detections = {}
    idx = -1
    # found_tag = False
    _ring_types = config.RingTypes()

    output_rel_path = Path(camera).joinpath(video_path.stem)
    tmp_path = output_root_path.joinpath("tmp", output_rel_path)
    
    if not tmp_path.exists():
        tmp_path.mkdir(parents=True, exist_ok=False)

    try:
        roi = None
        #cam = output_rel_path.parts[0]
        if camera in ["FAR6_SCALE", "FAR3_SCALE", "ROST2_SCALE", "FAR3_CUTE"]:
            #roi = _rois[output_rel_path.parts[0]]   
            roi = _rois[camera]  


        # found_tag = False
        bbs = []
        tags = []

        pred_letters_ema = np.zeros(4)
        pred_numbers_ema = np.zeros(4)
        alpha = 0.9

        #while True:
        while cap.isOpened():
            #gop = _video_coder.get_gop()
            #if gop is None:
            #    break
            ret, image = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            
            #for image in gop:
            idx += 1

            bb = _ring_detector.predict_cls(
                image, _ring_types.id(_ring_types.METAL_RING)
            )
            if bb is None:
                continue

            if bb.ndim == 1:
                bb=np.expand_dims(bb, axis=0)        

            if roi is None:
                bb = np.squeeze(bb[0])
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

            image_pil, image_pil_crop = _crop_image(image, bb)

            tag_images = _tag_detector.predict_tag(image_pil_crop)
            
            tag = _text_recognition.predict_tag(tag_images)

            #print(idx, type(tag), type(tag_images))
            #continue
            
            if len(tag) != 2: # and len(tag_r180) != 2:
                pred_letters_ema *= (1-alpha)
                pred_numbers_ema *= (1-alpha)
                continue

    
            # Rotate 180 deg        
            tag_images_r180 = []
            for ti in tag_images:
                tag_images_r180.append(ti.transpose(Image.ROTATE_180))
            tag_r180 = _text_recognition.predict_tag(tag_images_r180)

            if len(tag_r180) == 2: # and len(tag_r180) != 2:
                tag_min = np.min([tag["letters"]["pred"], tag["numbers"]["pred"]])
                tag_rot180_min = np.min([tag_r180["letters"]["pred"], tag_r180["numbers"]["pred"]])

                #print("---")
                #print(tag["letters"]["str"], tag["numbers"]["str"], [tag["letters"]["pred"], tag["numbers"]["pred"]])
                #print(tag_r180["letters"]["str"], tag_r180["numbers"]["str"], [tag_r180["letters"]["pred"], tag_r180["numbers"]["pred"]])

                if tag_min<tag_rot180_min:
                    tag = tag_r180
                    tag = tag_r180


            pred_letters_ema = alpha*tag["letters"]["pred"]+ (1-alpha)*pred_letters_ema
            pred_numbers_ema = alpha*tag["numbers"]["pred"]+ (1-alpha)*pred_numbers_ema

            #print("letters:", pred_letters_ema, np.min(pred_letters_ema), text_recognition_conf, str(tag["letters"]["str"]))
            #print("numbers:", pred_numbers_ema, np.min(pred_numbers_ema), text_recognition_conf, str(tag["numbers"]["str"]))

            if (np.min(pred_letters_ema) < text_recognition_conf) or (np.min(pred_numbers_ema) < text_recognition_conf):
                continue


            use_bb = True
            bbl = [
                bb[0],
                bb[1],
                bb[0] + bb[2],
                bb[1] + bb[3],
            ]

            for ii, bbi in enumerate(bbs):
                if metrics.IOU(bbl, bbi) > 0.9:
                    if tags[ii]["letters"]["str"] == tag["letters"]["str"] and tags[ii]["numbers"]["str"] == tag["numbers"]["str"]:
                        use_bb = False
                        break
            if not use_bb:
                continue

            bbs.append(bbl)
            tags.append(tag)

            # print(idx, tag["letters"]["str"], tag["numbers"]["str"])

            # output_path = output_root_path.joinpath(video_path.stem)
            # output_path = output_root_path

            # todo: error if exist
            # if not output_path.exists():
            #    output_path.mkdir(parents=True, exist_ok=False)
            # if not output_path.exists():
            #     output_path.mkdir(parents=True, exist_ok=False)

            name_stem = video_path.stem  # + "_" + str(idx)

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

            # if save_tag_image:  
            #     tag_images[0].save(
            #     tmp_path.joinpath(name_stem + "_tag0_" + str(idx) + ".jpg"),
            #     quality=100,
            #     subsampling=0,
            # )
            #     tag_images[1].save(
            #     tmp_path.joinpath(name_stem + "_tag1_" + str(idx) + ".jpg"),
            #     quality=100,
            #     subsampling=0,
            # )


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

            # #print(idx)
            #print(idx, tag["letters"]["str"],tag["numbers"]["str"])
            # #print(tag["letters"]["pred"])
            # #print(tag["numbers"]["pred"])

            save_yaml(
                output_yaml_dict,
                tmp_path.joinpath(name_stem + "_" + str(idx) + ".yaml"),
            )

        tag_strs = []
        for ti in tags:
            #print(ti["letters"]["str"], ti["numbers"]["str"])
            tag_strs.append(ti["letters"]["str"] + ti["numbers"]["str"])

        #tag_strs_unique, tag_strs_index, utag_strs_inverse, tag_strs_count,  = np.unique(tag_strs, return_index=True, unique_inverse= True, return_counts=True)
        tag_strs_unique, tag_strs_index, utag_strs_inverse, tag_strs_count,  = np.unique(tag_strs, return_index=True, return_inverse=True, return_counts=True,)
        if len(tag_strs_unique) >0:

            idx_max = np.argmax(tag_strs_count)
            # print(tag_strs_unique)
            # print(tag_strs_index)
            # print(utag_strs_inverse)
            # print(tag_strs_count)
             
            # print("Sel:", tag_strs_unique[idx_max])
        
            output_yaml_dict = {
                "video_rel_path": str(video_path),
                "number_of_detections": int(tag_strs_count[idx_max]),
                "str1": tag_strs_unique[idx_max][0:3],
                "str2": tag_strs_unique[idx_max][3:6],
            }

            save_yaml(
                output_yaml_dict,
                tmp_path.joinpath(name_stem + ".yaml"),
            )

            output_path = output_root_path.joinpath(output_rel_path)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=False)

            tmp_path.rename(output_path)

        else:
            shutil.rmtree(tmp_path)    


    #except mpegCoderDecoder:
    #    pass
    except Exception as err:
        logging.critical(
            f"pid: {os.getpid()} | _ring_tag_read : Exception: TODO: implement this specific exception: {err.__class__.__name__}, error: {err}, video path {video_path}"
        )
        # print("bb", bb)
        # print("tags", tags)
        # print("s", s)
    
    if tmp_path in locals():
        shutil.rmtree(tmp_path)    
    

    time_end = time.time()
    td = timedelta(seconds=time_end - time_begin)
    logging.info(f"pid: {os.getpid()} | Done predict on: {video_path}, time: {td}")


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
    
    remove_and_make_dir(output_root_path,remove=True)   

    time_begin = time.time()

    logger = Logger(
        root_path=Path.cwd().joinpath("log"),
        name="ring_tag_read_log",
        log_to_stream=True,
    )
    logger.start()

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

        # video_path_list = video_path_list[0:10]
        # for vpi in video_path_list:
        #     print(vpi[0],vpi[1])
        # sys.exit()    

       
        tmp_root_path = output_root_path.joinpath("tmp")
        if tmp_root_path.exists():
            shutil.rmtree(tmp_root_path)

        logging.info(f"Start multiprocessing")
        with Pool(multiprocessing_n_processes, maxtasksperchild=1) as p:
            ret = p.starmap(
                partial(
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
                ),
                video_path_list,
                #video_path_list[0:1],
            )

    # KeyboardInterrupt
    except FileNotFoundError:
        pass
    except Exception as err:
        logging.critical(f"Exception: ring_tag_read, {err.__class__.__name__}, Error: {err}")

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
