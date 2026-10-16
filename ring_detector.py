import logging
import numpy as np
import os
from pathlib import Path
import torch
from ultralytics import YOLO

class RingDetector:

    def __init__(
        self,
        model_path: Path,
        stream: bool = True,
        conf: float = 0.7,
        iou: float = 0.7,
        max_det: int = 1,
        device_id: int = 0,
        imgsz: int = 480,
    ):
        try:
            self.stream = stream
            self.conf = conf
            self.iou = iou
            self.max_det = max_det
            self.device_id = device_id
            self.imgsz = imgsz

            logging.info(
                f"pid: {os.getpid()} | Generate ring detection model {model_path} on GPU {device_id} (imgsz={imgsz})"
            )

            self.model = YOLO(str(model_path))
        except Exception as err:
            logging.critical(
                f"pid: {os.getpid()} | RingDetector, Exception: TODO: implement this specific exception: {err.__class__.__name__}, error: {err}"
            )

    def predict(self, image: np.ndarray):
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return self.model(
            image,
            stream=self.stream,
            conf=self.conf,
            iou=self.iou,
            show=False,
            save=False,
            save_txt=False,
            save_conf=False,
            max_det=self.max_det,
            verbose=False,
            device=f"cuda:{self.device_id}",
            half=True,
            imgsz=self.imgsz,
            agnostic_nms=True,
        )

    def predict_cls(self, image: np.ndarray, cls_target: int):
        try:
            results = self.predict(image=image)
            for res in results:
                boxes = res.boxes.cpu()
                if boxes is not None:
                    for box in boxes:
                        cls_pred = int(np.squeeze(np.array(box.cls)))
                        if cls_pred == cls_target:
                            xywhn = np.squeeze(np.array(box.xywhn.cpu()))
                            conf = float(np.squeeze(np.array(box.conf.cpu())))
                            return xywhn, conf
        except Exception as err:
            logging.critical(
                f"pid: {os.getpid()} | RingDetector, predict_cls: Exception: TODO: implement this specific exception: {err.__class__.__name__}, error: {err}"
            )
        return None
