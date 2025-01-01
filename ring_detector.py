import logging
import numpy as np
import os
from pathlib import Path
import torch
from ultralytics import YOLO

#torch.cuda.set_device(1)

class RingDetector:

    def __init__(
        self,
        model_path: Path,
        stream: bool = True,
        conf: float = 0.7,
        iou: float = 0.7,
        max_det: int = 1,
    ):
        try:
            self.stream = stream
            self.conf = conf
            self.iou = iou
            self.max_det = max_det

            logging.info(
                f"pid: {os.getpid()} | Generate ring detection model {model_path}"
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
            #device="cuda:1",
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
                            return np.squeeze(np.array(box.cpu().xywhn))
        except Exception as err:
            logging.critical(
                f"pid: {os.getpid()} | RingDetector, predict_cls: Exception: TODO: implement this specific exception: {err.__class__.__name__}, error: {err}"
            )
        return None
