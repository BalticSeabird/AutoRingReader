import cv2
import logging
import numpy as np
import os
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO


#_w_target = 224
_h_target = 96


class TagDetector:

    def __init__(
        self,
        model_path: Path,
        stream: bool = True,
        conf: float = 0.7,
        iou: float = 0.7,
        max_det: int = 2,
        device_id: int = 0,
    ):
        try:
            self.stream = stream
            self.conf = conf
            self.iou = iou
            self.max_det = max_det
            self.device_id = device_id

            logging.info(
                f"pid: {os.getpid()} | Generate tag detection model {model_path} on GPU {device_id}"
            )

            self.model = YOLO(str(model_path))

            #self.model2 = YOLO("/home/bsp/tmp/ikw/models/20240320204934_RingTagDataset1_yolov8n-obb_224.pt")
 

        except Exception as err:
            logging.critical(
                f"pid: {os.getpid()} | TagDetector, Exception: TODO: implement this specific exception: {err.__class__.__name__}, error: {err}"
            )

    def predict(self, image: np.ndarray):
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
        )

    def predict2(self, image: np.ndarray):
        return self.model2(
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
        )

    def _predict_tag_entries(self, image: np.ndarray):
        tags = []
        try:
            results = self.predict(image=image)
            image_np = np.array(image)

            for res in results:
                boxes_conf = None
                if res.boxes is not None and res.boxes.conf is not None:
                    boxes_conf = res.boxes.conf.cpu().numpy()

                # Handle segmentation results (masks)
                masks = res.masks
                if masks is not None:
                    for idx, mask in enumerate(res.masks):
                        rect = cv2.minAreaRect(np.float32(mask.cpu().xy))
                        rect_resized = (
                            rect[0],
                            (1.05 * rect[1][0], 1.05 * rect[1][1]),
                            rect[2],
                        )
                        src = cv2.boxPoints(rect_resized)
                        src = src.astype(np.float32)

                        wr, hr = np.array(rect_resized[1])
                        ar = hr / wr

                        if hr > wr:
                            w_target = int(_h_target * ar)
                            dst = np.array(
                                [
                                    [0, 0],
                                    [w_target - 1, 0],
                                    [w_target - 1, _h_target - 1],
                                    [0, _h_target - 1],
                                ],
                                dtype=np.float32,
                            )
                        else:
                            w_target = int(_h_target / ar)
                            dst = np.array(
                                [
                                    [0, _h_target - 1],
                                    [0, 0],
                                    [w_target - 1, 0],
                                    [w_target - 1, _h_target - 1],
                                ],
                                dtype=np.float32,
                            )

                        M = cv2.getPerspectiveTransform(src, dst)
                        img_np_aligned = cv2.warpPerspective(
                            image_np,
                            M,
                            dsize=(w_target, _h_target),
                            flags=cv2.INTER_AREA,
                        )
                        confidence = None
                        if boxes_conf is not None and idx < len(boxes_conf):
                            confidence = float(boxes_conf[idx])

                        tags.append((Image.fromarray(img_np_aligned).convert("RGB"), confidence))

                # Handle detection-only results (bounding boxes)
                elif res.boxes is not None and len(res.boxes) > 0:
                    for idx, box in enumerate(res.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Extract bounding box region
                        bbox_height = y2 - y1
                        bbox_width = x2 - x1
                        ar = bbox_height / bbox_width if bbox_width > 0 else 1.0

                        if bbox_height > bbox_width:
                            w_target = int(_h_target * ar)
                        else:
                            w_target = int(_h_target / ar)

                        # Extract and resize the tag region
                        tag_region = image_np[y1:y2, x1:x2]
                        if tag_region.size > 0:
                            tag_resized = cv2.resize(tag_region, (w_target, _h_target))
                            confidence = None
                            if boxes_conf is not None and idx < len(boxes_conf):
                                confidence = float(boxes_conf[idx])

                            tags.append((Image.fromarray(tag_resized).convert("RGB"), confidence))

        except Exception as err:
            logging.critical(
                f"pid: {os.getpid()} | TagDetector, predict_tag: Exception: {err.__class__.__name__}, error: {err}"
            )
        return tags

    def predict_tag(self, image: np.ndarray):
        return [tag for tag, _ in self._predict_tag_entries(image)]

    def predict_tag_with_meta(self, image: np.ndarray):
        return self._predict_tag_entries(image)
    
    # def predict_tag_2(self, image: np.ndarray):
    #     try:
    #         results = self.predict2(image=image)
    #         tags = []
    #         for res in results:
    #             obbs = res.obb
    #             if obbs is not None:
    #                 for i, obb in enumerate(res.obb):
    #                     # print(obb.cpu().conf.numpy()[0])
    #                     # print(obb.cpu().cls.numpy()[0])
    #                     # print(obb.cpu().xywhr.numpy().flatten())
    #                     rect = obb.cpu().xywhr.numpy().flatten()
    #                     rect_resized = (
    #                         (rect[0], rect[1]),
    #                         (1.05 * rect[2], 1.05 * rect[3]),
    #                         rect[4],
    #                     )
    #                     src = cv2.boxPoints(rect_resized)
    #                     #src = src.astype(np.float32)
                        
    #                     wr, hr = np.array(rect_resized[1])  # * [w, h]
    #                     #print(hr>wr)
    #                     ar = hr / wr

    #                     if hr>wr:
    #                         assert 0
    #                         #h_target = int(_w_target / ar)
    #                         w_target = int(_h_target * ar)

    #                        #print(w_target,_h_target)

    #                         dst = np.array(
    #                             # [
    #                             #     [0, 0],
    #                             #     [_w_target - 1, 0],
    #                             #     [_w_target - 1, h_target - 1],
    #                             #     [0, h_target - 1],
    #                             # ],
    #                             [
    #                                 [0, 0],
    #                                 [w_target - 1, 0],
    #                                 [w_target - 1, _h_target - 1],
    #                                 [0, _h_target - 1],
    #                             ],
    #                             dtype=np.float32,
    #                         )
    #                     else:
    #                         #h_target = int(_w_target / ar)
    #                         w_target = int(_h_target / ar)

    #                        #print(w_target,_h_target)

    #                         dst = np.array(
    #                             [
    #                                 [0, _h_target - 1],
    #                                 [0, 0],
    #                                 [w_target - 1, 0],
    #                                 [w_target - 1, _h_target - 1],
    #                             ],
    #                             # [
    #                             #     [0, 0],
    #                             #     [0, _h_target - 1],
    #                             #     [w_target - 1, _h_target - 1],
    #                             #     [w_target - 1, 0],
    #                             # ],
    #                             dtype=np.float32,
    #                         )

    #                     M = cv2.getPerspectiveTransform(src, dst)

    #                     image_np = np.array(image)
    #                     img_np_aligned = cv2.warpPerspective(
    #                         image_np,
    #                         M,
    #                         dsize=(
    #                             # _w_target,
    #                             # h_target,
    #                             w_target,
    #                             _h_target,
    #                         ),
    #                         flags=cv2.INTER_AREA,
    #                         # flags=cv2.INTER_LINEAR,
    #                         # flags=cv2.INTER_CUBIC,
    #                     )
    #                     tags.append(Image.fromarray(img_np_aligned).convert("RGB"))    

    #     except Exception as err:
    #         logging.critical(
    #             f"pid: {os.getpid()} | TagDetector, predict_tag: Exception: TODO: implement this specific exception: {err.__class__.__name__}, error: {err}"
    #         )
    #     return tags

