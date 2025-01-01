import logging
import numpy as np
import os
from pathlib import Path
from PIL import Image
import sys

module_path = Path(__file__).resolve().parent.joinpath("CLIP4STR")

sys.path.append(str(module_path))

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint

import re
import tag_letters

#torch.cuda.set_device(1)

#_letter_pattern = re.compile("A[ABCD][A-Z]")

_letter_pattern = re.compile(tag_letters._letters_pos_0 + tag_letters._letters_pos_1 + tag_letters._letters_pos_2)
_number_pattern = re.compile("[0-9]{3}")

class TextRecognition:

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        conf: float = 0.98,
    ):
        try:
            self.device = device
            self.conf = conf

            logging.info(
                f"pid: {os.getpid()} | Generate text recognition model {model_path}"
            )

            self.model = load_from_checkpoint(str(model_path)).eval().to(self.device)
            self.img_transform = SceneTextDataModule.get_transform(
                self.model.hparams.img_size
            )

        except Exception as err:
            logging.critical(
                f"pid: {os.getpid()} | TextRecognition, Exception: TODO: implement this specific exception: {err.__class__.__name__}, error: {err}"
            )

    def predict(self, image: Image):
        image_trans = self.img_transform(image).unsqueeze(0).to(self.device)

        p = self.model(image_trans).softmax(-1)
        # pred, p = self.model.tokenizer.decode(p)
        return self.model.tokenizer.decode(p)

    def predict_tag(self, tag_images: list):

        tag = {}
        images = {}

        try:

            for image in tag_images:
                pred = self.predict(image)
                #print(pred)
                tc = pred[0][0]
                pc = pred[1][0].cpu().detach().numpy()

                #if len(tc) == 3 and np.min(pc) > self.conf:
                if len(tc) == 3:

                    letter_match = _letter_pattern.fullmatch(tc)
                    if letter_match is not None:
                        # letter_tags.append(letter_match.string)
                        # print(f"Letter tag ok: {tc}: {pc}")
                        # letter_tag_ok = True
                        if "letters" not in tag:

                            tag["letters"] = {
                                "str": tc,
                                "pred": pc,
                            }
                            images["letters"] = image
                    else:
                        number_match = _number_pattern.fullmatch(tc)
                        if number_match is not None:
                            if "numbers" not in tag:
                                tag["numbers"] = {
                                    "str": tc,
                                    "pred": pc,
                                }
                                images["numbers"] = image

            if len(tag) != 2:
                # for key, val in tag.items():
                #    print(f"  {key}, {val}")
                # found_tag = True
                tag = {}
                images = {}

        except Exception as err:
            logging.critical(
                f"pid: {os.getpid()} | TextRecognition, predict_tag: Exception: TODO: implement this specific exception: {err.__class__.__name__}, error: {err}"
            )

        return tag
