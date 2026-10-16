import logging
import numpy as np
import os
from pathlib import Path
from PIL import Image
import sys

module_path = Path(__file__).resolve().parent.parent.joinpath("CLIP4STR")

sys.path.append(str(module_path))

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint

import re
from . import tag_letters

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
            self.img_transform = None
            self.model = None

            logging.info(
                f"pid: {os.getpid()} | Generate text recognition model {model_path}"
            )

            self.model = load_from_checkpoint(str(model_path)).eval().to(self.device)
            self.img_transform = SceneTextDataModule.get_transform(
                self.model.hparams.img_size
            )
            logging.info(f"pid: {os.getpid()} | TextRecognition model loaded successfully")

        except Exception as err:
            logging.critical(
                f"pid: {os.getpid()} | TextRecognition, Exception: TODO: implement this specific exception: {err.__class__.__name__}, error: {err}"
            )
            import traceback
            logging.critical(f"Traceback: {traceback.format_exc()}")

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
                # Convert numpy array to PIL Image if needed
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)

                pred = self.predict(image)
                tc = pred[0][0]
                pc = pred[1][0].cpu().detach().numpy()

                #if len(tc) == 3 and np.min(pc) > self.conf:
                if len(tc) == 3:

                    letter_match = _letter_pattern.fullmatch(tc)
                    if letter_match is not None:
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
                tag = {}
                images = {}

        except Exception as err:
            logging.critical(
                f"pid: {os.getpid()} | TextRecognition, predict_tag: Exception: TODO: implement this specific exception: {err.__class__.__name__}, error: {err}"
            )
            import traceback
            logging.critical(f"Traceback: {traceback.format_exc()}")

        return tag

    def predict_tag_batch(self, images_list: list):
        """
        Process a batch of images and return predictions for each.

        Processes multiple images through the model in a single batch operation,
        which is much faster than individual predictions.

        Args:
            images_list: List of PIL Images or numpy arrays

        Returns:
            List of dicts, one per image: {letters, numbers} or {} if no match found
        """
        import torch

        try:
            if not images_list:
                return []

            # Convert all images to PIL if needed
            pil_images = []
            for image in images_list:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                pil_images.append(image)

            # Transform all images and create batch tensor
            transformed_images = [self.img_transform(img) for img in pil_images]
            batch_tensor = torch.stack(transformed_images).to(self.device)

            # Single forward pass on entire batch
            with torch.no_grad():
                batch_predictions = self.model(batch_tensor).softmax(-1)

            # Decode each image's prediction
            results = []
            for i in range(len(pil_images)):
                pred_i = batch_predictions[i:i+1]  # Keep batch dimension
                tokens_i, confs_i = self.model.tokenizer.decode(pred_i)

                # Apply pattern matching to detect letters vs numbers
                tag_dict = {}
                if tokens_i and len(tokens_i) > 0 and len(tokens_i[0]) == 3:
                    tc = tokens_i[0]
                    pc = confs_i[0].cpu().detach().numpy()

                    letter_match = _letter_pattern.fullmatch(tc)
                    number_match = _number_pattern.fullmatch(tc)

                    if letter_match:
                        tag_dict["letters"] = {"str": tc, "pred": pc}
                    elif number_match:
                        tag_dict["numbers"] = {"str": tc, "pred": pc}

                results.append(tag_dict)

            return results

        except Exception as err:
            logging.critical(
                f"pid: {os.getpid()} | TextRecognition, predict_tag_batch: Exception: {err.__class__.__name__}, error: {err}"
            )
            import traceback
            logging.critical(f"Traceback: {traceback.format_exc()}")
            return [{} for _ in images_list]  # Return empty dicts for all images on error
