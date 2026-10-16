# AutoRingReader Technical Overview

## 1. Purpose and High-Level Concept
AutoRingReader automates reading alphanumeric identifiers on seabird leg rings from multi-camera video footage. The system chains three vision tasks:

1. **Ring localization**: YOLO-based detector isolates a ring on the bird’s leg.
2. **Tag extraction and rectification**: A segmentation-style detector crops the planar band surface and corrects perspective and orientation.
3. **Text recognition**: A CLIP4STR OCR model enforces valid patterns for letters and numbers to finalize the ID.

The main orchestration lives in [ring_tag_read.py](ring_tag_read.py).

## 2. Repository Layout
| Path | Role |
| --- | --- |
| [ring_tag_read.py](ring_tag_read.py) | Entry point that batches videos, runs multiprocessing, and glues detection/OCR modules. |
| [ring_detector.py](ring_detector.py) | Thin wrapper around Ultralytics YOLO for ring-class bounding boxes. |
| [tag_detector.py](tag_detector.py) | YOLO segmentation wrapper plus perspective warping to produce normalized tag crops. |
| [text_recognition.py](text_recognition.py) | CLIP4STR-based OCR with regex filtering of valid ring codes. |
| [config.py](config.py) | Shared enums (ring classes, defaults, file extensions). |
| [metrics.py](metrics.py) | Utility Intersection-over-Union used for duplicate suppression. |
| [annotations.py](annotations.py) | Helpers for reading/writing YAML annotations when creating datasets. |
| [logger.py](logger.py) | Multiprocessing-safe logging that mirrors stdout to timestamped log files. |
| [tag_letters.py](tag_letters.py) | Enumerates valid three-letter prefixes and generates regex pieces for OCR validation. |
| [batch/*.yaml](batch/batch_0.yaml) | Example batch jobs: input video roots, camera filters, model paths, detector/OCR thresholds, CPU count. |
| [eval/ring_tag_read_eval.py](eval/ring_tag_read_eval.py) | Variant of the main runner used for benchmarking against a CSV of known events. |
| [CLIP4STR](CLIP4STR/README.md) | Third-party STR implementation bundled as a submodule; provides OCR checkpoints and training scripts. |
| [requirements.txt](requirements.txt) | Consolidated Python dependencies (Ultralytics, PyTorch Lightning, CLIP4STR stack, etc.). |

## 3. Processing Pipeline

1. **Input discovery**  
   `main()` parses a batch YAML (see [batch/batch_0.yaml](batch/batch_0.yaml)) describing `video_root_path`, allowed `cameras`, output directory, per-model hyperparameters, and multiprocessing fan-out.  
   `ring_tag_read()` walks `video_root_path`, filters filenames by camera tag, and assembles `(video_path, camera)` tuples.

2. **Execution scaffolding**  
   - `remove_and_make_dir()` ensures `output_root_path` is empty before each run.  
   - `Logger` (from [logger.py](logger.py)) starts a QueueListener so every worker process logs to the same timestamped file under `./log/`.

3. **Parallel inference (`Pool.starmap`)**  
   Each worker runs `_ring_tag_read()` on a single video:
   - Opens the video with OpenCV and iterates frames.
   - Instantiates `RingDetector`, `TagDetector`, and `TextRecognition` with the configured checkpoints and thresholds.

4. **Ring localization** ([ring_detector.py](ring_detector.py))  
   - `predict_cls()` calls YOLO, iterates detections, and returns the first bounding box whose class matches `config.RingTypes.METAL_RING`.  
   - Boxes are normalized `(x_center, y_center, width, height)` in image-relative units.

5. **Region-of-interest filtering**  
   - `ROIS` in [ring_tag_read.py](ring_tag_read.py) defines camera-specific polygons (`shapely.geometry.Polygon`).  
   - When a camera has an ROI, each detected ring polygon is checked with `roi.contains()` to reject off-target detections (e.g., reflections or background rings).

6. **Ring crop creation**  
   - `_crop_image()` converts the frame to PIL, crops a fixed 224×224 window centered on the normalized box, and returns both the full frame and crop for downstream saving.

7. **Tag strip extraction** ([tag_detector.py](tag_detector.py))  
   - YOLO segmentation masks are converted to minimum-area rectangles.  
   - The rectangle is padded 5% and rectified with `cv2.getPerspectiveTransform`, producing a portrait-oriented strip of height 96 px (width depends on aspect ratio).  
   - Each strip is stored as a PIL image for OCR. (Optional `predict_tag_2()` shows experimentation with oriented bounding boxes.)

8. **OCR with pattern enforcement** ([text_recognition.py](text_recognition.py))  
   - Loads CLIP4STR via `load_from_checkpoint()` and obtains the same transform as during training.  
   - Each tag image is tokenized; predictions must yield exactly three characters.  
   - Regex patterns are assembled from [tag_letters.py](tag_letters.py) for letters (`[AAZ]`-style) and `[0-9]{3}` for digits, so only realistic codes survive.  
   - Both original and 180° rotated crops are evaluated to handle upside-down rings; whichever has higher confidence is retained.

9. **Temporal smoothing and deduplication**  
   - Exponential moving averages (`alpha = 0.9`) accumulate per-character probabilities (`pred_letters_ema`, `pred_numbers_ema`). OCR output is accepted only when the minimum EMA exceeds `text_recognition_conf`.  
   - Bounding boxes are compared with existing ones using `metrics.IOU`; if `IOU > 0.9` and the predicted strings match, the frame is skipped to prevent duplicate storage.

10. **Artifacts and metadata**  
    - Each accepted frame writes:  
      - Full-frame JPEG (`{stem}_{idx}.jpg`)  
      - Cropped ring JPEG (`{stem}_{idx}_crop.jpg`)  
      - YAML file with raw confidences and strings.  
    - Files are first stored under `output_root_path/tmp/<camera>/<video_stem>` to avoid partial results. After the video finishes, the most frequent tag combination is summarized into `{stem}.yaml` and the whole folder is atomically moved out of `tmp`.  
    - If no tag survives filtering, the temporary folder is deleted.
      - Detections that never yield a complete OCR tag are not persisted: `_ring_tag_read()` resets the EMA trackers and `continue`s whenever either the letters or numbers block is missing, so no images or YAMLs are emitted, and the temporary folder is removed at the end. Capturing those failures for statistics would require persisting them before that early return.
      - Once multiprocessing completes, `ring_tag_read()` emits `video_stats.csv` in the chosen `output_root_path`. Each row corresponds to a processed video/event and captures frame counts, how many detections each stage produced, average detector confidences, OCR candidate vs. accepted counts, EMA-derived confidence traces, and the majority-vote tag (if any). This CSV is the quickest way to compile per-stage performance tables for downstream analysis without re-reading every per-frame artifact.

11. **Completion**  
    - Any leftover temp folder is removed in `finally`.  
    - Total runtime per video and per batch is logged.

## 4. Core Modules in Detail

### 4.1 `ring_tag_read.py`
- `ring_tag_read()` orchestrates job setup, logging, input discovery, and multiprocessing.  
- `_ring_tag_read()` performs the end-to-end pipeline per video, including ROI checks, EMA smoothing, duplicate suppression, artifact writing, and summary YAML creation.  
- `ROIS` dataclass makes it easy to add new camera polygons; values are in pixel coordinates matching the raw frame resolution.  
- `ring_tag_read_eval.py` under `eval/` mirrors this flow but filters videos by event IDs loaded from `eval/OCR_Valid241105.csv` before benchmarking.

### 4.2 `RingDetector`
- Wraps Ultralytics YOLO (see [ring_detector.py](ring_detector.py)).  
- `predict()` exposes the raw `YOLO.__call__`.  
- `predict_cls()` returns the first bounding box for a target class index, enabling quick filtering by `config.RingTypes`.

### 4.3 `TagDetector`
- In [tag_detector.py](tag_detector.py), segmentation masks are interpreted as polygons, padded, and rectified into uniform strips.  
- The design assumes vertical orientation after warping; if the rectangle is taller than wide, the destination quadrilateral is defined differently to maintain orientation.  
- Optional `predict2()` shows experimentation with oriented bounding-box models (commented out).

### 4.4 `TextRecognition`
- Uses the CLIP4STR submodule from [CLIP4STR](CLIP4STR/README.md).  
- `predict()` runs a forward pass and decodes logits to tokens.  
- `predict_tag()` loops over all strip candidates, stores the first valid letter triplet and number triplet, and enforces regex validity.  
- Regex components `_letters_pos_0/1/2` originate in [tag_letters.py](tag_letters.py), which enumerates every allowed prefix the ornithologists expect (e.g., `AAA`, `AAB`, …). This keeps OCR from hallucinating visually plausible but biologically impossible codes.

## 5. Configuration and Extensibility

- **Batch files**: [batch/batch_0.yaml](batch/batch_0.yaml) illustrates how to control the pipeline. Key fields:  
  - `input.video_root_path`: absolute path scanned recursively for `.mp4`.  
  - `input.cameras`: substrings that must appear in filenames (used to determine ROI).  
  - `models.*`: paths, detection thresholds, maximum detections per frame.  
  - `text_recognition.conf`: EMA threshold; lowering it increases recall at the cost of more false positives.  
  - `multiprocessing.n_processes`: passed to `Pool`.

- **ROI tuning**: Update `ROIS` polygons to restrict detections to physically meaningful portions of each camera view.

- **Ring categories**: Extend `config.RingTypes` if you train additional YOLO classes (e.g., plastic vs. metal). The `id()` helper keeps the mapping consistent.

- **Output schema**: Per-frame YAMLs contain `video_path`, `image_idx`, and `tag` dictionaries with `str` and `pred` (per-character confidence arrays). The summary YAML stores `number_of_detections` and the split `str1`/`str2` components used for downstream aggregation.

- **Logging**: `Logger` (in [logger.py](logger.py)) must be started in the parent process before `Pool` is spawned so every worker shares the queue.

## 6. Models and Dependencies

- **Models**:  
  - Ring detector: YOLOv8 (`.pt`) trained on `RingDataset1`; expects `metal_ring`, `large_metal_ring`, `plastic_ring`.  
  - Tag detector: YOLOv8 segmentation model (optionally OBB variant).  
  - OCR: CLIP4STR checkpoints (`clip4str_base16x16_d70bde1f2d.ckpt`, etc.) stored under `CLIP4STR/models`.

- **Python stack**: [requirements.txt](requirements.txt) lists Ultralytics, PyTorch Lightning 1.6, Ray Tune, Hydra, shapely, OpenCV (both GUI and headless), and CLIP4STR’s NLP dependencies. GPU acceleration is assumed; adjust `device` in `TextRecognition` if you must run on CPU.

## 7. Evaluation Workflow

- `eval/ring_tag_read_eval.py` loads `eval/OCR_Valid241105.csv` to focus on clips with known ground-truth IDs.  
- The script mirrors the main pipeline but logs per-event outcomes so you can compute accuracy/recall metrics.  
- Modify the CSV to align with the subset used in your paper; only the `event` column (video stems) is referenced.

## 8. Running the System

1. Install dependencies (ideally in a virtual environment) and ensure CUDA/GPU drivers match Ultralytics + CLIP4STR requirements.
2. Populate `models/` with the YOLO checkpoints and `CLIP4STR/models/` with the OCR checkpoint.
3. Edit a batch file (copy [batch/batch_0.yaml](batch/batch_0.yaml)) with absolute paths and thresholds suited to your dataset.
4. Launch inference:  