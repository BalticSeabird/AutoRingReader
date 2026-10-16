# New Step-Based Pipeline Architecture

## Overview

The new pipeline reorganizes the processing into **3 independent sequential phases** instead of processing each video through all steps at once. This design enables:

- **GPU batching**: Group frames/crops from multiple videos for efficient inference
- **Better resource utilization**: 2 GPUs and 128 CPUs distributed across phases
- **Reduced model overhead**: Load each model once per phase, not once per video
- **30-50% performance improvement**: Conservative estimate based on batching gains
- **Easier debugging**: Each step is isolated and can be monitored independently

## Architecture

### Phase 1: Ring Detection
```
all_videos → [CPU workers extract frames] → [GPU workers YOLO inference (batched)]
                                        → [Save ring crops]
```
**Output**: Ring crop images saved to `output/step1_ring_crops/{camera}/{video_stem}/`

**Configuration** (batch.yaml):
```yaml
step1:
  n_processes_cpu: 16           # Frame extraction workers
  n_processes_gpu: 2             # YOLO inference workers
  gpu_devices: [0, 1]            # Distribute across 2 GPUs
  batch_size: 16                 # Frames per batch
  frame_sample_rate: 1           # Process every Nth frame
  conf_threshold: 0.7
  max_det: 1
```

### Phase 2: Tag Detection
```
all_ring_crops → [CPU workers load rings] → [GPU workers YOLO-seg (batched)]
                                        → [CPU workers warp] → [Save tag crops]
```
**Output**: Tag crop images saved to `output/step2_tag_crops/{camera}/{video_stem}/`

**Configuration** (batch.yaml):
```yaml
step2:
  n_processes_cpu: 16
  n_processes_gpu: 2
  gpu_devices: [0, 1]
  batch_size: 16
  conf_threshold: 0.7
  max_det: 2                    # Max tags per ring
  tag_height: 96               # Normalized height
```

### Phase 3: Text Recognition
```
all_tag_crops → [CPU workers batch] → [GPU worker CLIP4STR (single GPU)]
                                   → [CPU workers post-process] → [Save YAML]
```
**Output**: YAML files with recognized rings saved to `output/{video_stem}.yaml`

**Configuration** (batch.yaml):
```yaml
step3:
  n_processes_cpu: 32           # More CPU for post-processing
  n_processes_gpu: 1             # CLIP4STR uses 1 GPU
  gpu_devices: [0]
  batch_size: 32                # Tag crops per batch
  conf_threshold: 0.90
  rotational_validation: true
```

## Usage

### 1. Configure Batch File

Copy and edit `batch/batch_pipeline.yaml`:

```bash
cp batch/batch_pipeline.yaml batch/batch_test.yaml
```

Edit paths:
```yaml
input:
  video_root_path: /your/video/path
  cameras: [FAR3_SCALE, FAR6_SCALE]
  resume: true

output_root_path: /your/output/path

models:
  ring_detector:
    model_root_path: /path/to/models
  # ... etc
```

### 2. Run Pipeline

```bash
python pipeline_orchestrator.py batch/batch_test.yaml
```

### 3. Monitor Progress (in another terminal)

```bash
python monitor_pipeline.py
```

Shows real-time progress:
- Videos processed per step
- Success/error counts
- Recent completions

## Output Structure

```
output_root_path/
├── step1_ring_crops/
│   ├── FAR3_SCALE/
│   │   ├── video_0/
│   │   │   └── _metadata.json    (frame→detection mapping)
│   │   └── video_1/
│   │
├── step2_tag_crops/
│   ├── FAR3_SCALE/
│   │   ├── video_0/
│   │   │   ├── 001_ring_tag_0.jpg
│   │   │   └── 001_ring_tag_1.jpg
│   │
├── step3_ocr_results/
│   ├── FAR3_SCALE/
│   │   └── video_0/
│   │       ├── video_0.yaml      (final tag)
│   │       └── ...
│
├── video_stats.csv               (backward compatible with old pipeline)
└── log/
    └── events.db                 (SQLite database with per-step results)
```

## Database Schema

### step1_ring_detections
- video_path, camera, status, processing_seconds
- frames_total, frames_processed, ring_detect_count, ring_detect_rate
- ring_conf_mean, ring_conf_max

### step2_tag_detections
- video_path, camera, status, processing_seconds
- rings_processed, tags_detected_count, tags_detected_frames
- tag_detection_rate, tag_conf_mean, tag_conf_max

### step3_text_recognition
- video_path, camera, status, processing_seconds
- tags_processed, ocr_matches_count, ocr_success_rate
- letters/numbers confidence metrics

### final_results
- video_path, camera, final_tag, final_tag_count
- step1_seconds, step2_seconds, step3_seconds (timing per step)

## Resume Support

The pipeline supports resuming interrupted processing:

```yaml
input:
  resume: true    # Skip videos with status='success'
```

If a step fails:
1. Fix the issue
2. Re-run the same command
3. Pipeline automatically skips completed videos
4. Continues from the failed video

## Performance Tuning

### For your 128 CPUs and 2 GPUs:

**Conservative (default)**:
- Step 1: 16 CPU + 2 GPU
- Step 2: 16 CPU + 2 GPU
- Step 3: 32 CPU + 1 GPU

**Aggressive (if memory allows)**:
- Increase batch_size (16→32) in step1/2
- Increase n_processes_cpu if CPU utilization <80%

### Frame Sampling for Large Videos

To speed up on huge videos:

```yaml
step1:
  frame_sample_rate: 3    # Process every 3rd frame (3x speedup)
```

Quality vs. speed tradeoff - adjust based on needs.

## Key Improvements over Old Pipeline

| Aspect | Old | New |
|--------|-----|-----|
| Model loading | Once per video | Once per phase |
| Frame processing | Per-frame | Batched |
| GPU batching | No | Yes (16 frames/batch) |
| CPU utilization | Limited | Up to 128 CPUs |
| Resume on failure | Manual intervention | Automatic |
| Step isolation | Mixed in one script | Independent scripts |
| Estimated speedup | Baseline | 30-50% faster |

## Troubleshooting

### Pipeline hangs on Step 1

Check GPU memory:
```bash
nvidia-smi
```

If full, reduce batch_size:
```yaml
step1:
  batch_size: 8    # Was 16
```

### Database locked errors

Concurrent access to database. Ensure only one pipeline instance running.

### Missing ring crops in Step 2

Step 1 didn't complete successfully. Check logs:
```bash
tail -f log/YYYYMMDDHHMMSS_orchestrator.txt
tail -f log/YYYYMMDDHHMMSS_step1_worker_*.txt
```

### No OCR results in Step 3

Check Step 2 output - tag crops may not have detector matches. Verify:
```bash
ls output/step2_tag_crops/*/*/
```

## Logging

Logs are automatically written to `./log/` directory:

- `YYYYMMDDHHMMSS_orchestrator.txt` - Main orchestrator log
- `YYYYMMDDHHMMSS_step1_worker_*.txt` - Step 1 worker logs
- `YYYYMMDDHHMMSS_step2_worker_*.txt` - Step 2 worker logs
- `YYYYMMDDHHMMSS_step3_worker_*.txt` - Step 3 worker logs

Enable debug logging by modifying orchestrator.py:
```python
logging.basicConfig(level=logging.DEBUG)  # Instead of INFO
```

## Files

**Core Pipeline:**
- `pipeline_orchestrator.py` - Main entry point
- `pipeline_config.py` - Configuration dataclasses
- `pipeline_logger.py` - SQLite database logging for steps
- `pipeline_utils.py` - Shared utilities

**Step Implementations:**
- `step1_ring_detection.py` - Ring detection phase
- `step2_tag_detection.py` - Tag detection phase
- `step3_text_recognition.py` - Text recognition phase

**Utilities:**
- `monitor_pipeline.py` - Real-time progress monitoring
- `batch/batch_pipeline.yaml` - Example configuration

**Original Models (unchanged):**
- `ring_detector.py` - YOLO ring detection wrapper
- `tag_detector.py` - YOLO-seg tag detection wrapper
- `text_recognition.py` - CLIP4STR OCR wrapper

## Backward Compatibility

The new pipeline maintains compatibility with the old system:

- Same `video_stats.csv` format (34+ columns)
- Same per-frame YAML output structure
- Same ring ID format (3 letters + 3 digits)
- Can be used as drop-in replacement for `ring_tag_read.py`

## Next Steps

1. Test with a small batch (5 videos) to validate setup
2. Benchmark performance vs old pipeline
3. Tune batch sizes and worker counts based on GPU/CPU logs
4. Deploy to full dataset

## Questions?

Refer to `LLM_repo_documentation.md` for details on models and the original architecture.
