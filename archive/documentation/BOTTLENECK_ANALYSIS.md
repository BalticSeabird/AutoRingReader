# Pipeline Performance Bottleneck Analysis & Optimization Strategy

## Executive Summary

Your pipeline has **critical bottleneck in Step 3** (Text Recognition) where the model processes sequentially without batching or multi-GPU support. Step 1 and 2 also have optimization opportunities. With 2 GPUs and many CPUs available, significant speedup is achievable.

---

## Current Implementation Analysis

### Step 1: Ring Detection (YOLO Ring Detection)
**Current Approach:**
- 2 GPU workers, each processes one video sequentially
- Frames batched in groups of 16 before YOLO
- ROI cropping applied (efficient)
- Memory cleared between batches

**Bottleneck:** GPU sequentiality
- GPU processes all frames from video A, then moves to video B
- Both GPUs can't work on same video in parallel
- Batch size of 16 is conservative (modern GPUs can handle 32-64+)

**Symptomatic Issues:**
- If one video takes longer, other GPU stays idle waiting for task
- Frame batching at 16 is inefficient for modern GPUs
- No prefetching/pipelining of next video while current finishes

**Estimated Time Impact:** 15-25% slowdown from non-optimal batching + GPU idle time

---

### Step 2: Tag Detection (YOLO-Seg on Ring Crops)
**Current Approach:**
- CPU process loads crops in batches of 10
- GPU workers (2x) take items from queue in batches of 16
- One event folder (video) at a time processed sequentially
- Results saved to queue then written to disk

**Bottleneck:** Sequential event handling + queue bottleneck
- Even though architecture supports parallelism, only ONE event folder is processed at a time
- Line 515: `for video_stem, crop_paths in event_folders:` - sequential iteration
- Each event folder waits for previous to fully complete (load → infer → save)
- Batch size of 16 for tag detection is conservative

**Symptomatic Issues:**
- If you have 10 event folders (10 videos), process runs: event1 (100% done) → event2 (100% done) → ...
- Should run: events 1,2,3 loading + inferencing + saving in parallel

**Estimated Time Impact:** 25-40% slowdown from sequential event processing

---

### Step 3: Text Recognition (CLIP4STR OCR) - **CRITICAL BOTTLENECK**
**Current Approach:**
- Load model once: `TextRecognition(device="cuda:0")`
- Group all tag crops by ring (camera, video, frame_id)
- **Sequential loop:** `for i, (group_key, crop_paths) in enumerate(tag_groups.items()):`
  - Load images from disk
  - Call `model.predict_tag(images)` - processes ALL images in group
  - Append to results
- Majority voting over all groups
- No multiprocessing, no batching across groups, one GPU

**Bottleneck:** Complete lack of parallelization + sequential inference
- **GPU Utilization:** ONLY cuda:0 used - cuda:1 completely idle
- **Inference Pattern:** Each tag_group (2-4 images typically) processes individually
- **Text Recognition Model:** CLIP4STR is slow per-image (~50-200ms depending on GPU)
- **No Batching:** Could batch 32-64 images through model at once, but processes 2-4 at a time

**Symptomatic Evidence from Code:**
```python
# Line 120-168: Sequential loop, one group at a time
for i, (group_key, crop_paths) in enumerate(tag_groups.items()):
    # Load images (serial: reads from disk)
    for crop_path in crop_paths:
        img = cv.imread(...)  # Slow I/O
    # Infer (serial: one group through model)
    tag_dict = model.predict_tag(images)  # Processes 2-4 images
    # Results append (serial)
    all_results.append()
```

**Estimated Time Impact:** If you have 1000 tag groups at 100ms per group = **1.67 minutes just for OCR**
- With proper batching to 32-64 images: could reduce to **1-2 seconds batching overhead**
- **Potential speedup: 50-100x for Step 3**

**Additional Inefficiencies:**
- Line 103: `device="cuda:0"` hardcoded - cuda:1 never used
- No image pre-loading or pipelining
- I/O-bound: disk reads during inference instead of pre-loading while inferencing

---

## GPU/CPU Resource Utilization

| Step | GPU 0 | GPU 1 | CPUs | Status |
|------|-------|-------|------|--------|
| 1 (Ring) | Partial* | Partial* | Idle | *Sequential video processing |
| 2 (Tags) | ~60% | ~60% | Used for loading | Could be better |
| 3 (OCR) | ~20% | Idle! | Not used | **Major waste** |

---

## Optimization Strategy (Priority Order)

### Priority 1: Step 3 (Text Recognition) - Highest Impact
**Estimated Improvement:** 30-50x speedup (if 1000+ groups processed)

**Changes:**
1. **Batch OCR across groups** instead of per-group
   - Load all tag crops into memory (or batches of them)
   - Batch-process 32-64 images through CLIP4STR at once
   - 1 forward pass < 32 individual forward passes
   
2. **Multi-GPU support**
   - Load model on both cuda:0 and cuda:1
   - Split groups: even indices → cuda:0, odd indices → cuda:1
   - Run 2 GPUs in parallel processing via multiprocessing.Pool

3. **Asynchronous I/O**
   - CPU worker: pre-loads next batch while GPU inferencing current batch
   - Use queue-based architecture like Step 2

4. **Progress tracking**
   - Implement per-batch progress logging
   - Report ETA like Step 2 does

**Implementation Complexity:** Medium (requires multiprocessing refactor)

---

### Priority 2: Step 1 (Ring Detection) - Medium Impact
**Estimated Improvement:** 20-40% speedup

**Changes:**
1. **Increase frame batch size**
   - Current: 16 frames per batch
   - Recommended: 32-64 (test first based on GPU VRAM)
   - Modern GPUs typically have 8-24GB - can handle 64+ frames

2. **Multi-GPU frame batching** (advanced)
   - Instead of one GPU per video, distribute frames across both GPUs
   - Process frame batch_i on GPU0, frame batch_i+1 on GPU1 (pipelined)
   - Requires refactoring process_video_with_yolo

3. **Async frame loading**
   - One thread/process loads next frames while GPU processes current batch
   - Reduces I/O wait time

**Implementation Complexity:** Low-Medium (batch size is easiest)

---

### Priority 3: Step 2 (Tag Detection) - Medium Impact
**Estimated Improvement:** 15-30% speedup

**Changes:**
1. **Parallel event folder processing**
   - Instead of processing one event folder at a time, process 2-4 in parallel
   - Must manage queue sizes and GPU memory per process
   - Use ProcessPoolExecutor or async task distribution

2. **Increase batch size**
   - Current: 16 tag crops per batch
   - Recommended: 32-48 (YOLO-seg should handle it)

3. **Prefetch next event**
   - While processing event_folder_1, start loading crops for event_folder_2's CPU workers
   - Reduce pipeline bubble time

**Implementation Complexity:** Medium

---

## Recommended Approach: Phase-Based Rollout

### Phase 1 (Quick Win): Step 3 Batching
**Effort:** 4-6 hours  
**Expected speedup:** 30-50x for step3, 15-20% overall  
**Risk:** Low (isolated change)

1. Refactor `run_step3_text_recognition()` to:
   - Accumulate tag groups into batches of 32-64
   - Process batch through model at once
   - Log results per-batch
   - Display progress

### Phase 2: Multi-GPU Step 3
**Effort:** 6-8 hours  
**Expected speedup:** Additional 1.8-2x (near-linear with 2 GPUs)  
**Risk:** Low-Medium (multiprocessing)

1. Load model on both GPUs
2. Use ProcessPoolExecutor(max_workers=2) to process batches in parallel
3. Balance batch distribution between GPUs

### Phase 3: Step 1 Batch Size + Step 2 Parallel Events
**Effort:** 3-4 hours each  
**Expected speedup:** 20-30% combined  
**Risk:** Low-Medium

1. Test frame batch sizes: 32, 48, 64
2. Convert Step 2 event loop to asyncio/ProcessPoolExecutor
3. Monitor GPU/CPU utilization

---

## Configuration Changes Needed

For immediate testing, update your YAML config:

```yaml
step1:
  batch_size: 48  # Increase from 16
  n_processes_gpu: 2
  gpu_devices: [0, 1]

step2:
  batch_size: 32  # Increase from 16
  n_processes_gpu: 2
  gpu_devices: [0, 1]

step3:
  batch_size: 32  # Increase from default (not currently used)
  n_processes_gpu: 2  # Will use after phase 2
  gpu_devices: [0, 1]  # Will use after phase 2
```

---

## Expected Overall Speedup

| Scenario | Step 1 | Step 2 | Step 3 | Overall |
|----------|--------|--------|--------|---------|
| Current (baseline) | 1x | 1x | 1x | 1x |
| Phase 1 (Step3 batch) | 1x | 1x | 30-50x | **5-8x** |
| Phase 1+2 (+ multi-GPU Step3) | 1x | 1x | 50-90x | **8-12x** |
| All phases | 1.25x | 1.3x | 50-90x | **15-20x** |

**For 1-hour video with 1000 frames, 500 rings, 1000 tag groups:**
- Current time: ~45 minutes
- Phase 1 complete: ~4-6 minutes  
- All phases complete: ~2-3 minutes

---

## Next Steps

1. **Run diagnostic profiling** (generate timing data)
2. **Test Phase 1** (Step 3 batching) on your test dataset
3. **Measure GPU utilization** (nvidia-smi monitoring)
4. **Proceed to Phase 2** if Step 3 speedup validates our analysis

Would you like me to:
- [ ] Start Phase 1 implementation (Step 3 batching)?
- [ ] Create a profiling script to measure current bottlenecks?
- [ ] Generate a test config for benchmarking?
- [ ] Start with Phase 2 (multi-GPU Step 3)?
