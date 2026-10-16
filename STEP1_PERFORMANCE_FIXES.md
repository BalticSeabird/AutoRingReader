# Step 1 Performance Bottleneck Analysis

## Summary
The primary bottleneck in Step 1 is **sequential frame extraction from video**, not disk I/O for crop saving. Each frame read from NAS is a blocking network I/O operation happening one-at-a-time.

---

## Bottleneck Breakdown

### 1. **PRIMARY: Sequential Frame Reading (30-50% of total time)**

**Current Implementation** (`pipeline_utils.py:175-217`):
- Uses `cv.VideoCapture()` to read frames sequentially with `cap.read()`
- Each read blocks waiting for network data from NAS
- No pipelining - YOLO sits idle while waiting for next batch

**Impact:** With 1-hour video:
- Just frame I/O = ~2-3 minutes overhead
- YOLO inference runs only during batching, creating GPU idle time

---

### 2. **SECONDARY: Color Space Conversion (5-10% of total time)**

**Current** (`step1_ring_detection.py:53`):
```python
frames_rgb = [cv.cvtColor(f, cv.COLOR_BGR2RGB) for f in batch_frames]  # CPU loop
```

---

### 3. **TERTIARY: Crop Saving (5-15%)**

Uses async ThreadPoolExecutor (doesn't block GPU but consumes bandwidth).

---

## Why Disk I/O is NOT the Primary Bottleneck

1. **Crop saving is async** - doesn't wait
2. **Frame reading** time dominates sequentially

---

## Recommended Solutions (by impact)

### ✅ **Solution 1: Prefetch Frames in Background Thread (BEST ROI)**

Spawn producer thread reading frames into queue while main thread processes YOLO batches.

**Expected Speedup:** 1.5-2x (GPU no longer idles)
**Effort:** 30 minutes
**Risk:** Low

This overlaps frame I/O with GPU inference, hiding latency.

---

### ✅ **Solution 2: Use GPU Video Decoding (NVIDIA NVDEC)**

Use `av` (PyAV) library with NVIDIA hardware decoder (`h264_cuvid`).

**Expected Speedup:** 1.5-3x (depends on codec)
**Effort:** Medium
**Risk:** Medium (needs GPU-capable ffmpeg)

CPU decompression is the hidden cost in `cv.VideoCapture`.

---

### ✅ **Solution 3: Increase Batch Size (QUICK WIN)**

Change config from `batch_size: 32` to `batch_size: 64-128`.

**Expected Speedup:** 1.1-1.3x
**Effort:** 5 minutes
**Risk:** Low (check GPU memory: ~384MB × 128 frames)

---

### ✅ **Solution 4: Batch Color Space Conversion**

Use numpy reshape instead of per-frame cv.cvtColor loop.

**Expected Speedup:** 1.05-1.15x
**Effort:** 15 minutes
**Risk:** Low

---

## Implementation Priority

**DO THESE IN ORDER:**
1. Increase batch size (5 min) - immediate low-risk speedup
2. Add frame prefetch (30 min) - best total speedup
3. Try GPU decoding (medium) - if available
4. Batch color conversion (15 min) - nice-to-have

**Conservative combined:** Solutions 1+3 = **1.7-2.3x total**
**Optimistic combined:** Solutions 1+2+3+4 = **2.1-3.5x total**

---

## Verification

Once implemented, measure both separately:
```python
# Time just frame reads
read_start = time.time()
batch_frames = get_batch(...)
read_time = time.time() - read_start

# Time YOLO inference 
yolo_start = time.time()
results = detector.predict(batch_frames)
yolo_time = time.time() - yolo_start

logger.info(f"Read: {read_time:.3f}s | YOLO: {yolo_time:.3f}s")
```

This will show if prefetch is actually overlapping I/O with GPU time.

