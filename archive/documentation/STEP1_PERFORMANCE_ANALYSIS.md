# Step 1 Performance Issue: Memory Accumulation Bottleneck

## Problem Identified

**Long videos (3+ minutes) are extremely slow due to excessive memory accumulation**

### Root Cause

In `step1_ring_detection.py` lines 266-290, the code:
1. Processes frames correctly in batches
2. Clears batch_frames after each batch
3. **BUT stores ALL results in `all_results` list** (line 276, 289)

Then lines 292-333 iterate through `all_results` to save crops **after all frames are processed**.

### Memory Impact

For a 3-minute video @ 25 fps:
- Total frames: 3 min × 60 sec × 25 fps = **4,500 frames**
- Each frame: 1920×1080×3 bytes = **6 MB**
- Total stored in memory: 4,500 × 6 MB = **~27 GB RAM** 😱

### Why Short Videos Are Fast
Short videos (< 30 seconds) fit in memory, so:
- Processing is fast (frames loaded + processed quickly)
- Cropping completes
- Memory released

Long videos cause:
- Extreme RAM/VRAM usage
- Slow memory allocation/deallocation
- System thrashing
- Potential OOM crashes

## Solution

Process results **immediately** instead of storing all frames:
1. After each batch processes, save crops immediately
2. Clear frames from all_results (only keep metadata)
3. Never accumulate full frames in memory

Expected speedup: **4-5x faster** for 3-minute videos

## Files to Modify
- `step1_ring_detection.py` (refactor process_video_with_yolo function)
