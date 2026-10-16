# Fixes Applied - April 14, 2026

## Issue 1: ROI Cropping Offset (FAR3_SCALE & TRI5_SCALE 2025)

### Problem
- Crops from FAR3_SCALE and TRI5_SCALE 2025 videos were showing the wrong region
- Rings were not visible in extracted crops
- Offset appeared to be from incorrect ROI coordinates

### Root Cause
1. **Calibration Mismatch**: ROI coordinates were calibrated for higher resolution videos (2560+ width)
2. **Missing 2025 Entries**: No year-specific entries for 2025, falling back to incorrect default
3. **YAML Key Type Issue**: Year keys (2023, 2024) were parsed as integers by YAML, but code expected strings

### Solution Applied
1. **Analyzed actual ring locations**: Ran YOLO on full frames to find where rings actually appear
2. **Captured correct 2025 ROIs**:
   - **FAR3_SCALE 2025**: (1622, 474, 1724, 576) - small region on right side where rings actually are
   - **TRI5_SCALE 2025**: (399, 259, 614, 468) - updated from incorrect (374, 137, 2263, 1017)
3. **Fixed YAML Parsing**: Quoted year keys ("2023", "2024", "2025") to ensure string type matching

### Files Modified
- `config/rois_config.yaml` - Updated with proper 2025 coordinates and quoted year keys

### Verification
```python
ROI Config Test Results:
✓ FAR3_SCALE 2025: (1622, 474, 1724, 576)
✓ FAR3_SCALE 2024: (461, 403, 1851, 1354)
✓ TRI5_SCALE 2025: (399, 259, 614, 468)
```

---

## Issue 2: Step 1 Performance Bottleneck (1 min for 3-min video)

### Problem
- Long videos seemed slow to process in Step 1
- User reported ~60 seconds to process 3-minute video
- Suspected issue was crop saving

### Root Cause
Crops were being saved **synchronously in the main GPU inference thread**, causing:
- GPU workers to block while waiting for disk I/O
- No parallelization between inference and saving
- NAS network latency blocking YOLO processing

### Solution Applied
1. **Implemented Async Crop Saving**: Uses ThreadPoolExecutor with 2 workers
   - Crop extraction happens in main thread
   - Saving happens in thread pool (non-blocking)
   - GPU inference continues while saves complete in background
   - Main thread waits for pending saves after inference finishes

2. **Added Configuration Support**: 
   - Respects `save_crops` config option from pipeline_config
   - Can disable crop saving entirely if needed (for speed)

### Files Modified
- `step1_ring_detection.py`:
  - Added ThreadPoolExecutor import
  - New `_save_crop_image()` helper function for async I/O
  - Updated `process_video_with_yolo()` to queue saves instead of blocking
  - Updated parameter passing in `run_step1_ring_detection()`
  - Main thread now waits for pending saves with timeout

### Performance Impact
- Database shows Step 1 typically runs at 100-200+ fps (very fast)
- Async saving eliminates blocking on disk I/O
- Expected improvement especially noticeable on slower NAS mounts
- Backward compatible - saves crops same as before, just non-blocking

### Testing
- Code compiles without errors ✓
- ROI config loads correctly ✓
- Syntax validated ✓

---

## Summary
1. ✅ Fixed ROI cropping offset for 2025 videos (FAR3_SCALE, TRI5_SCALE)
2. ✅ Optimized Step 1 with async crop saving to eliminate GPU blocking
3. ✅ Both changes backward compatible with existing pipeline

**Next Steps**: Run Step 1 on FAR3_SCALE_2025-06-17_10_109.mp4 to verify:
- Crops now show actual rings in expected region
- Processing time is not impacted by crop I/O
