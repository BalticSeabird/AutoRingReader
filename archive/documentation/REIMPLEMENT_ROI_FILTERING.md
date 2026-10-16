## ROI Filtering Re-implementation - Summary

### Changes Made

The Step 1 ring detection now supports **ROI-based filtering of saved crops**. The approach:
1. **Run YOLO on the full frame** (no cropping)
2. **Filter detections by ROI** - only save crops for detections whose center is within the ROI bounding box
3. **Track filtering statistics** - log how many detections were filtered out

### Updated Files

**`step1_ring_detection.py`:**

1. **Added `_detection_in_roi()` helper function** (lines 68-93)
   - Checks if a detection's center point is within the ROI bounding box
   - Takes normalized bbox and ROI bbox in pixel coordinates
   - Returns True if detection is within ROI, False otherwise

2. **Modified `process_video_with_yolo()` function:**
   - **Lines 148-155**: Retrieve ROI bbox from ROI config using camera name and video year
   - **Line 168**: Added `filtered_count` variable to track detections filtered by ROI
   - **Lines 209-214**: Check each detection against ROI before saving crop (first batch loop)
   - **Lines 272-277**: Same ROI check for remaining frames (after loop)
   - **Lines 321-323**: Enhanced logging to show how many detections were filtered by ROI
   - **Return value**: Added `ring_crop_count` and `ring_filtered_count` fields

3. **Updated `run_step1_ring_detection()` logging** (lines 456-461)
   - Logs ROI filtering statistics when displaying results
   - Shows both total detections and crops saved

### How It Works

When a video is processed:

1. **Video start**: Checks if ROI config is available for the camera
2. **For each frame**:
   - YOLO inference runs on full frame
   - Detection center point is checked: is it within the ROI bbox?
   - If YES: crop is saved, contributes to `crop_count`
   - If NO: detection is skipped, contributes to `filtered_count`
3. **Video end**: Logs show both detected rings and saved crops, plus ROI filtering count

### Example Log Output

```
[CAMERA/video_stem] Using ROI bbox: (248, 1144, 1979, 1273)
[CAMERA/video_stem] Complete: 45/1200 frames, 38 crops saved (7 filtered by ROI) in 23.5s
✓ [1/10] video_stem: 38 crops saved from 45 detections (7 filtered by ROI)
```

### Behavior

- **No ROI configured**: All detections are saved (backward compatible)
- **ROI configured**: Only detections with center within ROI bbox are saved
- **Filtering is by detection center**: A detection must have its center point inside the ROI to be saved
  - This means rings partially outside ROI won't be saved (desired behavior)

### Testing

The implementation includes a test script (`test_roi_filtering.py`) that verifies:
- Detection center within ROI bbox ✓
- Detection center outside ROI bbox (all directions) ✓
- Edge cases and different frame sizes ✓

### ROI Configuration

ROI masks are defined in `config/rois_config.yaml`:
- Hierarchical: `station → region → year → polygon`
- Each polygon defines a region of interest as pixel coordinates
- The ROI bbox is the bounding box of the polygon

Example:
```yaml
FAR3:
  scale:
    "2024":
      polygon:
        - [1560, 1354]
        - [1851, 1067]
        - [1191, 403]
        - [461, 879]
```

To enable: add `roi_config_path: config/rois_config.yaml` to your batch YAML, or it loads automatically.

### Performance Impact

- Minimal: Only adds a simple bbox comparison for each detection
- Detection filtering happens after YOLO (all frames still processed by YOLO)
- Crop saving is what's filtered, not GPU inference

### Key Differences from Previous Approach

| Aspect | Old (Crop Before YOLO) | New (Filter After YOLO) |
|--------|----------------------|------------------------|
| YOLO processing | Only ROI region | Full frame |
| Memory usage | Lower (cropped frames) | Higher (full frames) |
| Ring detection | May miss rings at ROI edges | Detects all rings, saves only those in ROI |
| Speed | Faster YOLO | Standard YOLO speed |
| Accuracy | Potential edge-case misses | Sees full context, more robust |
| Implementation | Crop pre-processing | Post-filtering |

**New approach is recommended** because it provides better detection accuracy while still filtering saved crops to the ROI of interest.
