# ROI Filtering - Quick Reference

## How to Use

### 1. Enable ROI Filtering

Add this to your batch YAML config:
```yaml
roi_config_path: config/rois_config.yaml  # Optional, loads automatically if in default location
```

Or ensure `config/rois_config.yaml` exists (it's checked by default).

### 2. What Happens

When processing a video with camera "FAR3_SCALE":

```
Input: Full video frame → YOLO inference on entire frame
Check: Is detection center within FAR3_SCALE ROI bbox?
  ✓ YES → Save crop image (contributes to crop_count)
  ✗ NO → Skip crop (contributes to filtered_count)
Output: Only crops inside ROI are saved
```

### 3. Example Log Output

```
[FAR3_SCALE/video_2024] Using ROI bbox: (461, 403, 1851, 1354)
[FAR3_SCALE/video_2024] Complete: 45/1200 frames, 38 crops saved (7 filtered by ROI) in 23.5s
✓ [1/10] video_2024: 38 crops saved from 45 detections (7 filtered by ROI)
```

## Configuration

### ROI Definitions (config/rois_config.yaml)

The file defines ROIs hierarchically:
```yaml
rois:
  STATION:              # e.g., FAR3, FAR6, ROST2, TRI5
    region:             # e.g., cute, scale
      "YEAR":           # e.g., "2024", "2025" or "default"
        polygon:        # List of [x, y] pixel coordinates
          - [x1, y1]
          - [x2, y2]
          - [x3, y3]
          - [x4, y4]
```

### Adding New ROIs

Example: Add a new ROI for TRI5_SCALE in 2025:

```yaml
TRI5:
  scale:
    "2025":
      polygon:
        - [400, 250]
        - [620, 250]
        - [620, 480]
        - [400, 480]
    default:
      polygon:
        - [399, 259]
        - [614, 259]
        - [614, 468]
        - [399, 468]
```

### Current Supported Cameras

| Station | Region | Years Available |
|---------|--------|-----------------|
| FAR3 | cute | default (uses 2024 baseline) |
| FAR3 | scale | 2023, 2024, 2025, default |
| FAR6 | scale | 2023, 2024 |
| ROST2 | scale | default |
| TRI5 | scale | 2025, default |

## Behavior Details

### Detection Filtering

A detection is included if:
- Detection center point (x_norm * width, y_norm * height) is within ROI bbox
- ROI bbox format: (xmin, ymin, xmax, ymax) in pixel coordinates

### When ROI is Disabled

If `roi_config_path` is not provided or ROI file is missing:
- All detections above confidence threshold are saved
- `filtered_count` remains 0
- Logs show: "X crops saved" (without filtering info)

### Frame Processing

- YOLO processes all frames at full resolution
- Only crop SAVING is filtered by ROI
- Ring detection itself is not affected by ROI

## Diagnostics

### Check ROI Loading

```python
from pipeline_config import ROIConfig

roi = ROIConfig("config/rois_config.yaml")

# Check if ROI loaded for a camera
bbox = roi.get_roi_bbox("FAR3_SCALE", year=2024)
print(f"FAR3_SCALE ROI: {bbox}")  # (461, 403, 1851, 1354)
```

### Verify Configuration

Run the included integration tests:
```bash
python3 test_roi_filtering.py          # Unit tests (6 scenarios)
python3 test_roi_integration.py        # Integration tests (5 cameras)
```

## Performance Impact

- **GPU inference**: No change (YOLO processes full frame)
- **Crop saving**: Minimal overhead (simple bbox check per detection)
- **Memory**: No change (full frames processed as before)
- **Speed**: ~1-2% overhead for ROI checking (negligible)

## Troubleshooting

### No crops saved but detections found

Check:
1. Is ROI bbox correct? Log shows: `Using ROI bbox: (x1, y1, x2, y2)`
2. Are detected rings actually in the ROI? Manually verify ROI polygon covers intended region
3. Check filtered_count in logs - rings may be outside ROI by design

### All detections filtered

Check:
1. ROI bbox coordinates are in pixel space (not normalized)
2. ROI coordinates match camera resolution (typically 1920x1440 or similar)
3. Compare ROI bbox to actual frame size

### ROI not loading

Error message in logs: `No ROI found, will save all detections`

Check:
1. File exists: `config/rois_config.yaml`
2. Camera name matches YAML station/region format (e.g., "FAR3_CUTE")
3. Station and region keys in YAML are correct case (uppercase station, lowercase region)
