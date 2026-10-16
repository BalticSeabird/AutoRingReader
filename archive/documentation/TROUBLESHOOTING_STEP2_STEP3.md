# Pipeline Troubleshooting: Step 2 & Step 3 Not Producing Output

## Problem Summary
When running the pipeline, only Step 1 (ring_crops) was being generated. Steps 2 and 3 reported:
- **Step 2**: "0 tags from X crops" for every video  
- **Step 3**: No output directory created

## Root Cause
The tag detector model is a **detection model** (not segmentation), but the code was looking for **segmentation masks**.

- Model type: `detect` (Detection only)
- Code expected: `res.masks` (requires segmentation model)
- Result: No tags were extracted because `res.masks` was always `None`

### Technical Details
In `tag_detector.py` line 82-83:
```python
masks = res.masks
if masks is not None:  # This was NEVER true for detection models
    # Extract tags using mask coordinates
```

Detection models don't produce masks, so this condition always failed, and **zero tags were returned for all crops**.

## Solution Applied
Updated `_predict_tag_entries()` in `tag_detector.py` to handle both:

1. **Segmentation models** (original path): Extract tags from mask outlines
2. **Detection models** (new path): Extract tags from bounding box regions

### Changes Made

**File**: `/home/jonas/Documents/vscode/AutoRingReader/tag_detector.py`

**Key changes**:
- Initialize `tags = []` before try block (was causing NameError on exception)
- Convert image to numpy once: `image_np = np.array(image)`
- Add detection branch: `elif res.boxes is not None and len(res.boxes) > 0:`
- Extract tag crops from bounding box coordinates using resize instead of perspective transform

**Before**: ~160 lines, only handled segmentation
**After**: ~170 lines, handles both segmentation and detection

## Verification

Tested with sample crop `000408_ring.png`:
```
Before fix: 0 tags
After fix:  2 tags detected (conf: 0.966, 0.886)
```

Batch test on 10 crops: **18 tags detected from 10 crops (90% detection rate)**

## Next Steps

1. **Re-run the pipeline**:
   ```bash
   cd /home/jonas/Documents/vscode/AutoRingReader
   python pipeline_orchestrator.py config/batch_pipeline_test.yaml
   ```

2. **Expected results**:
   - Step 2 should now create `/step2_tag_crops/{camera}/{video}/*.jpg` files
   - Step 3 should now create `/step3_ocr_results/{camera}/{video}/*.yaml` files

3. **Verify output**:
   ```bash
   ls /mnt/BSP_NAS2_work/ring_reader/test_results/step2_tag_crops/
   ls /mnt/BSP_NAS2_work/ring_reader/test_results/step3_ocr_results/
   ```

## Notes
- The confidence threshold (0.5 in batch config) is appropriate for this model
- Most crops yield 2 tags (matches `max_det: 2` in step2 config)
- Some crops (like 000386_ring.png) have low confidence detections filtered at threshold 0.5
- With lower threshold (0.1), 90% of crops have detections; at 0.5, ~80% pass
