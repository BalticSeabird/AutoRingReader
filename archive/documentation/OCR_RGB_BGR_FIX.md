# OCR Quality Issue: RGB/BGR Channel Mismatch Fix

## Problem Summary

Poor OCR readings for certain stations (e.g., BJORN3TRI3_SCALE) despite good Step 2 tag detections were caused by **color channel corruption** in the tag crop pipeline.

## Root Cause

### Data Flow Before Fix:
1. **tag_detector.py** (line 161): Creates tag crops as PIL Images in **RGB format**
   ```python
   Image.fromarray(img_np_aligned).convert("RGB")  # ← RGB
   ```

2. **step2_tag_detection.py** (line 190): Converts PIL → numpy array (**still RGB**)
   ```python
   tag_crop_np = np.array(tag_crop)  # ← RGB data, shape (H, W, 3)
   ```

3. **step2_tag_detection.py** (line 290) - **BUG**: Saves with cv.imwrite
   ```python
   cv.imwrite(str(tag_path), result['tag_crop'])
   ```
   - **Problem**: cv.imwrite assumes **BGR** format
   - **Result**: RGB channels saved in wrong order (R↔B swapped)

4. **step3_text_recognition.py** (line 192): Reads with cv.imread
   ```python
   img = cv.imread(str(crop_path))  # ← Returns BGR (but corrupted!)
   ```

5. **step3_text_recognition.py** (line 194): Double conversion
   ```python
   img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # ← Double corruption!
   ```

### Effect:
- Channels corrupted/misaligned
- Image quality degraded
- OCR model receives malformed input → poor predictions

## Fix Applied

In **step2_tag_detection.py**, line 290-292:

```python
# BEFORE (incorrect):
cv.imwrite(str(tag_path), result['tag_crop'])

# AFTER (correct):
# Convert RGB to BGR for cv.imwrite (which expects BGR)
tag_crop_bgr = cv.cvtColor(result['tag_crop'], cv.COLOR_RGB2BGR)
cv.imwrite(str(tag_path), tag_crop_bgr)
```

### New Data Flow (After Fix):
1. tag_detector creates PIL Image in RGB ✓
2. step2 converts to numpy (RGB) ✓
3. **step2 converts RGB→BGR before saving** ✓ (NEW)
4. step3 reads as BGR ✓
5. step3 converts BGR→RGB ✓
6. OCR model receives correct RGB input ✓

## Expected Improvements

- ✅ Correct color channel ordering preserved
- ✅ Better image quality for OCR model
- ✅ Improved accuracy on BJORN3TRI3_SCALE and other stations
- ✅ Better confidence scores from text recognition

## Testing

To verify the fix works:

```bash
# Run with --force to reprocess all videos, including BJORN3TRI3_SCALE
python3 pipeline_orchestrator.py batch/your_config.yaml --force
```

Compare OCR results (confidence scores, detection rates) with previous runs. You should see:
- Higher confidence scores for letters/numbers
- More successful tag detections in Step 3
- Better majority voting results (fewer ambiguities)

## Files Modified

- `step2_tag_detection.py`: Added RGB→BGR conversion before cv.imwrite (line 291)
