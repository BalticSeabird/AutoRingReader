# Pipeline Diagnostics - Implementation Summary

## What's Been Added

### 1. **Complete File-Based Diagnostics System** (`pipeline_diagnostics.py`)

The new diagnostics module scans **actual output folders** instead of relying solely on database entries:

- **Step 1 metrics**: Counts PNG files in `step1_ring_crops/{camera}/{video_stem}/`
- **Step 2 metrics**: Counts JPG files in `step2_tag_crops/{camera}/{video_stem}/` and parses filenames
- **Step 3 metrics**: Reads YAML files in `step3_ocr_results/{camera}/{video_stem}/` to extract OCR data

### 2. **CSV Export** (One Row Per Video)

File: `diagnostics.csv` in your output root directory

Columns:
- `camera` - Camera name
- `video_stem` - Video identifier
- `step1_ring_detections`, `step1_status`, `step1_time`
- `step2_ring_crops_count`, `step2_tags_detected`, `step2_status`, `step2_time`
- `step3_final_tag`, `step3_ocr_matches`, `step3_letters_conf`, `step3_numbers_conf`, `step3_status`, `step3_time`

Perfect for importing into Excel, pandas, or other analysis tools.

### 3. **Enhanced Console Reports**

After each step completes:
- **Step 1 Report**: Shows total detections, avg per video, top 15 videos by detections
- **Step 2 Report**: Shows ring crops, tags detected, tags/ring ratio, top 15 videos
- **Step 3 Report**: Shows final tags, OCR confidence scores, all successfully recognized videos

After all steps:
- **Combined Report**: Overall completion rates, aggregate metrics, processing times, top performers

### 4. **Automatic Export**

Pipeline now automatically exports:
- `diagnostics.csv` - CSV format for spreadsheet import
- `diagnostics.json` - Machine-readable JSON

## Key Differences from Previous Version

| Aspect | Old | New |
|--------|-----|-----|
| Data Source | Database only | **Scans output folders** |
| Step 2/3 Metrics | Limited | **Complete (counts actual files, reads YAML)** |
| Step 2 Details | Generic counts | **Parses filenames, counts unique rings/tags** |
| Step 3 Details | Generic counts | **Reads YAML, extracts final tags and confidence scores** |
| Output Format | Console + JSON | **Console + CSV + JSON** |
| CSV Support | No | **Yes - one row per video** |

## Usage

### Automatic (During Pipeline Run)
```bash
python pipeline_orchestrator.py config.yaml
```

Reports print to console after each step, exports saved at end.

### Manual (On Existing Outputs)
```python
from pipeline_diagnostics import PipelineDiagnostics

diag = PipelineDiagnostics(
    output_root="/path/to/output",
    db_path="/path/to/pipeline.db"
)

# Print reports
diag.print_step_report("step1")
diag.print_step_report("step2")
diag.print_step_report("step3")
diag.print_combined_report()

# Export
diag.export_csv("my_diagnostics.csv")
diag.export_json("my_diagnostics.json")
```

## Example CSV Output

```csv
camera,video_stem,step1_ring_detections,step1_status,step2_ring_crops_count,step2_tags_detected,step2_status,step3_final_tag,step3_ocr_matches,step3_letters_conf,step3_numbers_conf,step3_status
FAR3_CUTE,video_001,125,success,125,142,success,A12B34,1,0.923,0.891,success
FAR3_SCALE,video_002,89,success,89,105,success,C56D78,1,0.945,0.912,success
FAR6_SCALE,video_003,0,pending,0,0,pending,,0,,,,pending
```

## Example Console Output

```
====================================================================================================
DIAGNOSTICS: STEP2 - POST-PROCESSING REPORT
====================================================================================================

🎯 STEP 2: TAG DETECTION
  Total Videos: 42
  Total Ring Crops: 1847
  Total Tags Detected: 2156
  Avg Tags per Ring: 1.17

📝 TOP 15 VIDEOS BY TAG DETECTIONS
   1. FAR3_CUTE        video_2025-01-15_10_30     | Rings:  125, Tags:  142 [success] 22.1s
   2. FAR3_SCALE       video_2025-01-15_11_20     | Rings:   89, Tags:  105 [success] 19.3s
   3. FAR6_SCALE       video_2025-01-15_12_10     | Rings:   78, Tags:   91 [success] 18.7s
```

## Performance Impact

- Minimal: Diagnostics run after each step and use file system scanning (fast for typical output sizes)
- Runs in parallel with pipeline (happens after step completes, doesn't slow down processing)

## Files Modified

1. **`pipeline_diagnostics.py`** - Complete rewrite (was generic, now scans output folders)
2. **`pipeline_orchestrator.py`** - Updated initialization + CSV export call
3. **`DIAGNOSTICS.md`** - Documentation

## Next Steps

Simply run your pipeline normally:
```bash
python pipeline_orchestrator.py your_config.yaml
```

Reports will automatically print and CSVs will be saved to your output directory!
