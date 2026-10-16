# Pipeline Diagnostics System

## Overview

The pipeline diagnostics system **scans output folders** to collect comprehensive performance metrics for each video as it progresses through Steps 1-3. Metrics are reported after each step and aggregated at the end.

## What's Tracked

### Per-Video Metrics

#### Step 1: Ring Detection
- **Ring Detections**: Count of ring crop files generated (step1_ring_crops/{camera}/{video_stem}/*.png)
- **Mean Confidence**: From database (if available)
- **Status**: success/pending/error (from database)
- **Processing Time**: From database (if available)

#### Step 2: Tag Detection  
- **Ring Crops**: Count of unique ring frames processed
- **Tags Detected**: Count of tag crop files generated (step2_tag_crops/{camera}/{video_stem}/*.jpg)
- **Mean Confidence**: From database (if available)
- **Status**: success/pending/error (from database)
- **Processing Time**: From database (if available)

#### Step 3: Text Recognition (OCR)
- **Final Tag**: The recognized ring ID (from YAML output)
- **OCR Matches**: 1 if final tag exists, 0 otherwise
- **Letters Confidence**: Mean confidence for letter characters (from YAML)
- **Numbers Confidence**: Mean confidence for number characters (from YAML)
- **Status**: success/pending/error (from database)
- **Processing Time**: From database (if available)

### Data Sources

The diagnostics scans **actual output folders** rather than relying solely on database entries:

```
output_root/
├── step1_ring_crops/{camera}/{video_stem}/*.png          (Ring crops)
├── step2_tag_crops/{camera}/{video_stem}/*.jpg           (Tag crops)
└── step3_ocr_results/{camera}/{video_stem}/*.yaml        (OCR results)
```

This means:
- **Step 1 metrics**: Based on counting PNG files in step1_ring_crops
- **Step 2 metrics**: Based on counting JPG files and parsing filenames in step2_tag_crops
- **Step 3 metrics**: Based on parsing YAML files in step3_ocr_results

## Usage

### Automatic Reporting

Diagnostics run automatically after each step completes during pipeline execution:

```bash
python pipeline_orchestrator.py config.yaml
```

Output includes:
- Step report after Step 1 completes (ring detection summary)
- Step report after Step 2 completes (tag detection summary)
- Step report after Step 3 completes (OCR results)
- Combined report after all steps finish
- CSV export to `output_root/diagnostics.csv`
- JSON export to `output_root/diagnostics.json`

### Programmatic Access

```python
from pipeline_diagnostics import PipelineDiagnostics

# Initialize (scans output folders on each call)
diagnostics = PipelineDiagnostics(
    output_root="/path/to/output",
    db_path="/path/to/pipeline.db"
)

# Scan all videos and get metrics
videos = diagnostics.scan_all_videos()  # Dict[video_stem -> VideoMetrics]

# Print formatted reports
diagnostics.print_step_report("step1")
diagnostics.print_step_report("step2")
diagnostics.print_step_report("step3")
diagnostics.print_combined_report()

# Export to files
diagnostics.export_csv("diagnostics.csv")
diagnostics.export_json("diagnostics.json")
```

## Output Formats

### Console Reports

After each step completes, a formatted report prints showing:
- Summary statistics
- Top 15 videos by key metric
- Status of each video

#### Step 1 Report Example
```
====================================================================================================
DIAGNOSTICS: STEP1 - POST-PROCESSING REPORT
====================================================================================================

🎯 STEP 1: RING DETECTION
  Total Videos: 42
  Total Ring Detections: 1847
  Avg Detections per Video: 43.9

📝 TOP 15 VIDEOS BY DETECTIONS
   1. FAR3_CUTE        video_2025-01-15_10_30     | Detections:  125 [success] 15.3s
   2. FAR3_SCALE       video_2025-01-15_11_20     | Detections:   89 [success] 13.2s
   3. FAR6_SCALE       video_2025-01-15_12_10     | Detections:   78 [success] 12.8s
   ...
```

#### Step 2 Report Example
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
   ...
```

#### Step 3 Report Example
```
====================================================================================================
DIAGNOSTICS: STEP3 - POST-PROCESSING REPORT
====================================================================================================

🎯 STEP 3: TEXT RECOGNITION (OCR)
  Total Videos with Results: 35
  Total OCR Matches: 35

📝 OCR RESULTS
   1. FAR3_CUTE        | Final Tag: A12B34               | L:0.923 N:0.891 [success] 18.5s
   2. FAR3_SCALE       | Final Tag: C56D78               | L:0.945 N:0.912 [success] 17.2s
   ...
```

#### Combined Report Example
```
====================================================================================================
COMPREHENSIVE DIAGNOSTICS: ALL STEPS COMBINED
====================================================================================================

📊 OVERALL SUMMARY
  Total Videos: 42
  Step 1 Complete: 42/42 (100.0%)
  Step 2 Complete: 42/42 (100.0%)
  Step 3 Complete: 35/42 (83.3%)
  All Steps Complete: 35/42 (83.3%)

📈 AGGREGATE METRICS
  Step 1 - Total Ring Detections: 1847
  Step 2 - Total Tag Detections: 2156
  Step 3 - Total OCR Matches: 35

⏱️  PROCESSING TIMES
  Step 1: 640.5s total, 15.3s avg
  Step 2: 925.8s total, 22.0s avg
  Step 3: 647.2s total, 18.5s avg

🏆 TOP PERFORMERS
  Most Detections: FAR3_CUTE - video_2025-01-15_10_30 (125 detections)
  Most Tags: FAR3_CUTE - video_2025-01-15_10_30 (142 tags)
  OCR Results: 35 videos with final tags

====================================================================================================
```

### CSV Export

File: `output_root/diagnostics.csv`

One row per video with columns:
- `camera`: Camera name (e.g., FAR3_CUTE)
- `video_stem`: Video identifier
- `step1_ring_detections`: Number of rings detected
- `step1_ring_conf_mean`, `step1_status`, `step1_time`
- `step2_ring_crops_count`: Number of unique ring frames
- `step2_tags_detected`: Number of tags detected
- `step2_tag_conf_mean`, `step2_status`, `step2_time`
- `step3_final_tag`: Final recognized tag ID
- `step3_ocr_matches`: 1 if OCR succeeded, 0 otherwise
- `step3_letters_conf`, `step3_numbers_conf`
- `step3_status`, `step3_time`

Example:
```csv
camera,video_stem,step1_ring_detections,step1_ring_conf_mean,step1_status,step1_time,step2_ring_crops_count,step2_tags_detected,step2_tag_conf_mean,step2_status,step2_time,step3_final_tag,step3_ocr_matches,step3_letters_conf,step3_numbers_conf,step3_status,step3_time
FAR3_CUTE,video_2025-01-15_10_30,125,0.918,success,15.3,125,142,0.905,success,22.1,A12B34,1,0.923,0.891,success,18.5
FAR3_SCALE,video_2025-01-15_11_20,89,0.931,success,13.2,89,105,0.912,success,19.3,C56D78,1,0.945,0.912,success,17.2
```

### JSON Export

File: `output_root/diagnostics.json`

Complete metrics in structured JSON format:
```json
{
  "total_videos": 42,
  "videos": [
    {
      "camera": "FAR3_CUTE",
      "video_stem": "video_2025-01-15_10_30",
      "step1_ring_detections": 125,
      "step1_ring_conf_mean": 0.918,
      "step1_status": "success",
      "step1_time": 15.3,
      "step2_ring_crops_count": 125,
      "step2_tags_detected": 142,
      "step2_tag_conf_mean": 0.905,
      "step2_status": "success",
      "step2_time": 22.1,
      "step3_final_tag": "A12B34",
      "step3_ocr_matches": 1,
      "step3_letters_conf": 0.923,
      "step3_numbers_conf": 0.891,
      "step3_status": "success",
      "step3_time": 18.5
    },
    ...
  ]
}
```

## Key Features

### 1. File-Based Metrics
- Counts actual files in output directories
- Parses filenames to extract frame/tag information
- Reads YAML files to get OCR results
- **No reliance on database for Step 2-3 core metrics**

### 2. Summary Statistics
- Totals across all videos
- Per-video rankings
- Success/completion rates
- Processing time breakdowns

### 3. Multiple Output Formats
- Console reports (immediately after each step)
- CSV file (importable to Excel/pandas)
- JSON file (machine-readable)

### 4. Flexibility
- Scans on-demand (no upfront indexing)
- Works incrementally as files are created
- Can be called independently of pipeline

## Troubleshooting

### "No Step X data found"
- Check that output folders exist (`step1_ring_crops`, `step2_tag_crops`, `step3_ocr_results`)
- Verify files are being generated (PNG, JPG, YAML respectively)
- Check file permissions

### CSV shows empty values
- `step1_ring_conf_mean`: Only populated if database has records
- `step3_final_tag`: Only populated if YAML output was generated
- Check database and output directories

### Low detection counts
- **Step 1**: No PNG files found in `step1_ring_crops/{camera}/{video_stem}/`
- **Step 2**: No JPG files found in `step2_tag_crops/{camera}/{video_stem}/`
- **Step 3**: No YAML files found in `step3_ocr_results/{camera}/{video_stem}/`

## Integration Points

- **Pipeline Config**: Provides output_root path
- **Pipeline Logger**: Database used for status/timing info
- **Step Outputs**: Scans file system (no code integration needed)

## Files Modified/Created

- `pipeline_diagnostics.py`: New module (complete rewrite for file scanning)
- `pipeline_orchestrator.py`: Updated to initialize diagnostics with output_root, export CSV
- `DIAGNOSTICS.md`: This documentation
