# 🔍 START HERE - Step 1 Ring Detection Troubleshooting

Your problem: **"I see clear ring detections in the video but Step 1 outputs nothing"**

This diagnostic suite will help you find exactly what's wrong.

---

## ⚡ Quick Start (Choose One)

### Option 1: "Just run the quick test" (⏱️ 1 minute)
```bash
python troubleshooting/step1_walkthrough.py \
  path/to/TRI5_SCALE_2025-06-09_10_04.mp4 \
  path/to/ring_detector_model.pt
```

**What you'll see:**
- Frame extraction working? ✓
- YOLO finding rings? ✓
- Passing confidence threshold? ✓

---

### Option 2: "Run full diagnostics" (⏱️ 10-30 minutes)
```bash
python troubleshooting/run_diagnostic.py \
  --video path/to/TRI5_SCALE_2025-06-09_10_04.mp4 \
  --model path/to/ring_detector_model.pt \
  --mode all
```

**What you'll get:**
- Detailed analysis of 100 frames
- Visualizations with detection boxes
- JSON report with all data
- Summary explaining what went wrong

**Output location:**
```
troubleshooting/step1_diagnostic_output/
├── 01_first_frame_raw.jpg              (sanity check)
├── 02_frame_*.jpg                      (detection visualizations)
├── *_diagnostic_summary.txt            (READ THIS FIRST)
└── *_diagnostic_report.json            (full data)
```

---

### Option 3: "Test if ROI is the culprit" (⏱️ 5 minutes)
```bash
python troubleshooting/test_roi_impact.py \
  path/to/TRI5_SCALE_2025-06-09_10_04.mp4 \
  path/to/ring_detector_model.pt
```

**What it checks:**
- Does YOLO find rings in FULL FRAME? (ROI disabled)
- Does YOLO find rings in ROI REGION ONLY? (ROI enabled)
- Are coordinates being transformed correctly?

**If you see:**
- "WITHOUT ROI: 10 detections / WITH ROI: 0 detections"
  → **ROI CONFIGURATION IS TOO AGGRESSIVE** (need to expand the region)

---

## 📋 After Running Diagnostics

### Read the Summary
```bash
cat troubleshooting/step1_diagnostic_output/*_diagnostic_summary.txt
```

You'll see something like:

```
STATISTICS
────────────────────────────────────────────────────────────────────
Total frames in video: 3600
Frames extracted: 100
Frames processed by YOLO: 100
Total detections found: 12              ← Frame 1: Is this > 0?
Detections above threshold: 0           ← Frame 2: Is this > 0?

CONFIDENCE STATISTICS
────────────────────────────────────────────────────────────────────
Min confidence: 0.642
Max confidence: 0.812
Mean confidence: 0.715                  ← Frame 3: Compare to threshold (0.7)

INTERPRETATION
════════════════════════════════════════════════════════════════════
⚠️ NO DETECTIONS ABOVE THRESHOLD!

Possible causes:
1. Confidence threshold is too high      ← THIS IS YOUR ANSWER
2. Model not trained on this type...
3. ROI masking filtering out detections
```

### Answer These Questions

1. **"Total detections found: X"**
   - ❌ If 0: YOLO not detecting → Run ROI test
   - ✓ If > 0: Go to next question

2. **"Detections above threshold: Y"**
   - ❌ If 0: Threshold too high → Lower it
   - ✓ If > 0: Should be saving → Check disk/permissions

3. **Look at visualizations** (02_frame_*.jpg)
   - Green boxes = will be saved
   - Red boxes = filtered by threshold
   - No boxes = not detected

---

## 🔧 Common Fixes

### Problem: "0 detections found"
**Likely cause:** YOLO not seeing rings

**Fix:**
```bash
# Test if ROI is filtering everything
python troubleshooting/test_roi_impact.py \
  path/to/video.mp4 path/to/model.pt
```

If "WITHOUT ROI: detections / WITH ROI: 0":
- Edit `config/rois_config.yaml`
- Expand the TRI5_SCALE region (increase polygon coordinates)

### Problem: "Detections found but all BELOW threshold"
**Likely cause:** Confidence threshold too high

**Fix:**
```bash
# Edit config/batch_step1_tri5.yaml
# Change:
step1:
  conf_threshold: 0.7
# To:
step1:
  conf_threshold: 0.5  ← Lower threshold

# Re-run pipeline
```

### Problem: "Detection coordinates are wrong"
**Likely cause:** ROI offset calculation issue

**Fix:**
```bash
# Test ROI transformation
python troubleshooting/test_roi_impact.py \
  path/to/video.mp4 path/to/model.pt

# Look for: "OUTSIDE ROI (transform error?)"
```

---

## 📂 All Available Tools

| Tool | Purpose | Time | Output |
|------|---------|------|--------|
| **step1_walkthrough.py** | 5-frame interactive test | 1 min | Console |
| **diagnose_step1.py** | 100-frame detailed analysis | 10-30 min | JSON + images + summary |
| **test_roi_impact.py** | Check ROI configuration | 5 min | Console |
| **check_step1_outputs.py** | Inspect saved results | 1 min | Console report |
| **compare_detections.py** | Visual side-by-side comparison | 5 min | JPEG images |
| **run_diagnostic.py** | Master script (runs all) | 30-50 min | All outputs |

---

## 📖 Full Documentation

For detailed troubleshooting guide:
```bash
cat troubleshooting/README.md
cat troubleshooting/QUICK_REFERENCE.txt
```

---

## ❓ What If I Still Don't Know?

Collect this information and it will be obvious:

```bash
# 1. What the diagnostic found
cat troubleshooting/step1_diagnostic_output/*_diagnostic_summary.txt

# 2. Show sample detections
ls -la troubleshooting/step1_diagnostic_output/02_frame_*.jpg

# 3. Your video specs
ffprobe -show_format -show_streams path/to/video.mp4 | head -20

# 4. Your config
cat config/batch_step1_tri5.yaml | head -40
cat config/rois_config.yaml | grep -A 15 "TRI5:"
```

**With this information, the issue becomes obvious.**

---

## ✅ You're Ready!

Pick one of the three quick start options above and run it now.

The diagnostic output will tell you exactly what's wrong.

Good luck! 🚀
