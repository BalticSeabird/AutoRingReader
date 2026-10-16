"""
TROUBLESHOOTING GUIDE FOR AUTORINGREADER PIPELINE

The pipeline has 3 steps, each building on the previous:
  1. Ring Detection (finds metal rings in video frames)
  2. Tag Detection (finds alphanumeric tags on the rings)
  3. Text Recognition (reads/recognizes the tag text)

This guide helps diagnose issues at each stage.

=============================
  QUICK START DIAGNOSTICS
=============================

├── Step 1: Ring Detection
│   ├─ diagnose_step1.py ............. Detailed trace of ring detection
│   ├─ step1_walkthrough.py .......... Interactive walkthrough (5 frames)
│   ├─ check_step1_outputs.py ........ Inspect what was actually saved
│   └─ compare_detections.py ......... Compare system vs visual detections
│
├── Step 2: Tag Detection
│   ├─ diagnose_step2.py ............. Detailed trace of tag detection
│   ├─ check_step2_outputs.py ........ Inspect tag detection outputs
│   └─ compare_tag_detections.py ..... Compare system vs visual tag detections
│
└── Step 3: Text Recognition
    ├─ diagnose_step3.py ............. Detailed trace of OCR
    └─ check_step3_outputs.py ........ Inspect OCR recognition outputs

=============================
  YOUR ISSUE: "I see rings but Step 1 outputs nothing"
=============================

This suggests that either:
  1. The YOLO model isn't detecting rings in THESE videos
  2. Confidence threshold is too high
  3. ROI masking is filtering out all detections
  4. Frame extraction or YOLO inference is failing
  5. Coordinates are wrong or crops aren't being saved

DIAGNOSIS STEPS (do these IN ORDER):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1A: UNDERSTAND WHAT STEP 1 DOES (5-minute overview)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run the walkthrough on a small sample (5 frames):

  python troubleshooting/step1_walkthrough.py \\
    path/to/TRI5_SCALE_2025-06-09_10_04.mp4 \\
    path/to/ring_detector_model.pt

This will show you exactly:
  - How many frames are extracted
  - What YOLO finds in each frame
  - Which detections pass the confidence threshold
  - What would be saved

Expected output:
  ✓ You see "Detection found!" messages with confidence values
  ✓ You see "Passes threshold: True/False"
  ✓ You know WHY detections are being saved or filtered

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1B: RUN DETAILED DIAGNOSTIC (5-30 minutes depending on video length)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run the full diagnostic on first 100 frames:

  python troubleshooting/diagnose_step1.py \\
    path/to/TRI5_SCALE_2025-06-09_10_04.mp4 \\
    path/to/ring_detector_model.pt \\
    --confidence 0.7 \\
    --max-frames 100

This outputs:
  ✓ troubleshooting/step1_diagnostic_output/
      ├─ 01_first_frame_raw.jpg ............. Raw first frame (sanity check)
      ├─ 02_frame_XXXXXX_detections.jpg .... Detection visualizations
      ├─ TRI5_..._diagnostic_report.json .. Full detection data
      └─ TRI5_..._diagnostic_summary.txt .. Human-readable summary

Examine the summary:
  cat troubleshooting/step1_diagnostic_output/TRI5_..._diagnostic_summary.txt

Look for:
  ✓ "Total detections found:" - Should not be 0
  ✓ "Detections above threshold:" - This is what gets saved
  ✓ "Confidence stats:" - Min/Max/Mean values
  ✓ "INTERPRETATION:" section - Explains what went wrong

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1C: CHECK WHAT WAS ACTUALLY SAVED (after running full pipeline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After running the full pipeline on your video, check the outputs:

  python troubleshooting/check_step1_outputs.py output_root

This shows:
  ✓ How many videos were processed
  ✓ How many detections each got
  ✓ How many videos got ZERO detections (this is your problem!)
  ✓ How many crop files were saved

Output example:
  Camera: TRI5_SCALE
  └─ Video TRI5_SCALE_2025-06-09_10_04: 0 detections (processed 3600 frames)
  
If this shows "0 detections", it means either:
  - No detections found by YOLO at confidence threshold
  - Detections found but below threshold
  - Issue with frame extraction or YOLO inference

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1D: VISUALLY COMPARE (debug coordinate/filtering issues)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create side-by-side comparison images:

  python troubleshooting/compare_detections.py \\
    path/to/TRI5_SCALE_2025-06-09_10_04.mp4 \\
    output_root/step1_ring_crops/TRI5_SCALE/TRI5_SCALE_2025-06-09_10_04/_metadata.json \\
    --sample-rate 10

This outputs:
  ✓ troubleshooting/comparison_output/
      └─ comparison_frame_XXXXXX.jpg ... side-by-side with detected boxes

Look at the images:
  - Left side: Original frame (what you see)
  - Right side: Same frame with green/red boxes showing detections
    - GREEN = confidence >= threshold (will be saved)
    - RED = confidence < threshold (filtered out)

Questions to answer:
  □ Are the green boxes in the right place?
  □ Is the model detecting rings but they're all red (below threshold)?
  □ Is the model not detecting anything at all?
  □ Are coordinates wildly off?

==============================
  COMMON PROBLEMS & FIXES
==============================

PROBLEM: "No detections found at all"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cause: YOLO model isn't detecting rings in this video.

Check:
  1. Is the model trained on videos from THIS camera/station?
  2. Are the rings actually visible in the video? (Check first_frame_raw.jpg)
  3. What's the video resolution? (from diagnose_step1.py output)
  4. Is the video compressed (h.264/h.265) or raw?

Fix:
  □ Try lower confidence threshold (0.5 or 0.4 instead of 0.7)
    -> Run diagnose_step1.py with --confidence 0.5
  
  □ Check if ROI masking is filtering everything out
    -> Temporarily disable ROI (remove roi_config from batch YAML)
  
  □ Verify the model file is not corrupted
    -> Re-download or verify with model checksum


PROBLEM: "Detections found but all below threshold"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cause: Model sees rings but isn't confident (probably low-res or unclear rings).

Evidence:
  diagnostic_summary.txt shows "Detections found: 50" but "Above threshold: 0"
  Confidence stats show mean < 0.7

Fix:
  □ LOWER the confidence threshold in batch YAML:
    
    step1:
      conf_threshold: 0.5  # Changed from 0.7
  
  □ Re-run pipeline with new threshold
  
  □ Check confidence distribution in diagnostic_summary.txt
    -> If mean is around 0.5-0.6, threshold of 0.5 is reasonable


PROBLEM: "Detections above threshold but no crops saved"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cause: Detections exist but crop saving failed (usually I/O issue).

Evidence:
  diagnostic_summary.txt shows "Detections above threshold: 10" 
  But _metadata.json shows empty or no crop files exist

Fix:
  □ Check output directory permissions
  □ Check available disk space
  □ Check that step1_ring_crops directory is writable


PROBLEM: "Detection coordinates are wrong"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cause: Bounding boxes are in wrong place (usually ROI offset issue).

Evidence:
  compare_detections.py shows green boxes not on the rings
  Boxes appear shifted or offset

Fix:
  □ Check ROI configuration in rois_config.yaml
    -> ROI coordinates might be wrong
    -> Try disabling ROI entirely (set roi_config_path to null in batch YAML)
  
  □ Check video resolution matches expected
    -> Model trained on 1920x1080?
    -> Video is different resolution?


PROBLEM: "Step 1 runs but crashes halfway"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cause: Usually GPU out-of-memory (OOM) or timeout.

Evidence:
  CUDA out of memory error in logs
  Or: "Timeout waiting for result"

Fix:
  □ Reduce batch_size in batch YAML:
    
    step1:
      batch_size: 8  # Changed from 16
  
  □ Ensure only one GPU is processing the video
  
  □ Check available GPU memory with: nvidia-smi


==============================
  NEXT STEPS AFTER STEP 1 FIXED
==============================

Once Step 1 is producing detections:

1. Run Step 2 (Tag Detection):
   python troubleshooting/diagnose_step2.py \\
     output_root/step1_ring_crops/CAMERA/VIDEO/_metadata.json \\
     path/to/tag_detector_model.pt

2. Check Step 2 outputs:
   python troubleshooting/check_step2_outputs.py output_root

3. Run Step 3 (Text Recognition):
   python troubleshooting/diagnose_step3.py \\
     output_root/step2_tag_crops/CAMERA/VIDEO/_metadata.json

==============================
  FILE STRUCTURE EXPLAINED
==============================

After Step 1 completes, you should have:

output_root/
└── step1_ring_crops/
    └── CAMERA_NAME/              ← Your camera (e.g., TRI5_SCALE)
        └── VIDEO_STEM/           ← Video name (e.g., TRI5_SCALE_2025-06-09_10_04)
            ├── _metadata.json    ← ← ← CHECK THIS FIRST
            ├── 000001_ring.png
            ├── 000015_ring.png
            ├── 000042_ring.png
            └── ...more crop files...

THE KEY FILES:
  _metadata.json
    └─ Contains list of ALL detections (detected or not)
    └─ Each detection has: frame_idx, detected, bbox, confidence
    └─ Use this to understand what the pipeline saw

  XXXXXX_ring.png
    └─ 224x224 crop of detected ring
    └─ Only exists if detection passed confidence threshold
    └─ These become input to Step 2

If _metadata.json exists but crop files are empty:
  → Detections were found but below threshold
  → Lower the threshold or check detector confidence

If _metadata.json doesn't exist:
  → Video wasn't even processed check logs for errors

If _metadata.json is empty:
  → Video was processed but no detections found
  → This is your problem - diagnose with the scripts above

==============================
  QUESTIONS TO ASK YOURSELF
==============================

When troubleshooting, ask:

1. Can I VISUALLY see the ring in the video?
   → If NO: This is not a pipeline issue, it's a video issue
   → If YES: Go to question 2

2. Does the diagnostic show YOLO detects the ring?
   → Look at "Total detections found:" in diagnostic_summary
   → If NO: Model issue - is it trained on this type of video?
   → If YES: Go to question 3

3. Are detections above the confidence threshold?
   → Look at "Detections above threshold:" in diagnostic_summary
   → If NO: Lower threshold (currently 0.7, try 0.5)
   → If YES: Go to question 4

4. Are crop files being saved?
   → Count *.png files in output directory
   → If NO: I/O or disk space issue
   → If YES: Pipeline is working correctly!

=============================
  STILL STUCK?
=============================

Collect the following and send to developers:

1. Log from your pipeline run:
   tail -100 log/*.txt

2. Diagnostic output:
   tar czf troubleshooting.tar.gz troubleshooting/step1_diagnostic_output/
   
3. Check outputs summary:
   python troubleshooting/check_step1_outputs.py output_root > step1_check.txt

4. Sample comparison images:
   ls -la troubleshooting/comparison_output/

5. Video file info:
   ffprobe -show_format -show_streams path/to/video.mp4

6. Model info:
   ls -la path/to/ring_detector_model.pt

With these, developers can quickly identify the issue!

"""

# This is just documentation - create as a file
