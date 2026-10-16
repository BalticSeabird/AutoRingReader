#!/bin/bash

# TROUBLESHOOTING QUICK START
# Run this to diagnose issues with the AutoRingReader pipeline

set -e

echo "========================================================================"
echo "  AutoRingReader Pipeline Diagnostic Tool"
echo "========================================================================"
echo ""

# Check for required arguments
if [ $# -lt 2 ]; then
    echo "USAGE:"
    echo "  bash troubleshooting/quickstart.sh <video_path> <model_path> [output_root]"
    echo ""
    echo "EXAMPLE:"
    echo "  bash troubleshooting/quickstart.sh \\"
    echo "    path/to/TRI5_SCALE_2025-06-09_10_04.mp4 \\"
    echo "    path/to/ring_detector_model.pt \\"
    echo "    output_root"
    echo ""
    exit 1
fi

VIDEO_PATH="$1"
MODEL_PATH="$2"
OUTPUT_ROOT="${3:-.output}"

echo "Video: $VIDEO_PATH"
echo "Model: $MODEL_PATH"
echo "Output root: $OUTPUT_ROOT"
echo ""

# Check files exist
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Video file not found: $VIDEO_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    exit 1
fi

echo "✓ Files exist"
echo ""

# Step 1: Walkthrough (quick understanding)
echo "========================================================================"
echo "STEP 1: Quick Walkthrough (5 frames)"
echo "========================================================================"
echo "Testing basic frame extraction and YOLO inference on 5 sample frames..."
echo ""

python troubleshooting/step1_walkthrough.py "$VIDEO_PATH" "$MODEL_PATH" --sample-frames 5

echo ""
echo "✓ Walkthrough complete. Check the output above for basic diagnostics."
echo ""

# Step 2: Full diagnostic (100 frames)
echo "========================================================================"
echo "STEP 2: Full Diagnostic (first 100 frames)"
echo "========================================================================"
echo "Running detailed trace with visualizations..."
echo ""

python troubleshooting/diagnose_step1.py "$VIDEO_PATH" "$MODEL_PATH" --max-frames 100

echo ""
echo "✓ Diagnostic output saved to: troubleshooting/step1_diagnostic_output/"
echo ""

# Step 3: Check pipeline outputs (if they exist)
if [ -d "$OUTPUT_ROOT/step1_ring_crops" ]; then
    echo "========================================================================"
    echo "STEP 3: Check Pipeline Outputs"
    echo "========================================================================"
    echo "Analyzing what was saved in the output directory..."
    echo ""

    python troubleshooting/check_step1_outputs.py "$OUTPUT_ROOT"

    echo ""
    echo "✓ Output analysis complete"
else
    echo "⚠️  No pipeline output directory found yet."
    echo "    This is normal if you haven't run the full pipeline."
    echo "    Run pipeline first, then re-run this script."
fi

echo ""
echo "========================================================================"
echo "DIAGNOSTICS COMPLETE"
echo "========================================================================"
echo ""
echo "📋 CHECK THESE FILES:"
echo ""
echo "  Human-readable summary:"
echo "    cat troubleshooting/step1_diagnostic_output/*_diagnostic_summary.txt"
echo ""
echo "  Full JSON data:"
echo "    cat troubleshooting/step1_diagnostic_output/*_diagnostic_report.json"
echo ""
echo "  Visualization images:"
echo "    ls -la troubleshooting/step1_diagnostic_output/02_frame_*"
echo ""
echo "📖 READ THIS:"
echo "    cat troubleshooting/README.md"
echo ""
echo "Need help? Check the README for detailed troubleshooting guide!"
echo ""
