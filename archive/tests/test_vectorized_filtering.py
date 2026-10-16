#!/usr/bin/env python3
"""Test optimized vectorized ROI filtering."""

import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent))

from step1_ring_detection import _filter_detections_by_roi_batch


def create_mock_result(has_boxes=True, confidence=0.8, x_norm=0.5, y_norm=0.5):
    """Create a mock YOLO result object."""
    result = Mock()
    if has_boxes:
        box = Mock()
        box.xywhn = [[x_norm, y_norm, 0.1, 0.1]]
        box.conf = [confidence]
        result.boxes = [box]
    else:
        result.boxes = []
    return result


def test_vectorized_roi_filtering():
    """Test the vectorized ROI filtering function."""

    print("Testing Vectorized ROI Filtering")
    print("=" * 60)

    # Create mock frames and results for a batch
    frame_height, frame_width = 1440, 1920

    # Test batch: 4 frames
    batch_results = [
        (0, create_mock_result(has_boxes=True, confidence=0.85, x_norm=0.5, y_norm=0.5), None),    # Center (in ROI)
        (1, create_mock_result(has_boxes=True, confidence=0.90, x_norm=0.1, y_norm=0.1), None),    # Top-left (outside ROI)
        (2, create_mock_result(has_boxes=False), None),                                             # No detection
        (3, create_mock_result(has_boxes=True, confidence=0.65, x_norm=0.5, y_norm=0.5), None),    # Low confidence
    ]

    frame_dims = {'height': frame_height, 'width': frame_width}

    # ROI bbox: (461, 403, 1851, 1354) - roughly right 2/3 of frame, middle
    roi_bbox = (461, 403, 1851, 1354)
    conf_threshold = 0.7

    # Run vectorized filtering
    filtered = _filter_detections_by_roi_batch(batch_results, frame_dims, roi_bbox, conf_threshold)

    print(f"\nInput batch size: {len(batch_results)} frames")
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    print(f"ROI bbox: {roi_bbox}")
    print(f"Confidence threshold: {conf_threshold}")
    print()

    # Verify results
    assert len(filtered) == len(batch_results), "Filtered list should have same length as input"
    print("✓ Output batch has correct size")

    # Frame 0: center, high conf, in ROI -> should_save_crop = True
    frame_idx, result, frame, should_save, bbox, conf = filtered[0]
    assert frame_idx == 0, f"Frame index mismatch: {frame_idx}"
    assert should_save == True, f"Frame 0 should be saved (in ROI, high conf), got {should_save}"
    assert conf == 0.85, f"Confidence mismatch: {conf}"
    print("✓ Frame 0 (center, high conf): save=True")

    # Frame 1: top-left, high conf, outside ROI -> should_save_crop = False
    frame_idx, result, frame, should_save, bbox, conf = filtered[1]
    assert frame_idx == 1, f"Frame index mismatch: {frame_idx}"
    assert should_save == False, f"Frame 1 should NOT be saved (outside ROI), got {should_save}"
    assert conf == 0.90, f"Confidence should still be tracked: {conf}"
    print("✓ Frame 1 (top-left, outside ROI): save=False")

    # Frame 2: no detection -> should_save_crop = False
    frame_idx, result, frame, should_save, bbox, conf = filtered[2]
    assert frame_idx == 2, f"Frame index mismatch: {frame_idx}"
    assert should_save == False, f"Frame 2 should NOT be saved (no detection), got {should_save}"
    print("✓ Frame 2 (no detection): save=False")

    # Frame 3: center, low conf -> should_save_crop = False (below confidence threshold)
    frame_idx, result, frame, should_save, bbox, conf = filtered[3]
    assert frame_idx == 3, f"Frame index mismatch: {frame_idx}"
    assert should_save == False, f"Frame 3 should NOT be saved (below confidence), got {should_save}"
    assert conf == 0.65, f"Confidence should be tracked: {conf}"
    print("✓ Frame 3 (center, low conf): save=False")

    print()
    print("=" * 60)
    print("✓ All vectorized filtering tests passed!")
    print()
    print("Performance benefit: All ROI checks done in single pass through batch")
    print("(vs. calling _detection_in_roi() for each frame in a loop)")


if __name__ == '__main__':
    test_vectorized_roi_filtering()
