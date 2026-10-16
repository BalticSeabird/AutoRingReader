#!/usr/bin/env python3
"""Test ROI filtering logic."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from step1_ring_detection import _detection_in_roi


def test_detection_in_roi():
    """Test the _detection_in_roi function with various scenarios."""

    # Test 1: Detection center is clearly within ROI bbox
    bbox = {
        'x_norm': 0.5,
        'y_norm': 0.5,
        'w_norm': 0.1,
        'h_norm': 0.1,
    }
    roi_bbox = (100, 100, 900, 900)  # Large ROI
    frame_height, frame_width = 1000, 1000

    result = _detection_in_roi(bbox, frame_height, frame_width, roi_bbox)
    assert result == True, f"Test 1 failed: center at (500, 500) should be in ROI (100-900, 100-900), got {result}"
    print("✓ Test 1 passed: Detection center within ROI")

    # Test 2: Detection center is outside ROI bbox (to the right)
    bbox = {
        'x_norm': 0.95,
        'y_norm': 0.5,
        'w_norm': 0.1,
        'h_norm': 0.1,
    }
    roi_bbox = (100, 100, 500, 900)  # ROI on left side
    frame_height, frame_width = 1000, 1000

    result = _detection_in_roi(bbox, frame_height, frame_width, roi_bbox)
    assert result == False, f"Test 2 failed: center at (950, 500) should be outside ROI (100-500, 100-900), got {result}"
    print("✓ Test 2 passed: Detection center outside ROI (right)")

    # Test 3: Detection center is outside ROI bbox (above)
    bbox = {
        'x_norm': 0.5,
        'y_norm': 0.05,
        'w_norm': 0.1,
        'h_norm': 0.1,
    }
    roi_bbox = (100, 200, 900, 900)  # ROI below y=200
    frame_height, frame_width = 1000, 1000

    result = _detection_in_roi(bbox, frame_height, frame_width, roi_bbox)
    assert result == False, f"Test 3 failed: center at (500, 50) should be outside ROI (100-900, 200-900), got {result}"
    print("✓ Test 3 passed: Detection center outside ROI (above)")

    # Test 4: Edge case - detection exactly at ROI edge (should be inside)
    bbox = {
        'x_norm': 0.1,
        'y_norm': 0.2,
        'w_norm': 0.1,
        'h_norm': 0.1,
    }
    roi_bbox = (100, 200, 900, 900)  # ROI edge at x=100, y=200
    frame_height, frame_width = 1000, 1000

    result = _detection_in_roi(bbox, frame_height, frame_width, roi_bbox)
    assert result == True, f"Test 4 failed: center at (100, 200) should be inside ROI edge (100-900, 200-900), got {result}"
    print("✓ Test 4 passed: Detection at ROI edge")

    # Test 5: Different frame size
    bbox = {
        'x_norm': 0.5,
        'y_norm': 0.5,
        'w_norm': 0.1,
        'h_norm': 0.1,
    }
    roi_bbox = (480, 360, 1920, 1440)  # ROI on right side
    frame_height, frame_width = 1440, 1920

    result = _detection_in_roi(bbox, frame_height, frame_width, roi_bbox)
    assert result == True, f"Test 5 failed: center at (960, 720) should be in ROI (480-1920, 360-1440), got {result}"
    print("✓ Test 5 passed: Different frame size")

    # Test 6: Detection clearly outside ROI (left side)
    bbox = {
        'x_norm': 0.1,
        'y_norm': 0.5,
        'w_norm': 0.1,
        'h_norm': 0.1,
    }
    roi_bbox = (500, 100, 900, 900)  # ROI on right side
    frame_height, frame_width = 1000, 1000

    result = _detection_in_roi(bbox, frame_height, frame_width, roi_bbox)
    assert result == False, f"Test 6 failed: center at (100, 500) should be outside ROI (500-900, 100-900), got {result}"
    print("✓ Test 6 passed: Detection outside ROI (left)")

    print("\n✓ All ROI filtering tests passed!")


if __name__ == '__main__':
    test_detection_in_roi()
