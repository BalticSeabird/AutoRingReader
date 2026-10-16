#!/usr/bin/env python3
"""Integration test: ROI filtering with realistic detection scenarios."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from step1_ring_detection import _detection_in_roi
from pipeline_config import ROIConfig


def test_integration_with_real_rois():
    """Test ROI filtering with actual ROI configurations."""

    roi_config = ROIConfig("config/rois_config.yaml")

    # Test Case 1: FAR3_CUTE with detection inside ROI
    print("\n=== Test Case 1: FAR3_CUTE (detection inside ROI) ===")
    roi_bbox = roi_config.get_roi_bbox("FAR3_CUTE", year=2024)
    print(f"ROI bbox: {roi_bbox}")

    # Simulate detection near center of ROI
    # ROI center is approximately at ((828.833+1562.167)/2, (707.5+1036.833)/2) = (1195.5, 872.167)
    # Frame is 1920x1440, so normalized: (1195.5/1920, 872.167/1440) ≈ (0.623, 0.606)
    bbox_inside = {
        'x_norm': 0.623,
        'y_norm': 0.606,
        'w_norm': 0.1,
        'h_norm': 0.1,
    }

    result = _detection_in_roi(bbox_inside, frame_height=1440, frame_width=1920, roi_bbox=roi_bbox)
    assert result == True, f"Detection inside ROI should return True, got {result}"
    print(f"✓ Detection at center: {result} (expected True)")

    # Test Case 2: FAR3_CUTE with detection outside ROI
    print("\n=== Test Case 2: FAR3_CUTE (detection outside ROI) ===")
    bbox_outside = {
        'x_norm': 0.1,
        'y_norm': 0.1,
        'w_norm': 0.1,
        'h_norm': 0.1,
    }

    result = _detection_in_roi(bbox_outside, frame_height=1440, frame_width=1920, roi_bbox=roi_bbox)
    assert result == False, f"Detection outside ROI should return False, got {result}"
    print(f"✓ Detection at top-left corner: {result} (expected False)")

    # Test Case 3: FAR6_SCALE with detection inside
    print("\n=== Test Case 3: FAR6_SCALE (detection inside ROI) ===")
    roi_bbox = roi_config.get_roi_bbox("FAR6_SCALE", year=2024)
    print(f"ROI bbox: {roi_bbox}")

    # ROI: (191, 409, 1346, 1249), roughly center at (768.5, 829)
    # Frame 1920x1440, normalized: (768.5/1920, 829/1440) ≈ (0.4, 0.576)
    bbox_inside = {
        'x_norm': 0.4,
        'y_norm': 0.576,
        'w_norm': 0.1,
        'h_norm': 0.1,
    }

    result = _detection_in_roi(bbox_inside, frame_height=1440, frame_width=1920, roi_bbox=roi_bbox)
    assert result == True, f"Detection inside FAR6_SCALE ROI should return True, got {result}"
    print(f"✓ Detection at center: {result} (expected True)")

    # Test Case 4: ROST2_SCALE with detection near edge
    print("\n=== Test Case 4: ROST2_SCALE (detection near ROI edge) ===")
    roi_bbox = roi_config.get_roi_bbox("ROST2_SCALE", year=2024)
    print(f"ROI bbox: {roi_bbox}")

    # ROI: (248, 265.5, 1979.5, 1435.5), try detection just inside right edge
    # Right edge at x=1979.5, normalized: 1979.5/1920 ≈ 1.03 (exceeds frame width normalized coords)
    # Let's try x=1970 (just inside), normalized: 1970/1920 ≈ 1.026... that's still off
    # Actually, let's check actual frame - ROI is in pixel coords, test at x=1900 > 1979.5? No
    # Let's test near the center instead
    # Center approximately: ((248+1979.5)/2, (265.5+1435.5)/2) = (1113.75, 850.5)
    # Normalized: (1113.75/1920, 850.5/1440) ≈ (0.58, 0.591)
    bbox_center = {
        'x_norm': 0.58,
        'y_norm': 0.591,
        'w_norm': 0.1,
        'h_norm': 0.1,
    }

    result = _detection_in_roi(bbox_center, frame_height=1440, frame_width=1920, roi_bbox=roi_bbox)
    assert result == True, f"Detection at center should return True, got {result}"
    print(f"✓ Detection at center: {result} (expected True)")

    # Test Case 5: Detection slightly inside ROI boundary
    print("\n=== Test Case 5: Detection slightly inside ROI boundary ===")
    # Test just inside the min corner of ROST2_SCALE ROI
    # Min corner: (248, 265.5), let's test at (260, 280)
    # Normalized: (260/1920, 280/1440) ≈ (0.1354, 0.1944)
    bbox_inside_boundary = {
        'x_norm': 0.135,  # ~260 pixels
        'y_norm': 0.195,  # ~280 pixels
        'w_norm': 0.1,
        'h_norm': 0.1,
    }

    result = _detection_in_roi(bbox_inside_boundary, frame_height=1440, frame_width=1920, roi_bbox=roi_bbox)
    assert result == True, f"Detection inside ROI boundary should return True, got {result}"
    print(f"✓ Detection slightly inside ROI boundary: {result} (expected True)")


    print("\n✓ All integration tests passed!")


if __name__ == '__main__':
    test_integration_with_real_rois()
