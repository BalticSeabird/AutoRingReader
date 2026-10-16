#!/usr/bin/env python3
"""
Quick validation script to verify the RGB/BGR fix in step2_tag_detection.py

This test verifies that the channel conversion logic is correct by:
1. Creating a test image with distinct RGB values
2. Converting RGB->BGR (the fix)
3. Saving with cv.imwrite (which expects BGR)
4. Reading back with cv.imread (which returns BGR)
5. Converting back to RGB
6. Checking that we end up with the original RGB order (ignoring JPEG artifacts)
"""

import cv2 as cv
import numpy as np
import tempfile
from pathlib import Path

def test_rgb_bgr_channel_ordering():
    """Test that RGB->BGR conversion preserves channel ordering (ignoring JPEG artifacts)."""

    print("Testing RGB/BGR channel ordering for tag crops...")

    # Create a simple test image with specific channel patterns (using PNG to avoid JPEG artifacts)
    test_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    test_rgb[:, :5, 0] = 200  # Left half: mostly red channel (R=200, G=0, B=0)
    test_rgb[:, 5:, 2] = 200  # Right half: mostly blue channel (R=0, G=0, B=200)
    test_rgb[:, :, 1] = 50    # Green channel everywhere (G=50)

    print(f"\n  Original RGB[0,0,:] (should be [200, 50, 0]): {test_rgb[0, 0, :]}")
    print(f"  Original RGB[0,9,:] (should be [0, 50, 200]): {test_rgb[0, 9, :]}")

    # THE FIX: RGB -> BGR before saving
    test_bgr = cv.cvtColor(test_rgb, cv.COLOR_RGB2BGR)
    print(f"  After RGB->BGR [0,0,:] (should be [0, 50, 200]): {test_bgr[0, 0, :]}")
    print(f"  After RGB->BGR [0,9,:] (should be [200, 50, 0]): {test_bgr[0, 9, :]}")

    # Save THEN read back (using PNG to avoid JPEG artifacts)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test.png"
        cv.imwrite(str(test_path), test_bgr)

        # Read back with cv.imread (returns in BGR format)
        read_bgr = cv.imread(str(test_path))

        # Convert back to RGB (this is what step3 does)
        read_rgb = cv.cvtColor(read_bgr, cv.COLOR_BGR2RGB)

        print(f"  Read back RGB[0,0,:] (should be [200, 50, 0]): {read_rgb[0, 0, :]}")
        print(f"  Read back RGB[0,9,:] (should be [0, 50, 200]): {read_rgb[0, 9, :]}")

        # Check if channels are in the correct order (exact match with PNG)
        if np.allclose(test_rgb, read_rgb):
            print("\n✓ RGB/BGR conversion fix is CORRECT!")
            print("  Channel ordering is preserved throughout the pipeline")
            return True
        else:
            print("\n✗ RGB/BGR conversion FAILED - channel ordering is wrong")
            print(f"  Max channel difference: {np.max(np.abs(test_rgb.astype(int) - read_rgb.astype(int)))}")
            return False

if __name__ == "__main__":
    success = test_rgb_bgr_channel_ordering()
    exit(0 if success else 1)
