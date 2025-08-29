#!/usr/bin/env python3
"""
Test script to verify the radar detection system is working properly.
This script tests the core components of the enhanced radar processing system.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from radar_target_detection import EnhancedRadarDetector
    from radar_visualization import EnhancedRadarVisualization
    print("✅ Successfully imported radar detection modules")
except ImportError as e:
    print(f"❌ Failed to import radar detection modules: {e}")
    sys.exit(1)

def create_test_radar_image():
    """Create a simple test radar image with some targets."""
    # Create a blank radar image (black background)
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Add some radar-like patterns
    # Draw concentric circles (typical radar display)
    center = (200, 200)
    for radius in range(50, 200, 30):
        cv2.circle(img, center, radius, (50, 50, 50), 1)

    # Draw radial lines
    for angle in range(0, 360, 30):
        x = int(center[0] + 180 * np.cos(np.radians(angle)))
        y = int(center[1] + 180 * np.sin(np.radians(angle)))
        cv2.line(img, center, (x, y), (30, 30, 30), 1)

    # Add some "targets" as bright spots
    targets = [
        (150, 150),  # Target 1
        (250, 180),  # Target 2
        (180, 280),  # Target 3
    ]

    for i, (x, y) in enumerate(targets):
        # Draw bright target
        cv2.circle(img, (x, y), 8, (255, 255, 255), -1)
        cv2.circle(img, (x, y), 12, (200, 200, 200), 2)

    return img

def test_radar_detection():
    """Test the radar detection system."""
    print("\n🧪 Testing Radar Detection System...")

    # Create test image
    test_image = create_test_radar_image()
    print("✅ Created test radar image")

    # Initialize detector
    try:
        detector = EnhancedRadarDetector()
        print("✅ Initialized EnhancedRadarDetector")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False

    # Test detection
    try:
        result = detector.detect_targets(test_image)
        targets = result.get('targets', [])
        print(f"✅ Detected {len(targets)} targets")

        if len(targets) > 0:
            print("📊 Target details:")
            for i, target in enumerate(targets):
                pixel_pos = target.get('pixel_position', (0, 0))
                print(f"   Target {i+1}: Position ({pixel_pos[0]:.1f}, {pixel_pos[1]:.1f}), "
                      f"Confidence: {target.get('confidence', 0):.2f}, "
                      f"Distance: {target.get('distance_meters', 0)} meters")

    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return False

    # Test visualization
    try:
        visualizer = EnhancedRadarVisualization()
        print("✅ Initialized EnhancedRadarVisualization")

        # Create numbered visualization
        numbered_img = visualizer.create_numbered_visualization(test_image, result)
        print("✅ Created numbered visualization")

        # Generate target details table
        details_html = visualizer.generate_target_details_table(result)
        print("✅ Generated target details table")

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

    return True

def main():
    """Main test function."""
    print("🚀 Starting Radar Detection System Test")
    print("=" * 50)

    # Check Python version
    print(f"🐍 Python version: {sys.version}")

    # Check if required packages are available
    required_packages = ['cv2', 'numpy', 'PIL']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is not available")

    # Test the radar detection system
    success = test_radar_detection()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The radar detection system is ready.")
        print("\n📝 Next steps:")
        print("   1. Run 'streamlit run app.py' to start the web interface")
        print("   2. Upload your 240 radar images for processing")
        print("   3. Use batch_radar_processor.py for automated batch processing")
    else:
        print("❌ Some tests failed. Please check the error messages above.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
