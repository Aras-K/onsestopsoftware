# batch_radar_processor.py - Professional Batch Processing for 240 Sequential Radar Images

import os
import sys
import cv2
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time
import concurrent.futures
from tqdm import tqdm
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our enhanced detection system
from radar_target_detection import EnhancedRadarDetector
from radar_visualization import EnhancedRadarVisualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchRadarProcessor:
    """Professional batch processor for 240 sequential radar images."""

    def __init__(self, input_dir: str, output_dir: str = None,
                 max_workers: int = 4, confidence_threshold: float = 0.3):
        """
        Initialize batch processor.

        Args:
            input_dir: Directory containing radar images
            output_dir: Directory for output (optional)
            max_workers: Maximum parallel workers
            confidence_threshold: Detection confidence threshold
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "processed"
        self.max_workers = max_workers
        self.confidence_threshold = confidence_threshold

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

        # Initialize detector
        self.detector = EnhancedRadarDetector(confidence_threshold=confidence_threshold)

        logger.info(f"Batch processor initialized for {self.input_dir}")

    def discover_images(self) -> List[Path]:
        """Discover all radar images in input directory."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

        images = []
        for ext in image_extensions:
            images.extend(self.input_dir.glob(f"*{ext}"))
            images.extend(self.input_dir.glob(f"*{ext.upper()}"))

        # Sort by filename (assuming sequential naming)
        images.sort(key=lambda x: x.name)

        logger.info(f"Discovered {len(images)} radar images")
        return images

    def process_single_image(self, image_path: Path) -> Dict:
        """Process a single radar image."""
        try:
            start_time = time.time()

            # Detect targets
            result = self.detector.detect_targets(
                str(image_path),
                radar_type="marine",
                range_setting=12.0
            )

            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['image_path'] = str(image_path)
            result['filename'] = image_path.name

            return result

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'error': str(e),
                'image_path': str(image_path),
                'filename': image_path.name,
                'total': 0,
                'targets': []
            }

    def create_visualization(self, image_path: Path, targets_metadata: Dict) -> Optional[str]:
        """Create enhanced visualization for an image."""
        try:
            output_path = self.output_dir / "visualizations" / f"{image_path.stem}_targets.png"

            saved_path = EnhancedRadarVisualization.save_visualization(
                str(image_path),
                targets_metadata,
                str(output_path)
            )

            return saved_path

        except Exception as e:
            logger.error(f"Error creating visualization for {image_path}: {e}")
            return None

    def process_batch(self, images: List[Path], batch_size: int = 10) -> Dict:
        """Process images in batches with progress tracking."""
        total_images = len(images)
        all_results = []

        logger.info(f"Starting batch processing of {total_images} images")

        with tqdm(total=total_images, desc="Processing Images") as pbar:
            for i in range(0, total_images, batch_size):
                batch = images[i:i + batch_size]

                # Process batch (with parallel processing for larger batches)
                if len(batch) > 1 and self.max_workers > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        batch_results = list(executor.map(self.process_single_image, batch))
                else:
                    batch_results = [self.process_single_image(img) for img in batch]

                # Create visualizations for successful detections
                for img_path, result in zip(batch, batch_results):
                    if result.get('total', 0) > 0:
                        viz_path = self.create_visualization(img_path, result)
                        result['visualization_path'] = viz_path

                all_results.extend(batch_results)
                pbar.update(len(batch))

        return {
            'total_processed': len(all_results),
            'results': all_results,
            'processing_timestamp': datetime.now().isoformat()
        }

    def generate_summary_report(self, results: Dict) -> Dict:
        """Generate comprehensive summary report."""
        all_results = results['results']

        # Filter successful results
        successful_results = [r for r in all_results if 'error' not in r]

        if not successful_results:
            return {'error': 'No successful processing results found'}

        # Calculate statistics
        total_targets = sum(r.get('total', 0) for r in successful_results)
        total_vessels = sum(r.get('vessels', 0) for r in successful_results)
        total_landmasses = sum(r.get('landmasses', 0) for r in successful_results)
        total_obstacles = sum(r.get('obstacles', 0) for r in successful_results)

        # Distance statistics
        all_distances = []
        all_targets = []
        for result in successful_results:
            for target in result.get('targets', []):
                all_distances.append(target['range_nm'])
                all_targets.append(target)

        # Risk assessment
        risk_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'SAFE': 0}
        for target in all_targets:
            risk = target.get('risk_level', 'LOW')
            risk_counts[risk] += 1

        summary = {
            'processing_summary': {
                'total_images_processed': len(successful_results),
                'total_images_with_errors': len(all_results) - len(successful_results),
                'total_targets_detected': total_targets,
                'average_targets_per_image': total_targets / len(successful_results) if successful_results else 0
            },
            'target_breakdown': {
                'vessels': total_vessels,
                'landmasses': total_landmasses,
                'obstacles': total_obstacles
            },
            'distance_statistics': {
                'average_distance_nm': np.mean(all_distances) if all_distances else 0,
                'min_distance_nm': min(all_distances) if all_distances else 0,
                'max_distance_nm': max(all_distances) if all_distances else 0,
                'median_distance_nm': np.median(all_distances) if all_distances else 0
            },
            'risk_assessment': risk_counts,
            'processing_timestamp': results['processing_timestamp']
        }

        return summary

    def save_results(self, results: Dict, summary: Dict):
        """Save all results and summary to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.output_dir / "data" / f"batch_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary
        summary_file = self.output_dir / "data" / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save CSV summary
        csv_file = self.output_dir / "data" / f"targets_summary_{timestamp}.csv"
        self._save_targets_csv(results, csv_file)

        logger.info(f"Results saved to {self.output_dir}/data/")

    def _save_targets_csv(self, results: Dict, csv_path: Path):
        """Save targets data to CSV format."""
        all_targets = []

        for result in results['results']:
            if 'error' in result:
                continue

            image_name = result['filename']
            for target in result.get('targets', []):
                target_data = {
                    'image': image_name,
                    'target_id': target['id'],
                    'target_type': target['type'],
                    'range_nm': target['range_nm'],
                    'distance_meters': target['distance_meters'],
                    'bearing_deg': target['bearing_deg'],
                    'confidence': target['confidence'],
                    'echo_strength': target['echo_strength'],
                    'risk_level': target['risk_level'],
                    'pixel_x': target['pixel_position'][0],
                    'pixel_y': target['pixel_position'][1]
                }
                all_targets.append(target_data)

        if all_targets:
            df = pd.DataFrame(all_targets)
            df.to_csv(csv_path, index=False)

    def run_complete_processing(self) -> Dict:
        """Run complete batch processing pipeline."""
        logger.info("Starting complete batch processing pipeline")

        # Discover images
        images = self.discover_images()

        if not images:
            raise ValueError(f"No radar images found in {self.input_dir}")

        # Process all images
        results = self.process_batch(images)

        # Generate summary
        summary = self.generate_summary_report(results)

        # Save results
        self.save_results(results, summary)

        logger.info("Batch processing completed successfully")
        return summary

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch process 240 radar images for target detection")
    parser.add_argument("input_dir", help="Directory containing radar images")
    parser.add_argument("--output_dir", help="Output directory (optional)")
    parser.add_argument("--workers", type=int, default=4, help="Maximum parallel workers")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")

    args = parser.parse_args()

    # Initialize processor
    processor = BatchRadarProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.workers,
        confidence_threshold=args.confidence
    )

    # Run processing
    try:
        summary = processor.run_complete_processing()

        print("\n🎯 Batch Processing Complete!")
        print(f"📊 Images Processed: {summary['processing_summary']['total_images_processed']}")
        print(f"🎯 Total Targets: {summary['processing_summary']['total_targets_detected']}")
        print(f"🚢 Vessels: {summary['target_breakdown']['vessels']}")
        print(f"🏝️ Landmasses: {summary['target_breakdown']['landmasses']}")
        print(f"⚠️ Obstacles: {summary['target_breakdown']['obstacles']}")
        print(f"📏 Avg Distance: {summary['distance_statistics']['average_distance_nm']:.1f} NM")
        print(f"🔥 Closest Target: {summary['distance_statistics']['min_distance_nm']:.1f} NM")
        print(f"📐 Farthest Target: {summary['distance_statistics']['max_distance_nm']:.1f} NM")

        print(f"\n📁 Results saved to: {processor.output_dir}/data/")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
