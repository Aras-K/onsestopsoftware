# radar_visualization.py
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from radar_target_detection import TargetType

logger = logging.getLogger(__name__)

class RadarVisualization:
    """Visualize detected targets on radar images."""
    
    @staticmethod
    def visualize_targets(image_path: str, targets_metadata: Dict) -> np.ndarray:
        """
        Draw detected targets on radar image.
        
        Args:
            image_path: Path to original radar image
            targets_metadata: Dictionary containing detected targets from metadata
        
        Returns:
            Image with visualized targets
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Get targets list
        targets = targets_metadata.get('targets', [])
        
        # Find radar center and radius (simplified - you may need to adjust)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 50
        
        # Draw each target
        for target in targets:
            # Convert radar coordinates to pixel coordinates
            range_nm = target['range_nm']
            bearing_deg = target['bearing_deg']
            
            # Calculate pixel position
            # Convert bearing to radians (0° = North = up)
            bearing_rad = np.radians(bearing_deg - 90)  # Adjust for screen coordinates
            
            # Scale range to pixels (assuming full radius = range setting)
            # You might need to get actual range setting
            max_range = targets_metadata.get('range_setting', 12.0)
            pixel_distance = (range_nm / max_range) * radius
            
            x = int(center[0] + pixel_distance * np.cos(bearing_rad))
            y = int(center[1] + pixel_distance * np.sin(bearing_rad))
            
            # Choose color based on target type
            target_type = target['type']
            if target_type == 'vessel':
                color = (0, 255, 0)  # Green
                symbol = 'V'
            elif target_type == 'landmass':
                color = (255, 165, 0)  # Orange
                symbol = 'L'
            elif target_type == 'obstacle':
                color = (255, 0, 0)  # Red
                symbol = 'O'
            else:
                color = (128, 128, 128)  # Gray
                symbol = '?'
            
            # Draw target marker
            cv2.circle(image, (x, y), 8, color, 2)
            
            # Add label
            label = f"{symbol} {target['confidence']:.0%}"
            cv2.putText(image, label, (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw range/bearing info
            info = f"{range_nm:.1f}NM/{bearing_deg:.0f}°"
            cv2.putText(image, info, (x + 10, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw movement indicator if moving
            if target.get('is_moving', False):
                # Draw arrow indicating movement
                cv2.arrowedLine(image, (x, y), 
                              (x + 20, y - 20), color, 2)
        
        # Add summary box
        summary = targets_metadata.get('summary', {})
        if summary:
            # Draw summary in corner
            y_offset = 30
            cv2.putText(image, "DETECTED TARGETS:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            cv2.putText(image, f"Vessels: {summary.get('vessels', 0)}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            
            cv2.putText(image, f"Landmasses: {summary.get('landmasses', 0)}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            y_offset += 20
            
            cv2.putText(image, f"Obstacles: {summary.get('obstacles', 0)}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return image
    
    @staticmethod
    def save_visualization(image_path: str, targets_metadata: Dict, 
                         output_path: str = None) -> str:
        """
        Save visualization to file.
        
        Args:
            image_path: Original image path
            targets_metadata: Detected targets metadata
            output_path: Where to save (optional)
        
        Returns:
            Path to saved visualization
        """
        import os
        
        # Visualize
        viz_image = RadarVisualization.visualize_targets(image_path, targets_metadata)
        
        if viz_image is None:
            return None
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_targets.png"
        
        # Save
        cv2.imwrite(output_path, viz_image)
        logger.info(f"Saved visualization to: {output_path}")
        
        return output_path