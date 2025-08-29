# radar_visualization.py
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from radar_target_detection import TargetType
from datetime import datetime
logger = logging.getLogger(__name__)

class RadarVisualization:
    """Visualize detected targets on radar images."""
    
    @staticmethod
    def visualize_targets(image_path: str, targets_metadata: Dict) -> np.ndarray:
        """Draw detected targets on radar image with professional annotations."""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Create a copy for annotation
        annotated = image.copy()
        
        # Add semi-transparent overlay for better visibility
        overlay = annotated.copy()
        
        targets = targets_metadata.get('targets', [])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 50
        
        # Draw targets
        for target in targets:
            range_nm = target['range_nm']
            bearing_deg = target['bearing_deg']
            
            # Calculate position
            bearing_rad = np.radians(bearing_deg - 90)
            max_range = 12.0  # Default max range
            pixel_distance = (range_nm / max_range) * radius
            
            x = int(center[0] + pixel_distance * np.cos(bearing_rad))
            y = int(center[1] + pixel_distance * np.sin(bearing_rad))
            
            # Color and style based on type
            target_type = target['type']
            if target_type == 'vessel':
                color = (0, 255, 0)
                thickness = 2
                radius_circle = 12
            elif target_type == 'landmass':
                color = (0, 165, 255)
                thickness = 3
                radius_circle = 15
            elif target_type == 'obstacle':
                color = (0, 0, 255)
                thickness = 2
                radius_circle = 10
            else:
                color = (128, 128, 128)
                thickness = 1
                radius_circle = 8
            
            # Draw on overlay
            cv2.circle(overlay, (x, y), radius_circle, color, thickness)
            
            # Add crosshair for vessels
            if target_type == 'vessel':
                cv2.line(overlay, (x-15, y), (x+15, y), color, 1)
                cv2.line(overlay, (x, y-15), (x, y+15), color, 1)
            
            # Add text label with background
            label = f"{range_nm:.1f}NM"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness_text = 1
            
            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness_text)
            
            # Draw background rectangle
            cv2.rectangle(overlay, 
                        (x + 10, y - text_height - 2),
                        (x + 10 + text_width + 4, y + 2),
                        (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(overlay, label, (x + 12, y),
                    font, font_scale, color, thickness_text)
        
        # Blend overlay with original
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        # Add legend
        legend_y = 30
        legend_x = w - 200
        
        # Legend background
        cv2.rectangle(annotated, (legend_x - 10, 10), (w - 10, 130), (0, 0, 0), -1)
        cv2.rectangle(annotated, (legend_x - 10, 10), (w - 10, 130), (255, 255, 255), 1)
        
        # Legend title
        cv2.putText(annotated, "TARGETS", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Legend items
        items = [
            (targets_metadata.get('vessels', 0), "Vessels", (0, 255, 0)),
            (targets_metadata.get('landmasses', 0), "Landmasses", (0, 165, 255)),
            (targets_metadata.get('obstacles', 0), "Obstacles", (0, 0, 255))
        ]
        
        for i, (count, name, color) in enumerate(items):
            y_pos = legend_y + 30 + (i * 25)
            cv2.circle(annotated, (legend_x + 10, y_pos), 5, color, -1)
            cv2.putText(annotated, f"{name}: {count}", (legend_x + 25, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, timestamp, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated
    
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