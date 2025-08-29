# radar_target_detection_yolo.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import logging
from enum import Enum
from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)

class TargetType(Enum):
    VESSEL = "vessel"
    LANDMASS = "landmass"
    OBSTACLE = "obstacle"
    UNKNOWN = "unknown"

@dataclass
class RadarTarget:
    """Represents a detected target on radar."""
    target_id: int
    target_type: TargetType
    position: Tuple[float, float]  # (range_nm, bearing_deg)
    pixel_position: Tuple[int, int]  # (x, y) in image
    size_estimate: float
    confidence: float
    echo_strength: float
    is_moving: bool = False
    velocity: Optional[Tuple[float, float]] = None

class YOLORadarDetector:
    """YOLO-based radar target detection."""
    
    def __init__(self, model_path: str = None):
        """Initialize YOLO detector."""
        self.logger = logging.getLogger(__name__)
        
        if model_path:
            # Use custom trained model
            self.model = YOLO(model_path)
        else:
            # Use pretrained model and adapt for radar
            self.model = YOLO('yolov8n.pt')  # Nano model for speed
            
        # Map YOLO classes to radar targets
        self.class_mapping = {
            'boat': TargetType.VESSEL,
            'ship': TargetType.VESSEL,
            'person': TargetType.UNKNOWN,  # Could be small vessel
            'car': TargetType.OBSTACLE,  # If near shore
            'truck': TargetType.OBSTACLE,
            # Add more mappings as needed
        }
    
    def detect_targets(self, image_path: str, radar_type: str, 
                      range_setting: float) -> List[RadarTarget]:
        """Detect targets using YOLO."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # Preprocess for radar (enhance contrast)
            processed = self._preprocess_radar_image(image, radar_type)
            
            # Run YOLO detection
            results = self.model(processed, conf=0.25, iou=0.45)
            
            # Get image dimensions for coordinate conversion
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            radius = min(w, h) // 2 - 50
            
            targets = []
            for idx, result in enumerate(results):
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # Get detection info
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    # Get class name
                    class_name = self.model.names.get(cls, 'unknown')
                    
                    # Map to radar target type
                    target_type = self._classify_target(
                        class_name, 
                        (x2-x1) * (y2-y1),  # area
                        image.shape
                    )
                    
                    # Calculate center
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # Convert to radar coordinates
                    range_nm, bearing = self._pixel_to_radar_coords(
                        (cx, cy), center, radius, range_setting
                    )
                    
                    # Skip if outside radar range
                    if range_nm > range_setting * 1.2:
                        continue
                    
                    # Create target
                    targets.append(RadarTarget(
                        target_id=idx,
                        target_type=target_type,
                        position=(range_nm, bearing),
                        pixel_position=(cx, cy),
                        size_estimate=self._estimate_size(x2-x1, y2-y1, radius, range_setting),
                        confidence=conf,
                        echo_strength=conf,
                        is_moving=self._check_movement(box)
                    ))
            
            # If no YOLO detections, fallback to brightness-based detection
            if len(targets) == 0:
                targets = self._fallback_detection(processed, center, radius, range_setting)
            
            logger.info(f"YOLO detected {len(targets)} targets")
            return targets
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            # Fallback to simple detection
            return self._simple_detection(image_path, radar_type, range_setting)
    
    def _preprocess_radar_image(self, image: np.ndarray, radar_type: str) -> np.ndarray:
        """Preprocess radar image for better detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR for YOLO
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced_bgr
    
    def _simple_detection(self, image_path: str, radar_type: str, 
                         range_setting: float) -> List[RadarTarget]:
        """Simple brightness-based detection as fallback."""
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find bright spots
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 50
        
        targets = []
        for idx, contour in enumerate(contours[:50]):  # Limit to 50
            area = cv2.contourArea(contour)
            if area < 10:
                continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Convert to radar coordinates
            range_nm, bearing = self._pixel_to_radar_coords(
                (cx, cy), center, radius, range_setting
            )
            
            if range_nm > range_setting * 1.2:
                continue
            
            # Simple classification by size
            if area > 1000:
                target_type = TargetType.LANDMASS
            elif area > 100:
                target_type = TargetType.VESSEL
            else:
                target_type = TargetType.OBSTACLE
            
            targets.append(RadarTarget(
                target_id=idx,
                target_type=target_type,
                position=(range_nm, bearing),
                pixel_position=(cx, cy),
                size_estimate=np.sqrt(area) / radius * range_setting if radius > 0 else 0,
                confidence=0.5,
                echo_strength=0.5,
                is_moving=False
            ))
        
        return targets
    
    def _classify_target(self, class_name: str, area: float, 
                        image_shape: Tuple) -> TargetType:
        """Classify target based on YOLO class and size."""
        # Check if we have a direct mapping
        if class_name in ['boat', 'ship']:
            return TargetType.VESSEL
        
        # Use area for classification if no direct mapping
        if area > image_shape[0] * image_shape[1] * 0.05:  # > 5% of image
            return TargetType.LANDMASS
        elif area > 500:
            return TargetType.VESSEL
        elif area > 100:
            return TargetType.OBSTACLE
        
        return TargetType.UNKNOWN
    
    def _pixel_to_radar_coords(self, pixel: Tuple[int, int], 
                              center: Tuple[int, int],
                              radius: int, 
                              range_setting: float) -> Tuple[float, float]:
        """Convert pixel coordinates to radar range and bearing."""
        dx = pixel[0] - center[0]
        dy = center[1] - pixel[1]
        
        pixel_distance = np.sqrt(dx**2 + dy**2)
        range_nm = (pixel_distance / radius) * range_setting if radius > 0 else 0
        
        bearing = np.degrees(np.arctan2(dx, dy))
        if bearing < 0:
            bearing += 360
        
        return range_nm, bearing
    
    def _estimate_size(self, width: float, height: float, 
                      radius: int, range_setting: float) -> float:
        """Estimate target size in nautical miles."""
        avg_size_pixels = (width + height) / 2
        size_nm = (avg_size_pixels / radius) * range_setting if radius > 0 else 0
        return size_nm
    
    def _check_movement(self, box) -> bool:
        """Check if target appears to be moving."""
        # For now, return False. You could implement motion detection
        # by comparing with previous frames
        return False
    
    def _fallback_detection(self, image: np.ndarray, center: Tuple[int, int],
                          radius: int, range_setting: float) -> List[RadarTarget]:
        """Fallback detection using traditional CV methods."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        targets = []
        for idx, contour in enumerate(contours[:30]):
            area = cv2.contourArea(contour)
            if area < 50:
                continue
            
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            range_nm, bearing = self._pixel_to_radar_coords(
                (cx, cy), center, radius, range_setting
            )
            
            targets.append(RadarTarget(
                target_id=idx,
                target_type=TargetType.UNKNOWN,
                position=(range_nm, bearing),
                pixel_position=(cx, cy),
                size_estimate=0.1,
                confidence=0.3,
                echo_strength=0.3,
                is_moving=False
            ))
        
        return targets