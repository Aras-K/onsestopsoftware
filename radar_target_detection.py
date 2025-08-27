import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import logging
from enum import Enum
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
    size_estimate: float  # Approximate size in NM
    confidence: float
    echo_strength: float
    is_moving: bool = False
    velocity: Optional[Tuple[float, float]] = None  # (speed_kn, course_deg)

class RadarTargetDetector:
    """Detects and classifies targets in radar images."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_targets(self, image_path: str, radar_type: str, 
                    range_setting: float) -> List[RadarTarget]:
        """Optimized target detection with performance improvements."""
        try:
            # Load and immediately resize for performance
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # CRITICAL OPTIMIZATION: Work with smaller image
            original_height, original_width = image.shape[:2]
            max_dimension = 800  # Reduced from 1200
            
            if original_width > max_dimension or original_height > max_dimension:
                scale = max_dimension / max(original_width, original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                scale_factor = original_width / new_width
            else:
                scale_factor = 1.0
            
            # Simplified radar detection - skip complex processing
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            radius = min(w, h) // 2 - 20
            
            # FAST echo detection - simplified
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Simple threshold instead of complex processing
            threshold_value = 100 if "FURUNO" in radar_type else 120
            _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Small kernel for speed
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours with approximation for speed
            contours, _ = cv2.findContours(
                cleaned, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE  # Simple approximation is faster
            )
            
            # Limit number of targets processed
            MAX_TARGETS = 50
            targets = []
            
            # Sort by area and process only largest contours
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5:  # Minimum area threshold
                    valid_contours.append((area, contour))
            
            # Sort by area and take only top MAX_TARGETS
            valid_contours.sort(key=lambda x: x[0], reverse=True)
            valid_contours = valid_contours[:MAX_TARGETS]
            
            for idx, (area, contour) in enumerate(valid_contours):
                # Fast centroid calculation
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"]) 
                cy = int(M["m01"] / M["m00"])
                
                # Scale back to original coordinates if needed
                if scale_factor != 1.0:
                    cx = int(cx * scale_factor)
                    cy = int(cy * scale_factor)
                    area = area * (scale_factor ** 2)
                
                # Simplified position calculation
                dx = cx - center[0]
                dy = center[1] - cy
                pixel_distance = np.sqrt(dx**2 + dy**2)
                range_nm = (pixel_distance / radius) * range_setting if radius > 0 else 0
                
                # Skip if too far
                if range_nm > range_setting * 1.2:
                    continue
                
                bearing = np.degrees(np.arctan2(dx, dy))
                if bearing < 0:
                    bearing += 360
                
                # Simplified classification
                if area > 500:
                    target_type = TargetType.LANDMASS
                elif area > 50:
                    target_type = TargetType.VESSEL
                elif area > 10:
                    target_type = TargetType.OBSTACLE
                else:
                    target_type = TargetType.UNKNOWN
                
                # Simple confidence based on size
                confidence = min(0.5 + (area / 1000), 0.95)
                
                targets.append(RadarTarget(
                    target_id=idx,
                    target_type=target_type,
                    position=(range_nm, bearing),
                    pixel_position=(cx, cy),
                    size_estimate=np.sqrt(area) / radius * range_setting if radius > 0 else 0,
                    confidence=confidence,
                    echo_strength=confidence,
                    is_moving=False  # Skip movement detection for speed
                ))
            
            return targets[:MAX_TARGETS]  # Limit final results
            
        except Exception as e:
            logger.error(f"Error in optimized target detection: {e}")
            return []
    def _find_radar_display_area(self, image: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """Find the circular radar display area."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Hough Circle Transform to find radar display
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=100,
            maxRadius=500
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Get the largest circle (likely the radar display)
            largest_circle = max(circles[0], key=lambda x: x[2])
            return (int(largest_circle[0]), int(largest_circle[1])), int(largest_circle[2])
        
        # Fallback: assume center of image
        h, w = image.shape[:2]
        return (w//2, h//2), min(w, h)//2 - 50
    
    def _detect_echo_returns(self, radar_region: np.ndarray, 
                            radar_type: str) -> np.ndarray:
        """Detect radar echo returns (bright spots)."""
        # Convert to grayscale
        if len(radar_region.shape) == 3:
            gray = cv2.cvtColor(radar_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = radar_region
        
        # Apply adaptive thresholding to find bright echoes
        # Adjust parameters based on radar type
        if "FURUNO" in radar_type:
            # Furuno typically has green display
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        else:
            # JRC and others
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Remove noise
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _classify_targets(self, echo_mask: np.ndarray, center: Tuple[int, int],
                         radius: int, range_setting: float) -> List[RadarTarget]:
        """Classify detected echoes into target types."""
        targets = []
        
        # Find contours of echo returns
        contours, _ = cv2.findContours(
            echo_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for idx, contour in enumerate(contours):
            # Get contour properties
            area = cv2.contourArea(contour)
            if area < 10:  # Skip very small echoes (noise)
                continue
            
            # Get center of mass
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Convert pixel to radar coordinates
            range_nm, bearing = self._pixel_to_radar_coords(
                (cx, cy), center, radius, range_setting
            )
            
            # Classify based on characteristics
            target_type = self._classify_by_characteristics(
                contour, area, echo_mask.shape
            )
            
            # Calculate confidence based on echo strength
            confidence = self._calculate_confidence(contour, echo_mask)
            
            # Estimate size
            size_estimate = self._estimate_target_size(area, radius, range_setting)
            
            target = RadarTarget(
                target_id=idx,
                target_type=target_type,
                position=(range_nm, bearing),
                pixel_position=(cx, cy),
                size_estimate=size_estimate,
                confidence=confidence,
                echo_strength=confidence,
                is_moving=self._check_if_moving(contour)
            )
            
            targets.append(target)
        
        return targets
    
    def _classify_by_characteristics(self, contour: np.ndarray, 
                                    area: float, 
                                    image_shape: Tuple) -> TargetType:
        """Classify target based on echo characteristics."""
        # Get contour properties
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1
        
        # Landmass detection (large, irregular shape)
        if area > 1000:
            if circularity < 0.3:  # Irregular shape
                return TargetType.LANDMASS
        
        # Vessel detection (small to medium, relatively circular)
        if 50 < area < 500:
            if circularity > 0.5 and 0.7 < aspect_ratio < 1.3:
                return TargetType.VESSEL
        
        # Obstacle detection (medium size, various shapes)
        if 20 < area < 200:
            return TargetType.OBSTACLE
        
        return TargetType.UNKNOWN
    
    def _pixel_to_radar_coords(self, pixel: Tuple[int, int], 
                              center: Tuple[int, int],
                              radius: int, 
                              range_setting: float) -> Tuple[float, float]:
        """Convert pixel coordinates to radar range and bearing."""
        dx = pixel[0] - center[0]
        dy = center[1] - pixel[1]  # Invert Y axis
        
        # Calculate range
        pixel_distance = np.sqrt(dx**2 + dy**2)
        range_nm = (pixel_distance / radius) * range_setting
        
        # Calculate bearing (0-360 degrees, 0 = North)
        bearing = np.degrees(np.arctan2(dx, dy))
        if bearing < 0:
            bearing += 360
        
        return range_nm, bearing
    
    def _calculate_confidence(self, contour: np.ndarray, 
                            echo_mask: np.ndarray) -> float:
        """Calculate confidence based on echo characteristics."""
        # Create mask for this contour
        mask = np.zeros(echo_mask.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Calculate mean intensity in the contour area
        mean_intensity = cv2.mean(echo_mask, mask=mask)[0] / 255.0
        
        # Consider size and shape
        area = cv2.contourArea(contour)
        confidence = mean_intensity * min(1.0, area / 100)
        
        return min(confidence, 1.0)
    
    def _estimate_target_size(self, area: float, radius: int, 
                            range_setting: float) -> float:
        """Estimate target size in nautical miles."""
        # Convert pixel area to radar scale
        pixels_per_nm = radius / range_setting
        area_nm = area / (pixels_per_nm ** 2)
        
        # Approximate as circular target
        equivalent_diameter = 2 * np.sqrt(area_nm / np.pi)
        
        return equivalent_diameter
    
    def _check_if_moving(self, contour: np.ndarray) -> bool:
        """Check if target appears to be moving (has wake or trail)."""
        # Simplified check - look for elongation
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            return aspect_ratio > 2.0  # Elongated shape suggests movement
        
        return False
    
    def _filter_targets(self, targets: List[RadarTarget]) -> List[RadarTarget]:
        """Filter out likely false positives."""
        filtered = []
        
        for target in targets:
            # Remove very low confidence targets
            if target.confidence < 0.2:
                continue
            
            # Remove targets outside reasonable range
            if target.position[0] > 100:  # Beyond 100 NM is unlikely
                continue
            
            filtered.append(target)
        
        return filtered