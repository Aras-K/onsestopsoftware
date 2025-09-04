# radar_target_detection.py - Enhanced Professional Target Detection System
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import logging
from enum import Enum
import math

logger = logging.getLogger(__name__)

class TargetType(Enum):
    VESSEL = "vessel"
    LANDMASS = "land"
    OWNVESSEL = "ownvessel"
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
    distance_meters: float  # Calculated distance in meters
    is_moving: bool = False
    velocity: Optional[Tuple[float, float]] = None
    risk_level: str = "LOW"

class EnhancedRadarDetector:
    """Enhanced professional radar target detection with improved reliability."""

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.3):
        """Initialize enhanced detector."""
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold

        # Try to load YOLO model, fallback to traditional methods if not available
        try:
            from ultralytics import YOLO
            if model_path:
                self.model = YOLO("/Users/aras.koplanto/Documents/OneStop/onsestopsoftware/best.pt")
            else:
                self.model = YOLO('yolov8n.pt')
            self.yolo_available = True
        except ImportError:
            self.logger.warning("YOLO not available. Using traditional detection.")
            self.model = None
            self.yolo_available = False

        # Enhanced class mapping for better vessel detection
        self.class_mapping = {
            'boat': TargetType.VESSEL,
            'ship': TargetType.VESSEL,
            'vessel': TargetType.VESSEL,
            'yacht': TargetType.VESSEL,
            'ferry': TargetType.VESSEL,
            'cargo': TargetType.VESSEL,
            'tanker': TargetType.VESSEL,
            'car': TargetType.OBSTACLE,
            'truck': TargetType.OBSTACLE,
            'bus': TargetType.OBSTACLE,
            'motorcycle': TargetType.OBSTACLE,
            'bicycle': TargetType.OBSTACLE,
            'person': TargetType.OBSTACLE,
            'building': TargetType.LANDMASS,
            'house': TargetType.LANDMASS,
            'skyscraper': TargetType.LANDMASS,
            'bridge': TargetType.LANDMASS,
        }

    def detect_targets(self, image_input, radar_type: str = "marine",
                      range_setting: float = 12.0) -> Dict:
        """Enhanced target detection with multiple methods."""
        try:
            # Handle both file paths and numpy arrays
            if isinstance(image_input, str):
                # Load image from file path
                image = cv2.imread(image_input)
                if image is None:
                    return self._create_empty_result()
                image_path = image_input
            elif isinstance(image_input, np.ndarray):
                # Use numpy array directly
                image = image_input.copy()
                image_path = "numpy_array_input"
            else:
                raise ValueError("image_input must be either a file path (str) or numpy array")

            # Enhanced preprocessing
            processed = self._enhanced_preprocess(image, radar_type)

            targets = []

            # Method 1: YOLO detection (if available)
            if self.yolo_available:
                yolo_targets = self._yolo_detection(processed, image, range_setting, radar_type)
                targets.extend(yolo_targets)

            # Method 2: Enhanced brightness-based detection
            brightness_targets = self._enhanced_brightness_detection(
                processed, image, range_setting, radar_type
            )
            targets.extend(brightness_targets)

            # Method 3: Edge-based detection for additional targets
            edge_targets = self._edge_based_detection(processed, image, range_setting, radar_type)
            targets.extend(edge_targets)

            # Remove duplicates and filter by confidence
            unique_targets = self._filter_and_deduplicate(targets)

            # Sort by range (closest first)
            unique_targets.sort(key=lambda x: x.position[0])

            # Reassign IDs based on sorted order
            for i, target in enumerate(unique_targets):
                target.target_id = i + 1

            # Create result dictionary
            result = self._create_result_dict(unique_targets, image_path)

            logger.info(f"Enhanced detection found {len(unique_targets)} targets")
            return result

        except Exception as e:
            logger.error(f"Enhanced detection error: {e}")
            return self._create_empty_result()

    def _enhanced_preprocess(self, image: np.ndarray, radar_type: str) -> np.ndarray:
        """Enhanced preprocessing for better target detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Enhance radar-specific features
        if radar_type == "marine":
            # Boost contrast for marine radar characteristics
            filtered = cv2.convertScaleAbs(filtered, alpha=1.2, beta=10)

        # Morphological operations to enhance target blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morphed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

        return morphed

    def _yolo_detection(self, processed: np.ndarray, original: np.ndarray,
                       range_setting: float, radar_type: str) -> List[RadarTarget]:
        """YOLO-based detection with enhanced processing."""
        try:
            # Prepare image for YOLO
            yolo_input = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

            # Run detection
            results = self.model(yolo_input, conf=self.confidence_threshold, iou=0.45)

            targets = []
            h, w = original.shape[:2]
            center = (w // 2, h // 2)
            radius = min(w, h) // 2 - 50

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    class_name = self.model.names.get(cls, 'unknown')
                    target_type = self.class_mapping.get(class_name.lower(), TargetType.UNKNOWN)

                    # Skip unknown targets with low confidence
                    if target_type == TargetType.UNKNOWN and conf < 0.5:
                        continue

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    range_nm, bearing = self._pixel_to_radar_coords(
                        (cx, cy), center, radius, range_setting
                    )

                    if range_nm > range_setting * 1.2:
                        continue

                    targets.append(RadarTarget(
                        target_id=0,  # Will be reassigned
                        target_type=target_type,
                        position=(range_nm, bearing),
                        pixel_position=(cx, cy),
                        size_estimate=self._estimate_size(x2-x1, y2-y1, radius, range_setting),
                        confidence=conf,
                        echo_strength=conf,
                        distance_meters=self._calculate_distance_meters(range_nm, radar_type),
                        risk_level=self._assess_risk_level(range_nm),
                        is_moving=False
                    ))

            return targets

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def _enhanced_brightness_detection(self, processed: np.ndarray, original: np.ndarray,
                                     range_setting: float, radar_type: str) -> List[RadarTarget]:
        """Enhanced brightness-based detection with multiple thresholds."""
        targets = []

        # Multiple threshold levels for different target types
        thresholds = [30, 50, 80, 120]

        h, w = original.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 50

        for threshold in thresholds:
            _, binary = cv2.threshold(processed, threshold, 255, cv2.THRESH_BINARY)

            # Clean up noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 5:  # Minimum area threshold
                    continue

                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Check if within radar range area
                pixel_distance = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
                if pixel_distance > radius:
                    continue

                range_nm, bearing = self._pixel_to_radar_coords(
                    (cx, cy), center, radius, range_setting
                )

                if range_nm > range_setting * 1.2:
                    continue

                # Classify based on area and intensity
                mean_intensity = cv2.mean(processed, mask=cv2.drawContours(
                    np.zeros_like(processed), [contour], -1, 255, -1
                ))[0]

                target_type = self._classify_by_area_and_intensity(area, mean_intensity)

                # Calculate confidence based on multiple factors
                confidence = self._calculate_brightness_confidence(
                    area, mean_intensity, threshold, pixel_distance, radius
                )

                targets.append(RadarTarget(
                    target_id=0,
                    target_type=target_type,
                    position=(range_nm, bearing),
                    pixel_position=(cx, cy),
                    size_estimate=np.sqrt(area) / radius * range_setting if radius > 0 else 0,
                    confidence=confidence,
                    echo_strength=mean_intensity / 255.0,
                    distance_meters=self._calculate_distance_meters(range_nm, radar_type),
                    risk_level=self._assess_risk_level(range_nm),
                    is_moving=False
                ))

        return targets

    def _edge_based_detection(self, processed: np.ndarray, original: np.ndarray,
                            range_setting: float, radar_type: str) -> List[RadarTarget]:
        """Edge-based detection for targets missed by other methods."""
        # Apply Canny edge detection
        edges = cv2.Canny(processed, 30, 100)

        # Dilate to connect edge fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours in edges
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        targets = []
        h, w = original.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 50

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10 or area > 1000:  # Reasonable size limits
                continue

            # Get bounding rectangle
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            cx = x + w_contour // 2
            cy = y + h_contour // 2

            # Check if within radar area
            pixel_distance = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
            if pixel_distance > radius:
                continue

            range_nm, bearing = self._pixel_to_radar_coords(
                (cx, cy), center, radius, range_setting
            )

            if range_nm > range_setting * 1.2:
                continue

            # Calculate edge strength
            edge_mask = np.zeros_like(processed)
            cv2.drawContours(edge_mask, [contour], -1, 255, -1)
            edge_strength = cv2.mean(processed, mask=edge_mask)[0]

            targets.append(RadarTarget(
                target_id=0,
                target_type=TargetType.UNKNOWN,  # Will be classified later
                position=(range_nm, bearing),
                pixel_position=(cx, cy),
                size_estimate=np.sqrt(area) / radius * range_setting if radius > 0 else 0,
                confidence=0.4,  # Lower confidence for edge-based detection
                echo_strength=edge_strength / 255.0,
                distance_meters=self._calculate_distance_meters(range_nm, radar_type),
                risk_level=self._assess_risk_level(range_nm),
                is_moving=False
            ))

        return targets

    def _classify_by_area_and_intensity(self, area: float, intensity: float) -> TargetType:
        """Classify targets based on area and intensity characteristics."""
        # Normalize intensity
        norm_intensity = intensity / 255.0

        if area > 500 and norm_intensity > 0.7:
            return TargetType.LANDMASS
        elif area > 100 and norm_intensity > 0.5:
            return TargetType.VESSEL
        elif area > 20 and norm_intensity > 0.3:
            return TargetType.OBSTACLE
        else:
            return TargetType.UNKNOWN

    def _calculate_brightness_confidence(self, area: float, intensity: float,
                                       threshold: int, pixel_distance: float,
                                       max_radius: float) -> float:
        """Calculate confidence score for brightness-based detection."""
        # Base confidence from area
        area_conf = min(area / 200.0, 1.0)

        # Intensity confidence
        intensity_conf = intensity / 255.0

        # Distance confidence (closer targets are more reliable)
        distance_conf = 1.0 - (pixel_distance / max_radius)

        # Threshold confidence (higher thresholds are more reliable)
        threshold_conf = threshold / 120.0

        # Combine confidences
        confidence = (area_conf * 0.3 + intensity_conf * 0.3 +
                     distance_conf * 0.2 + threshold_conf * 0.2)

        return min(confidence, 1.0)

    def _filter_and_deduplicate(self, targets: List[RadarTarget]) -> List[RadarTarget]:
        """Filter targets by confidence and remove duplicates."""
        # Filter by confidence
        filtered = [t for t in targets if t.confidence >= self.confidence_threshold]

        # Remove duplicates based on proximity
        unique_targets = []
        for target in filtered:
            is_duplicate = False
            for existing in unique_targets:
                # Check if targets are too close (within 0.1 NM and 5 degrees)
                range_diff = abs(target.position[0] - existing.position[0])
                bearing_diff = min(abs(target.position[1] - existing.position[1]),
                                 360 - abs(target.position[1] - existing.position[1]))

                if range_diff < 0.1 and bearing_diff < 5:
                    # Keep the one with higher confidence
                    if target.confidence > existing.confidence:
                        existing = target
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_targets.append(target)

        return unique_targets

    def _calculate_distance_meters(self, range_nm: float, radar_type: str) -> float:
        """Calculate distance in meters from nautical miles."""
        # 1 nautical mile = 1852 meters
        return range_nm * 1852.0

    def _assess_risk_level(self, range_nm: float) -> str:
        """Assess collision risk based on range."""
        if range_nm < 0.5:
            return "CRITICAL"
        elif range_nm < 1.0:
            return "HIGH"
        elif range_nm < 3.0:
            return "MEDIUM"
        elif range_nm < 6.0:
            return "LOW"
        else:
            return "SAFE"

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

    def _create_empty_result(self) -> Dict:
        """Create empty result dictionary."""
        return {
            'total': 0,
            'vessels': 0,
            'landmasses': 0,
            'obstacles': 0,
            'targets': [],
            'processing_time': 0.0
        }

    def _create_result_dict(self, targets: List[RadarTarget], image_path: str) -> Dict:
        """Create result dictionary from detected targets."""
        # Count by type
        vessels = sum(1 for t in targets if t.target_type == TargetType.VESSEL)
        landmasses = sum(1 for t in targets if t.target_type == TargetType.LANDMASS)
        obstacles = sum(1 for t in targets if t.target_type == TargetType.OBSTACLE)

        # Convert targets to dictionary format
        targets_dict = []
        for target in targets:
            targets_dict.append({
                'id': target.target_id,
                'type': target.target_type.value,
                'range_nm': round(target.position[0], 2),
                'bearing_deg': round(target.position[1], 1),
                'distance_meters': round(target.distance_meters, 0),
                'pixel_position': target.pixel_position,
                'size_estimate': round(target.size_estimate, 2),
                'confidence': round(target.confidence, 3),
                'echo_strength': round(target.echo_strength, 3),
                'risk_level': target.risk_level
            })

        return {
            'total': len(targets),
            'vessels': vessels,
            'landmasses': landmasses,
            'obstacles': obstacles,
            'targets': targets_dict,
            'processing_time': 0.0
        }