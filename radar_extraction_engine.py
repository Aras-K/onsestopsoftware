# Hybrid Radar Data Extraction System - Core Engine (Azure Compatible)
# This implements the main extraction logic with multiple methods

import os
import cv2
import numpy as np
import json
import base64
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import asyncio
import concurrent.futures
from PIL import Image, ImageEnhance
import pytesseract
import re
import platform
import subprocess
import logging
from openai import OpenAI
import google.generativeai as genai

# Import our architecture
from radar_extraction_architecture import (
    ExtractionMethod, RadarType, FieldDefinition, 
    ExtractionResult, RadarImageAnalysis, RADAR_FIELDS,
    RADAR_TYPE_FEATURES, SystemConfig
)
logger = logging.getLogger(__name__)
try:
    from radar_target_detection import YOLORadarDetector, TargetType
    TARGET_DETECTION_AVAILABLE = True
except ImportError:
    logger.warning("Target detection module not available")
    YOLORadarDetector = None
    TargetType = None
    TARGET_DETECTION_AVAILABLE = False
# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('radar_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Tesseract for Azure environment
def configure_tesseract():
    """Configure Tesseract OCR for the current environment."""
    # Check if running on Azure (Linux container)
    if platform.system() == 'Linux':
        # Azure App Service Linux container
        tesseract_cmd = 'tesseract'
        
        # Check if tesseract is installed
        try:
            subprocess.run(['which', 'tesseract'], check=True, capture_output=True)
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            logger.info(f"Tesseract configured for Linux: {tesseract_cmd}")
        except subprocess.CalledProcessError:
            logger.warning("Tesseract not found. Installing may be required.")
            # In Azure, you might need to install it in your startup script
            
    elif platform.system() == 'Windows':
        # Local Windows development
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.expandvars(r'%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe'),
            os.path.expandvars(r'%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe'),
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Tesseract configured for Windows: {path}")
                break
        else:
            logger.warning("Tesseract not found on Windows. OCR features may not work.")
    
    else:
        # Mac or other Unix-like systems
        pytesseract.pytesseract.tesseract_cmd = 'tesseract'
        logger.info("Tesseract configured for Unix-like system")

# Initialize Tesseract configuration
configure_tesseract()

class RadarTypeDetector:
    """Detects the type of radar display using computer vision techniques."""
    
    @staticmethod
    def detect_radar_type(image_path: str) -> Tuple[RadarType, float]:
        """
        Detect radar type with confidence score.
        Returns: (RadarType, confidence_score)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return RadarType.UNKNOWN, 0.0
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Check for Furuno Classic (green display)
            green_lower = np.array([40, 40, 40])
            green_upper = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
            
            if green_ratio > 0.15:  # More than 15% green pixels
                logger.info(f"Detected Furuno Classic radar (green ratio: {green_ratio:.2%})")
                return RadarType.FURUNO_CLASSIC, min(green_ratio * 5, 1.0)
            
            # Check for Modern Dark interface
            gray_lower = np.array([0, 0, 20])
            gray_upper = np.array([180, 30, 100])
            gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
            gray_ratio = np.sum(gray_mask > 0) / (image.shape[0] * image.shape[1])
            
            if gray_ratio > 0.3:  # More than 30% dark gray
                logger.info(f"Detected Modern Dark radar (gray ratio: {gray_ratio:.2%})")
                return RadarType.MODERN_DARK, min(gray_ratio * 2, 1.0)
            
            return RadarType.UNKNOWN, 0.5
            
        except Exception as e:
            logger.error(f"Error detecting radar type: {e}")
            return RadarType.UNKNOWN, 0.0

class ImagePreprocessor:
    """Advanced image preprocessing for better extraction accuracy."""
    
    @staticmethod
    def enhance_image(image: np.ndarray, radar_type: RadarType) -> np.ndarray:
        """Apply radar-specific image enhancement."""
        enhanced = image.copy()
        
        if radar_type == RadarType.FURUNO_CLASSIC:
            # Enhance green channel for Furuno
            b, g, r = cv2.split(enhanced)
            g = cv2.equalizeHist(g)
            enhanced = cv2.merge([b, g, r])
            
            # Increase contrast
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            
        elif radar_type == RadarType.MODERN_DARK:
            # Enhance contrast for dark displays
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=20)
            
            # Denoise
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Sharpen text
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    @staticmethod
    def extract_text_regions(image: np.ndarray, radar_type: RadarType) -> List[Dict[str, Any]]:
        """Extract regions likely to contain text."""
        regions = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply appropriate threshold based on radar type
        if radar_type == RadarType.FURUNO_CLASSIC:
            # For green text on black
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        else:
            # For white/gray text
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size (likely text regions)
            if 10 < w < 500 and 8 < h < 50:
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': w * h,
                    'aspect_ratio': w / h
                })
        
        # Sort by position (top to bottom, left to right)
        regions.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))
        
        return regions

class AIVisionExtractor:
    """Handles extraction using AI vision APIs (Claude, GPT-4V, Gemini)."""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """Initialize AI clients with API keys from environment or parameters."""
        self.clients = {}
        
        # Use environment variables if api_keys not provided
        if api_keys is None:
            api_keys = {
                'openai': os.environ.get('OPENAI_API_KEY'),
                'google': os.environ.get('GOOGLE_API_KEY'),
                'anthropic': os.environ.get('ANTHROPIC_API_KEY')
            }
        
        # Initialize OpenAI client
        if api_keys.get('openai'):
            try:
                self.clients['gpt4v'] = OpenAI(api_key=api_keys['openai'])
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Initialize Google Gemini client
        if api_keys.get('google'):
            try:
                genai.configure(api_key=api_keys['google'])
                self.clients['gemini'] = genai.GenerativeModel('gemini-1.5-pro-latest')
                logger.info("Gemini client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
        
        if not self.clients:
            logger.warning("No AI vision clients initialized. AI extraction will not be available.")
    
    def create_extraction_prompt(self, radar_type: RadarType, target_fields: List[str] = None) -> str:
        """Create the extraction prompt for AI models."""
        if target_fields is None:
            target_fields = list(RADAR_FIELDS.keys())
        
        prompt = f"""Analyze this {radar_type.value} marine radar display image and extract ALL the following data fields.

CRITICAL INSTRUCTIONS - READ CAREFULLY:

1. "presentation_mode": "READ THE ACTUAL TEXT - look for 'NORTH UP', 'N UP', 'HEAD UP', 'H UP' text on screen - DO NOT GUESS!",

2. COG vs SOG - DO NOT CONFUSE THESE:
   - COG (Course Over Ground) = Direction of movement in degrees (000-360)
   - SOG (Speed Over Ground) = Speed of movement in knots (typically 0-30)
   - COG is ALWAYS a bearing/direction (degrees)
   - SOG is ALWAYS a speed (knots)
   - Look for labels like "COG", "SOG", "GND CRS", "GND SPD"

3. GAIN/SEA/RAIN CLUTTER:
   - These are often shown as BARS or PERCENTAGE indicators
   - Look for labels: "GAIN", "SEA", "RAIN", "A/C SEA", "A/C RAIN"
   - May be in a control panel area or side menu
   - Values are 0-100 (percentage)
   - If shown as a bar, estimate the percentage filled

Return ONLY a JSON object with these exact field names (use lowercase with underscores):
{{
  "presentation_mode": "IN THE DISPLAY ORIENTATION INDICATOR",
  "heading": "ship heading in degrees (HDG)",
  "speed": "ship speed through water in knots (SPD)",
  "cog": "course over ground in DEGREES (not speed!)",
  "sog": "speed over ground in KNOTS (not direction!)",
  "position": "exact position as shown",
  "position_source": "GPS/DGPS/GNSS",
  "gain": "gain percentage 0-100 (look for GAIN control/bar)",
  "sea_clutter": "sea clutter percentage 0-100 (look for SEA or A/C SEA)",
  "rain_clutter": "rain clutter percentage 0-100 (look for RAIN or A/C RAIN)",
  "tune": "tune percentage if visible",
  "range": "radar range in NM",
  "range_rings": "range ring interval",
  "cursor_position": "cursor position if visible",
  "set": "current set direction in degrees",
  "drift": "current drift speed in knots",
  "vector": "vector mode TRUE/REL/OFF",
  "vector_duration": "vector time in minutes",
  "cpa_limit": "CPA limit in NM",
  "tcpa_limit": "TCPA limit in minutes",
  "vrm1": "VRM1 distance if visible",
  "vrm2": "VRM2 distance if visible",
  "index_line_rng": "index line range",
  "index_line_brg": "index line bearing",
  "ais_on_off": "AIS status ON/OFF",
  "depth": "water depth in meters"
}}

IMPORTANT REMINDERS:
- COG is a DIRECTION (degrees), SOG is a SPEED (knots) - never swap them
- Read PRESENTATION MODE from the actual indicator, not from display appearance
- GAIN/SEA/RAIN are often shown as bars - estimate percentage
- Use null for fields not visible
- Extract numbers only (no units)"""
        
        return prompt
    
    async def extract_with_gemini(self, image_path: str, radar_type: RadarType,
                                target_fields: List[str] = None) -> Dict[str, Any]:
        """Extract data using Google Gemini Vision API."""
        if 'gemini' not in self.clients:
            logger.warning("Gemini client not available")
            return {}
            
        try:
            img = Image.open(image_path)
            prompt = self.create_extraction_prompt(radar_type, target_fields)
            
            response = self.clients['gemini'].generate_content([prompt, img])
            
            if response and response.text:
                response_text = response.text.strip()
                # Clean markdown if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                return json.loads(response_text)
            
            return {}
            
        except Exception as e:
            logger.error(f"Gemini extraction error: {e}")
            return {}
    
    async def extract_with_gpt4v(self, image_path: str, radar_type: RadarType,
                                target_fields: List[str] = None) -> Dict[str, Any]:
        """Extract data using GPT-4 Vision API."""
        if 'gpt4v' not in self.clients:
            logger.warning("GPT-4V client not available")
            return {}
            
        try:
            # Open and potentially convert the image
            import io
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Save to bytes buffer as PNG
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                
                # Encode the buffer content
                image_data = base64.b64encode(buffer.read()).decode('utf-8')
            
            prompt = self.create_extraction_prompt(radar_type, target_fields)
            
            response = self.clients['gpt4v'].chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }],
                temperature=0,
                max_tokens=2048
            )
            
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"GPT-4V extraction error: {e}")
            return {}

class OCRExtractor:
    """Specialized OCR extraction for specific fields."""
    
    def __init__(self):
        """Initialize OCR engines."""
        self.tesseract_available = self._check_tesseract()
        self.tesseract_config = {
            'numeric': '--psm 7 -c tessedit_char_whitelist=0123456789.',
            'alphanumeric': '--psm 7',
            'single_line': '--psm 8',
            'sparse_text': '--psm 11'
        }
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False
    
    def extract_from_roi(self, image: np.ndarray, roi: Tuple[int, int, int, int],
                        field_type: str = 'alphanumeric') -> str:
        """Extract text from a specific region of interest."""
        if not self.tesseract_available:
            return ""
            
        try:
            x1, y1, x2, y2 = roi
            roi_image = image[y1:y2, x1:x2]
            
            # Preprocess ROI
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if len(roi_image.shape) == 3 else roi_image
            
            # Scale up for better OCR
            scale = 3
            width = int(gray.shape[1] * scale)
            height = int(gray.shape[0] * scale)
            scaled = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # Apply threshold
            _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR
            config = self.tesseract_config.get(field_type, self.tesseract_config['alphanumeric'])
            text = pytesseract.image_to_string(binary, config=config).strip()
            
            return text
        except Exception as e:
            logger.warning(f"OCR extraction error: {e}")
            return ""
    
    def extract_field(self, image: np.ndarray, field_name: str, 
                     radar_type: RadarType) -> Optional[str]:
        """Extract a specific field using OCR."""
        if not self.tesseract_available:
            return None
            
        if field_name not in RADAR_FIELDS:
            return None
        
        field_def = RADAR_FIELDS[field_name]
        
        # Get ROI for this field and radar type
        if radar_type in field_def.roi_coordinates:
            roi = field_def.roi_coordinates[radar_type]
            
            # Determine extraction type based on field data type
            if field_def.data_type == 'numeric':
                ocr_type = 'numeric'
            else:
                ocr_type = 'alphanumeric'
            
            return self.extract_from_roi(image, roi, ocr_type)
        
        return None

class ValidationEngine:
    """Validates and corrects extracted data."""
    
    def validate_field(self, field_name: str, value: Any) -> Tuple[bool, Optional[Any], str]:
        """
        Validate a field value against its definition.
        Returns: (is_valid, corrected_value, error_message)
        """
        if field_name not in RADAR_FIELDS:
            return False, None, f"Unknown field: {field_name}"
        
        field_def = RADAR_FIELDS[field_name]
        
        # Handle null values
        if value is None or value == "" or str(value).lower() == "null":
            if field_def.required:
                return False, None, f"{field_name} is required but null"
            return True, None, ""
        
        # Convert to string for processing
        str_value = str(value).strip()
        
        # Handle numeric fields
        if field_def.data_type in ['numeric', 'bearing', 'speed']:
            # Extract numeric value
            numeric_patterns = [
                r'^(-?\d+\.?\d*)\s*(?:kn|kt|knots|KN|KT|KNOTS)?$',  # Speed
                r'^(-?\d+\.?\d*)\s*(?:°|deg|degrees|DEG)?$',         # Bearing
                r'^(-?\d+\.?\d*)\s*(?:nm|NM|Nm)?$',                  # Range
                r'^(-?\d+\.?\d*)\s*(?:%|percent)?$',                 # Percentage
                r'^(-?\d+\.?\d*)\s*(?:m|meters|M)?$',                # Depth
                r'^(-?\d+\.?\d*)\s*(?:min|MIN|minutes)?$',           # Time
                r'^(-?\d+\.?\d*)$'                                   # Plain number
            ]
            
            for pattern in numeric_patterns:
                match = re.match(pattern, str_value, re.IGNORECASE)
                if match:
                    try:
                        num_value = float(match.group(1))
                        
                        # Validate range
                        if field_def.min_value is not None and num_value < field_def.min_value:
                            return False, None, f"{field_name} below minimum: {num_value} < {field_def.min_value}"
                        
                        if field_def.max_value is not None and num_value > field_def.max_value:
                            return False, None, f"{field_name} above maximum: {num_value} > {field_def.max_value}"
                        
                        return True, num_value, ""
                    except ValueError:
                        continue
            
            # Try loose pattern
            loose_match = re.search(r'(-?\d+\.?\d*)', str_value)
            if loose_match:
                try:
                    return True, float(loose_match.group(1)), ""
                except:
                    pass
                    
            return False, None, f"{field_name} not numeric: {str_value}"
        
        # Handle text fields
        if field_def.data_type == 'text':
            cleaned = str_value.upper().strip()
            
            # Presentation mode
            if field_name == 'presentation_mode':
                mode_text = re.sub(r'^.*MODE\s*:\s*', '', cleaned)
                mode_text = re.sub(r'^.*:\s*', '', mode_text)
                mode_text = mode_text.strip()
                
                if re.search(r'\bNORTH\s*UP\b', mode_text) or re.search(r'\bN\s*UP\b', mode_text):
                    value = "NORTH UP"
                elif re.search(r'\bHEAD\s*UP\b', mode_text) or re.search(r'\bH\s*UP\b', mode_text):
                    value = "HEAD UP"
                elif re.search(r'\bCOURSE\s*UP\b', mode_text) or re.search(r'\bC\s*UP\b', mode_text):
                    value = "COURSE UP"
                else:
                    value = str_value
                
                if ' RM' in cleaned or cleaned.endswith('RM'):
                    value += " RM"
                elif ' TM' in cleaned or cleaned.endswith('TM'):
                    value += " TM"
                
                return True, value, ""
            
            # AIS status
            elif field_name == 'ais_on_off':
                if 'ON' in cleaned and 'OFF' not in cleaned:
                    return True, "ON", ""
                elif 'OFF' in cleaned:
                    return True, "OFF", ""
            
            # Vector mode
            elif field_name == 'vector':
                if 'TRUE' in cleaned or 'TRU' in cleaned:
                    return True, "TRUE", ""
                elif 'REL' in cleaned:
                    return True, "REL", ""
                elif 'OFF' in cleaned:
                    return True, "OFF", ""
            
            # Position source
            elif field_name == 'position_source':
                if 'DGPS' in cleaned:
                    return True, "DGPS", ""
                elif 'GPS' in cleaned:
                    return True, "GPS", ""
                elif 'GNSS' in cleaned:
                    return True, "GNSS", ""
        
        # Coordinate validation
        if field_def.data_type == 'coordinate':
            if re.search(r'\d+.*[NS].*\d+.*[EW]', str_value, re.IGNORECASE):
                return True, str_value, ""
        
        # Default: accept the value
        return True, str_value, ""
    
    def cross_field_validation(self, data: Dict[str, Any]) -> List[str]:
        """Perform validation across multiple fields."""
        warnings = []
        
        # COG vs heading validation
        if 'heading' in data and 'cog' in data and 'speed' in data:
            if data['speed'] and float(data['speed']) > 1.0:
                heading = float(data['heading']) if data['heading'] else 0
                cog = float(data['cog']) if data['cog'] else 0
                diff = abs(heading - cog)
                if diff > 180:
                    diff = 360 - diff
                if diff > 30:
                    warnings.append(f"Large difference between heading ({heading}°) and COG ({cog}°)")
        
        # SOG vs Speed validation
        if 'speed' in data and 'sog' in data:
            if data['speed'] and data['sog']:
                speed = float(data['speed'])
                sog = float(data['sog'])
                if abs(speed - sog) > 2.0:
                    warnings.append(f"Large difference between speed ({speed}kn) and SOG ({sog}kn)")
        
        return warnings

class ConfidenceScorer:
    """Calculate confidence scores for extracted data."""
    
    @staticmethod
    def calculate_field_confidence(field_name: str, value: Any, 
                                 method: ExtractionMethod, 
                                 validation_result: Tuple[bool, Any, str]) -> float:
        """Calculate confidence score for a single field."""
        base_confidence = {
            ExtractionMethod.AI_VISION: 0.9,
            ExtractionMethod.SPECIALIZED_OCR: 0.7,
            ExtractionMethod.TEMPLATE_MATCHING: 0.85,
            ExtractionMethod.PATTERN_MATCHING: 0.6,
            ExtractionMethod.MANUAL_REVIEW: 1.0
        }
        
        confidence = base_confidence.get(method, 0.5)
        
        
        is_valid, _, error = validation_result
        if not is_valid:
            confidence *= 0.5
        elif error:
            confidence *= 0.8
        
        # Adjust based on field complexity
        if field_name in ['position', 'cursor_position']:
            confidence *= 0.9
        
        return min(max(confidence, 0.0), 1.0)
class HybridRadarExtractor:
    """Main extraction engine that coordinates all methods."""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """Initialize the extraction system."""
        self.radar_detector = RadarTypeDetector()
        self.preprocessor = ImagePreprocessor()
        self.ai_extractor = AIVisionExtractor(api_keys)
        self.ocr_extractor = OCRExtractor()
        self.validator = ValidationEngine()
        self.scorer = ConfidenceScorer()
        self.target_detector = None

        if TARGET_DETECTION_AVAILABLE:
            try:
                self.target_detector = YOLORadarDetector()
                logger.info("YOLO target detection initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize YOLO: {e}")
        # Check if learning system is available (optional)
        self.learning_system = None
        self.adaptive_engine = None
        try:
            from intelligent_learning_system import IntelligentLearningSystem, AdaptiveExtractionEngine
            self.learning_system = IntelligentLearningSystem()
            self.adaptive_engine = AdaptiveExtractionEngine(self.learning_system)
            logger.info("Learning system initialized")
        except ImportError:
            logger.info("Learning system not available - running without adaptive features")
        
        logger.info("Hybrid Radar Extractor initialized")
    # In radar_extraction_engine.py, add this method to HybridRadarExtractor class:

    async def extract_with_target_detection(self, image_path: str, 
                                        skip_field_extraction: bool = False,
                                        detection_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract data with enhanced target detection capabilities.
        
        Args:
            image_path: Path to radar image
            skip_field_extraction: If True, only perform target detection
            detection_params: Optional parameters for target detection
                - min_target_size: Minimum target size in pixels
                - confidence_threshold: Minimum confidence for detection
                - detect_vessels: Enable vessel detection
                - detect_landmasses: Enable landmass detection
                - detect_obstacles: Enable obstacle detection
        
        Returns:
            Dictionary containing extraction results and detected targets
        """
        
        result = {
            'filename': os.path.basename(image_path),
            'timestamp': datetime.now().isoformat(),
            'field_extraction': None,
            'target_detection': None
        }
        
        # Field extraction (unless skipped)
        if not skip_field_extraction:
            analysis = await self.extract_image(image_path)
            result['field_extraction'] = {
                'radar_type': analysis.radar_type.value,
                'confidence': analysis.overall_confidence,
                'fields': {
                    k: {'value': v.value, 'confidence': v.confidence}
                    for k, v in analysis.extraction_results.items()
                }
            }
            range_setting = 12.0  # Default
            if 'range' in analysis.extraction_results:
                try:
                    range_setting = float(analysis.extraction_results['range'].value)
                except:
                    pass
            radar_type = analysis.radar_type.value
        else:
            # Manual radar type detection for target detection only
            detector = RadarTypeDetector()
            radar_type, _ = detector.detect_radar_type(image_path)
            radar_type = radar_type.value
            range_setting = detection_params.get('range_setting', 12.0) if detection_params else 12.0
        
        # Target detection
        if self.target_detector:
            try:
                # Apply custom detection parameters if provided
                if detection_params:
                    # You could pass these to the detect_targets method
                    # For now, using standard detection
                    pass
                
                targets = self.target_detector.detect_targets(
                    image_path,
                    radar_type,
                    range_setting
                )
                
                # Group targets by type
                target_groups = {
                    'vessels': [],
                    'landmasses': [],
                    'obstacles': [],
                    'unknown': []
                }
                
                for target in targets:
                    target_data = {
                        'id': target.target_id,
                        'range_nm': round(target.position[0], 2),
                        'bearing_deg': round(target.position[1], 1),
                        'confidence': round(target.confidence, 2),
                        'size_estimate_nm': round(target.size_estimate, 3),
                        'is_moving': target.is_moving,
                        'pixel_position': target.pixel_position
                    }
                    
                    if target.target_type.value == 'vessel':
                        target_groups['vessels'].append(target_data)
                    elif target.target_type.value == 'landmass':
                        target_groups['landmasses'].append(target_data)
                    elif target.target_type.value == 'obstacle':
                        target_groups['obstacles'].append(target_data)
                    else:
                        target_groups['unknown'].append(target_data)
                
                result['target_detection'] = {
                    'success': True,
                    'range_setting_nm': range_setting,
                    'radar_type': radar_type,
                    'summary': {
                        'total_targets': len(targets),
                        'vessels': len(target_groups['vessels']),
                        'landmasses': len(target_groups['landmasses']),
                        'obstacles': len(target_groups['obstacles']),
                        'unknown': len(target_groups['unknown'])
                    },
                    'targets_by_type': target_groups,
                    'collision_risks': self._assess_collision_risks(target_groups['vessels'])
                }
                
            except Exception as e:
                logger.error(f"Target detection failed: {e}")
                result['target_detection'] = {
                    'success': False,
                    'error': str(e)
                }
        else:
            result['target_detection'] = {
                'success': False,
                'error': 'Target detection module not available'
            }
        
        return result

    def _assess_collision_risks(self, vessels: List[Dict]) -> List[Dict]:
        """
        Assess collision risks based on detected vessels.
        
        Args:
            vessels: List of detected vessel targets
        
        Returns:
            List of collision risk assessments
        """
        risks = []
        
        for vessel in vessels:
            # Simple risk assessment based on range
            range_nm = vessel['range_nm']
            
            if range_nm < 1.0:
                risk_level = 'CRITICAL'
                action = 'Immediate action required'
            elif range_nm < 3.0:
                risk_level = 'HIGH'
                action = 'Close monitoring required'
            elif range_nm < 6.0:
                risk_level = 'MEDIUM'
                action = 'Monitor situation'
            else:
                risk_level = 'LOW'
                action = 'Normal monitoring'
            
            if vessel['is_moving']:
                # Moving targets are higher risk
                if risk_level == 'LOW':
                    risk_level = 'MEDIUM'
                elif risk_level == 'MEDIUM':
                    risk_level = 'HIGH'
            
            risks.append({
                'target_id': vessel['id'],
                'range_nm': range_nm,
                'bearing_deg': vessel['bearing_deg'],
                'risk_level': risk_level,
                'action_required': action,
                'is_moving': vessel['is_moving']
            })
        
        # Sort by risk level and range
        risk_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        risks.sort(key=lambda x: (risk_order[x['risk_level']], x['range_nm']))
        
        return risks
    

    async def extract_image(self, image_path: str) -> RadarImageAnalysis:
        """Extract all data from a radar image using the hybrid approach."""
        start_time = datetime.now()
        filename = os.path.basename(image_path)
        
        logger.info(f"Starting extraction for: {filename}")
        
        # Step 1: Detect radar type
        radar_type, type_confidence = self.radar_detector.detect_radar_type(image_path)
        logger.info(f"Detected radar type: {radar_type.value} (confidence: {type_confidence:.2f})")
        
        # Step 2: Preprocess image
        image = cv2.imread(image_path)
        enhanced_image = self.preprocessor.enhance_image(image, radar_type)
        
        # Step 3: Primary extraction with AI Vision
        extraction_results = {}
        primary_data = {}

        try:
            # Try Gemini first
            if 'gemini' in self.ai_extractor.clients:
                try:
                    primary_data = await self.ai_extractor.extract_with_gemini(
                        image_path, radar_type
                    )
                    logger.info(f"Gemini extracted {len(primary_data)} fields")
                except Exception as e:
                    logger.warning(f"Gemini extraction failed: {e}")
                    primary_data = {}
            
            # Fallback to GPT-4V
            if not primary_data and 'gpt4v' in self.ai_extractor.clients:
                try:
                    primary_data = await self.ai_extractor.extract_with_gpt4v(
                        image_path, radar_type
                    )
                    logger.info(f"GPT-4V extracted {len(primary_data)} fields")
                except Exception as e:
                    logger.warning(f"GPT-4V extraction failed: {e}")
                    primary_data = {}
                    
        except Exception as e:
            logger.error(f"Primary AI extraction failed: {e}")
        
        # Process primary results
        for field_name, value in primary_data.items():
            if field_name in RADAR_FIELDS:
                validation_result = self.validator.validate_field(field_name, value)
                is_valid, corrected_value, error = validation_result
                
                confidence = self.scorer.calculate_field_confidence(
                    field_name, value, ExtractionMethod.AI_VISION, validation_result
                )
                
                extraction_results[field_name] = ExtractionResult(
                    field_name=field_name,
                    value=corrected_value if is_valid else value,
                    confidence=confidence,
                    method_used=ExtractionMethod.AI_VISION,
                    raw_text=str(value),
                    error=error if not is_valid else None
                )
        
        # Step 4: Secondary extraction for missing/low-confidence fields
        missing_fields = [
            field for field in RADAR_FIELDS 
            if field not in extraction_results 
            or extraction_results[field].confidence < 0.7
        ]
        
        if missing_fields and self.ocr_extractor.tesseract_available:
            logger.info(f"Attempting OCR extraction for {len(missing_fields)} fields")
            
            for field_name in missing_fields:
                ocr_value = self.ocr_extractor.extract_field(
                    enhanced_image, field_name, radar_type
                )
                
                if ocr_value:
                    validation_result = self.validator.validate_field(field_name, ocr_value)
                    confidence = self.scorer.calculate_field_confidence(
                        field_name, ocr_value, ExtractionMethod.SPECIALIZED_OCR, validation_result
                    )
                    
                    # Apply adaptive boost if available
                    if self.adaptive_engine:
                        confidence = self.adaptive_engine.apply_confidence_boost(
                            field_name, radar_type.value, confidence
                        )
                    
                    # Only use OCR result if better than existing
                    if field_name not in extraction_results or \
                       confidence > extraction_results[field_name].confidence:
                        
                        extraction_results[field_name] = ExtractionResult(
                            field_name=field_name,
                            value=validation_result[1] if validation_result[0] else ocr_value,
                            confidence=confidence,
                            method_used=ExtractionMethod.SPECIALIZED_OCR,
                            raw_text=ocr_value
                        )
        
        # Step 5: Cross-field validation
        extracted_data = {
            field: result.value 
            for field, result in extraction_results.items() 
            if result.value is not None
        }
        validation_warnings = self.validator.cross_field_validation(extracted_data)
        
        # Step 6: Calculate overall confidence
        if extraction_results:
            overall_confidence = sum(
                r.confidence for r in extraction_results.values()
            ) / len(extraction_results)
        else:
            overall_confidence = 0.0
        
        # Determine if review is needed
        requires_review = (
            overall_confidence < SystemConfig.REQUIRE_REVIEW_THRESHOLD or
            len(validation_warnings) > 0 or
            len(extraction_results) < len([f for f in RADAR_FIELDS.values() if f.required])
        )
        
        # Create final analysis result
        processing_time = (datetime.now() - start_time).total_seconds()
        
        analysis = RadarImageAnalysis(
            filename=filename,
            radar_type=radar_type,
            extraction_results=extraction_results,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            requires_review=requires_review,
            validation_errors=validation_warnings,
            metadata={
                'radar_type_confidence': type_confidence,
                'extraction_methods_used': list(set(
                    r.method_used.value for r in extraction_results.values()
                )),
                'fields_extracted': len(extraction_results),
                'fields_missing': len(RADAR_FIELDS) - len(extraction_results)
            }
        )
        
        logger.info(f"Extraction complete: {filename} "
                   f"(confidence: {overall_confidence:.2f}, "
                   f"review needed: {requires_review})")
        if self.target_detector:
            try:
                # Get range setting from extracted fields
                range_setting = 12.0  # Default
                if 'range' in extraction_results:
                    range_value = extraction_results['range'].value
                    if range_value:
                        try:
                            range_setting = float(range_value)
                            logger.info(f"Using extracted range setting: {range_setting} NM")
                        except (ValueError, TypeError):
                            logger.warning(f"Could not parse range value: {range_value}")
                
                # Detect targets
                detected_targets = self.target_detector.detect_targets(
                    image_path,
                    radar_type.value,
                    range_setting
                )
                
                # Add detected targets to metadata
                analysis.metadata['detected_targets'] = {
                    'vessels': int(len([t for t in detected_targets if t.target_type == TargetType.VESSEL])),
                    'landmasses': int(len([t for t in detected_targets if t.target_type == TargetType.LANDMASS])),
                    'obstacles': int(len([t for t in detected_targets if t.target_type == TargetType.OBSTACLE])),
                    'unknown': int(len([t for t in detected_targets if t.target_type == TargetType.UNKNOWN])),
                    'total': int(len(detected_targets)),
                    'targets': [
                        {
                            'type': str(t.target_type.value),
                            'range_nm': float(round(t.position[0], 2)),
                            'bearing_deg': float(round(t.position[1], 1)),
                            'confidence': float(round(t.confidence, 3)),
                            'size_estimate': float(round(t.size_estimate, 3)),
                            'is_moving': bool(t.is_moving)
                        }
                        for t in detected_targets
                    ]
                }
                
                logger.info(f"Detected {len(detected_targets)} targets in {filename}")
                
            except Exception as e:
                logger.error(f"Target detection failed for {filename}: {e}")
                analysis.metadata['detected_targets'] = {
                    'error': str(e),
                    'total': 0
                }
        
        logger.info(f"Extraction complete: {filename} "
                   f"(confidence: {overall_confidence:.2f}, "
                   f"review needed: {requires_review})")
        
        
        return analysis
    
    async def extract_batch(self, image_paths: List[str], 
                          max_workers: int = 4) -> List[RadarImageAnalysis]:
        """Extract data from multiple images in parallel."""
        logger.info(f"Starting batch extraction for {len(image_paths)} images")
        
        results = []
        
        for i in range(0, len(image_paths), max_workers):
            batch = image_paths[i:i + max_workers]
            batch_tasks = [self.extract_image(path) for path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for path, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {path}: {result}")
                    results.append(RadarImageAnalysis(
                        filename=os.path.basename(path),
                        radar_type=RadarType.UNKNOWN,
                        extraction_results={},
                        overall_confidence=0.0,
                        processing_time=0.0,
                        requires_review=True,
                        validation_errors=[str(result)],
                        metadata={'error': str(result)}
                    ))
                else:
                    results.append(result)
        
        return results

# Utility functions for testing
def save_results(analysis: RadarImageAnalysis, output_dir: str):
    """Save extraction results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    result_dict = asdict(analysis)
    result_dict['extraction_results'] = {
        field: asdict(result) 
        for field, result in analysis.extraction_results.items()
    }
    
    output_path = os.path.join(output_dir, f"{analysis.filename}_results.json")
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_path}")

async def test_single_image(image_path: str, api_keys: Dict[str, str] = None):
    """Test extraction on a single image."""
    extractor = HybridRadarExtractor(api_keys)
    result = await extractor.extract_image(image_path)
    
    print(f"\n=== Extraction Results for {result.filename} ===")
    print(f"Radar Type: {result.radar_type.value}")
    print(f"Overall Confidence: {result.overall_confidence:.2%}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    print(f"Requires Review: {result.requires_review}")
    
    print("\nExtracted Fields:")
    for field_name, extraction in result.extraction_results.items():
        print(f"  {field_name}: {extraction.value} "
              f"(confidence: {extraction.confidence:.2%}, "
              f"method: {extraction.method_used.value})")
    
    if result.validation_errors:
        print("\nValidation Warnings:")
        for warning in result.validation_errors:
            print(f"  - {warning}")
    
    return result
# Add this test function at the bottom of radar_extraction_engine.py

async def test_target_detection(image_path: str, output_dir: str = "target_detection_results"):
    """Test the complete extraction pipeline with target detection."""
    import os
    import json
    from radar_visualization import EnhancedRadarVisualization
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Testing Target Detection on: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Initialize extractor
    extractor = HybridRadarExtractor()
    
    # Run extraction with target detection
    result = await extractor.extract_image(image_path)
    
    # Display results
    print(f"\n📊 EXTRACTION RESULTS:")
    print(f"  Radar Type: {result.radar_type.value}")
    print(f"  Overall Confidence: {result.overall_confidence:.2%}")
    print(f"  Fields Extracted: {len(result.extraction_results)}")
    
    # Display target detection results
    if 'detected_targets' in result.metadata:
        targets = result.metadata['detected_targets']
        
        print(f"\n🎯 TARGET DETECTION RESULTS:")
        print(f"  Total Targets: {targets.get('total', 0)}")
        print(f"  - Vessels: {targets.get('vessels', 0)}")
        print(f"  - Landmasses: {targets.get('landmasses', 0)}")
        print(f"  - Obstacles: {targets.get('obstacles', 0)}")
        print(f"  - Unknown: {targets.get('unknown', 0)}")
        
        # Show detailed target info
        if 'targets' in targets and targets['targets']:
            print(f"\n📍 DETAILED TARGET INFORMATION:")
            for i, target in enumerate(targets['targets'][:5], 1):  # Show first 5
                print(f"\n  Target {i}:")
                print(f"    Type: {target['type']}")
                print(f"    Range: {target['range_nm']:.2f} NM")
                print(f"    Bearing: {target['bearing_deg']:.1f}°")
                print(f"    Confidence: {target['confidence']:.1%}")
                print(f"    Moving: {'Yes' if target['is_moving'] else 'No'}")
        
        # Create visualization
        try:
            viz = EnhancedRadarVisualization()
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            viz_path = os.path.join(output_dir, f"{base_name}_targets.png")
            
            viz_image = viz.visualize_targets(image_path, targets)
            if viz_image is not None:
                import cv2
                cv2.imwrite(viz_path, viz_image)
                print(f"\n✅ Visualization saved to: {viz_path}")
        except Exception as e:
            print(f"\n⚠️ Could not create visualization: {e}")
    else:
        print(f"\n❌ No target detection results available")
    
    # Save results to JSON
    results_file = os.path.join(output_dir, f"{result.filename}_results.json")
    results_data = {
        'filename': result.filename,
        'radar_type': result.radar_type.value,
        'confidence': result.overall_confidence,
        'targets': result.metadata.get('detected_targets', {})
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"📄 Results saved to: {results_file}")
    print(f"\n{'='*60}\n")
    
    return result

# Add this to run the test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Use a default test image
        test_image = "radar_image.png"
    
    if os.path.exists(test_image):
        asyncio.run(test_target_detection(test_image))
    else:
        print(f"Image not found: {test_image}")
        print("Usage: python radar_extraction_engine.py <image_path>")
