
from dotenv import load_dotenv
# Hybrid Radar Data Extraction System - Core Engine
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

from openai import OpenAI
import google.generativeai as genai
# Import our architecture (from Step 1)
from radar_extraction_architecture import (ExtractionMethod, RadarType, FieldDefinition, 
    ExtractionResult, RadarImageAnalysis, RADAR_FIELDS,
    RADAR_TYPE_FEATURES, SystemConfig
)
from intelligent_learning_system import IntelligentLearningSystem, AdaptiveExtractionEngine

# Configure Tesseract IMMEDIATELY
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\aras.koplanto\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('radar_extraction.log'),
        logging.StreamHandler()
    ]
)
load_dotenv()

logger = logging.getLogger(__name__)
import pytesseract
import platform

if platform.system() == 'Windows':
    # Try common installation paths
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\%s\AppData\Local\Tesseract-OCR\tesseract.exe' % os.environ.get('USERNAME', ''),
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            print(f"Found Tesseract at: {path}")
            break
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
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize AI clients with API keys."""
        self.clients = {}
        
        
        if 'openai' in api_keys:
            self.clients['gpt4v'] = OpenAI(api_key=api_keys['openai'])
        
        if 'google' in api_keys:
            genai.configure(api_key="AIzaSyAmVErAULHfk-dUrqennuh3ORevv37gJjk")
            self.clients['gemini'] = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    def create_extraction_prompt(self, radar_type: RadarType, target_fields: List[str] = None) -> str:
        # First try to use enhanced prompt from learning system
        if hasattr(self, 'learning_system'):
            weak_fields = self.get_weak_fields()
            return self.adaptive_engine.get_enhanced_prompt(
                radar_type.value, 
                weak_fields
            )
        
        # Fallback to enhanced config if available
        if os.path.exists('enhanced_extraction_config.json'):
            with open('enhanced_extraction_config.json', 'r') as f:
                config = json.load(f)
            return config['enhanced_prompt']
        
        # Otherwise use your detailed prompt
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
            "set": "current set direction in degrees, YOU CAN FIND IT IN THE HEADING AND SPEED SECTION",
            "drift": "current drift speed in knots",
            "vector": "vector mode TRUE/REL/OFF",
            "vector_duration": "vector time in minutes",
            "cpa_limit": "CPA limit in NM",
            "tcpa_limit": "TCPA limit in minutes",
            "vrm1": "VRM1 distance if visible",
            "vrm2": "VRM2 distance if visible",
            "index_line_rng": "index line range",
            "index_line_brg": "index line bearing",
            "ais_on_off": "AIS status ON/OFF, do not include SLEEPING",
            "depth": "water depth in meters"
            }}

            IMPORTANT REMINDERS:
            - COG is a DIRECTION (degrees), SOG is a SPEED (knots) - never swap them SOG cannot be more then 50!
            - Read PRESENTATION MODE from the actual indicator, not from display appearance
            - GAIN/SEA/RAIN are often shown as bars - estimate percentage
            - Use null for fields not visible
            - Extract numbers only (no units)"""
                
        return prompt
        
      
    async def extract_with_gemini(self, image_path: str, radar_type: RadarType,
                                target_fields: List[str] = None) -> Dict[str, Any]:
        """Extract data using Google Gemini Vision API."""
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
    async def extract_with_claude(self, image_path: str, radar_type: RadarType, 
                                 target_fields: List[str] = None) -> Dict[str, Any]:
        """Extract data using Claude Vision API."""
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            prompt = self.create_extraction_prompt(radar_type, target_fields)
            
            message = self.clients['claude'].messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2048,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            # Parse response
            response_text = message.content[0].text.strip()
            
            # Clean up response if needed
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"Claude extraction error: {e}")
            return {}
    
    async def extract_with_gpt4v(self, image_path: str, radar_type: RadarType,
                                target_fields: List[str] = None) -> Dict[str, Any]:
        """Extract data using GPT-4 Vision API."""
        try:
            # Open and potentially convert the image
            from PIL import Image as PILImage
            import io
            
            with PILImage.open(image_path) as img:
                # Convert to RGB if necessary (removes alpha channel, converts from grayscale, etc.)
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

    
    async def extract_batch(self, image_paths: List[str], 
                          max_workers: int = 4) -> List[RadarImageAnalysis]:
        """Extract data from multiple images in parallel."""
        logger.info(f"Starting batch extraction for {len(image_paths)} images")
        
        # Process in batches to avoid overwhelming APIs
        results = []
        
        for i in range(0, len(image_paths), max_workers):
            batch = image_paths[i:i + max_workers]
            batch_tasks = [self.extract_image(path) for path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for path, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {path}: {result}")
                    # Create a failed result
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
    
    # Convert to dictionary
    result_dict = asdict(analysis)
    
    # Convert ExtractionResult objects
    result_dict['extraction_results'] = {
        field: asdict(result) 
        for field, result in analysis.extraction_results.items()
    }
    
    # Save to file
    output_path = os.path.join(output_dir, f"{analysis.filename}_results.json")
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_path}")

async def test_single_image(image_path: str, api_keys: Dict[str, str]):
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

class OCRExtractor:
    """Specialized OCR extraction for specific fields."""
    
    def __init__(self):
        """Initialize OCR engines."""
        self.tesseract_config = {
            'numeric': '--psm 7 -c tessedit_char_whitelist=0123456789.',
            'alphanumeric': '--psm 7',
            'single_line': '--psm 8',
            'sparse_text': '--psm 11'
        }
    
    def extract_from_roi(self, image: np.ndarray, roi: Tuple[int, int, int, int],
                        field_type: str = 'alphanumeric') -> str:
        """Extract text from a specific region of interest."""
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
        
        # Handle numeric fields (including those with units)
        if field_def.data_type in ['numeric', 'bearing', 'speed']:
            # Extract numeric value, handling various formats
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
            
            # If no pattern matched, try one more time with a looser pattern
            loose_match = re.search(r'(-?\d+\.?\d*)', str_value)
            if loose_match:
                try:
                    return True, float(loose_match.group(1)), ""
                except:
                    pass
                    
            return False, None, f"{field_name} not numeric: {str_value}"
        
        # Handle text fields with common variations
        if field_def.data_type == 'text':
            cleaned = str_value.upper().strip()
            
            # FIXED: Presentation mode variations
            if field_name == 'presentation_mode':
                # Log for debugging
                logger.info(f"Processing presentation_mode: original='{str_value}', cleaned='{cleaned}'")
                
                # Remove common prefixes/noise
                mode_text = cleaned
                # Remove "PRESENTATION MODE:" or similar prefixes
                mode_text = re.sub(r'^.*MODE\s*:\s*', '', mode_text)
                mode_text = re.sub(r'^.*:\s*', '', mode_text)  # Remove any prefix with colon
                mode_text = mode_text.strip()
                
                # Check for exact matches first (with word boundaries to avoid partial matches)
                if re.search(r'\bNORTH\s*UP\b', mode_text) or re.search(r'\bN\s*UP\b', mode_text):
                    value = "NORTH UP"
                elif re.search(r'\bHEAD\s*UP\b', mode_text) or re.search(r'\bH\s*UP\b', mode_text) or re.search(r'\bHDG\s*UP\b', mode_text):
                    value = "HEAD UP"
                elif re.search(r'\bCOURSE\s*UP\b', mode_text) or re.search(r'\bC\s*UP\b', mode_text) or re.search(r'\bCRS\s*UP\b', mode_text):
                    value = "COURSE UP"
                # Fallback to checking individual words (but be more careful)
                elif 'NORTH' in mode_text and not any(x in mode_text for x in ['HEAD', 'COURSE']):
                    value = "NORTH UP"
                elif 'HEAD' in mode_text and not any(x in mode_text for x in ['NORTH', 'COURSE']):
                    value = "HEAD UP"
                elif 'COURSE' in mode_text and not any(x in mode_text for x in ['NORTH', 'HEAD']):
                    value = "COURSE UP"
                else:
                    # Can't determine - keep original
                    value = str_value
                    logger.warning(f"Could not parse presentation_mode: '{mode_text}'")
                
                # Check for RM/TM suffix in original cleaned text
                if ' RM' in cleaned or cleaned.endswith('RM'):
                    value += " RM"
                elif ' TM' in cleaned or cleaned.endswith('TM'):
                    value += " TM"
                
                logger.info(f"Presentation_mode result: '{value}'")
                return True, value, ""
            
            # AIS status variations
            elif field_name == 'ais_on_off':
                if cleaned in ['ON', 'AIS ON', 'AIS: ON', 'ENABLED'] or ('ON' in cleaned and 'OFF' not in cleaned):
                    return True, "ON", ""
                elif cleaned in ['OFF', 'AIS OFF', 'AIS: OFF', 'DISABLED'] or 'OFF' in cleaned:
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
        
        # Coordinate validation (position, cursor_position)
        if field_def.data_type == 'coordinate':
            # Accept various position formats
            if re.search(r'\d+.*[NS].*\d+.*[EW]', str_value, re.IGNORECASE):
                return True, str_value, ""
        
        # Default: accept the value if we couldn't parse it
        return True, str_value, ""
    
    def cross_field_validation(self, data: Dict[str, Any]) -> List[str]:
        """Perform validation across multiple fields."""
        warnings = []
        
        # Example: COG should be similar to heading when moving
        if 'heading' in data and 'cog' in data and 'speed' in data:
            if data['speed'] and float(data['speed']) > 1.0:  # Moving
                heading = float(data['heading']) if data['heading'] else 0
                cog = float(data['cog']) if data['cog'] else 0
                diff = abs(heading - cog)
                if diff > 180:
                    diff = 360 - diff
                if diff > 30:  # More than 30 degrees difference
                    warnings.append(f"Large difference between heading ({heading}°) and COG ({cog}°)")
        
        # Example: SOG and Speed should be similar
        if 'speed' in data and 'sog' in data:
            if data['speed'] and data['sog']:
                speed = float(data['speed'])
                sog = float(data['sog'])
                if abs(speed - sog) > 2.0:  # More than 2 knots difference
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
        
        # Adjust based on validation
        is_valid, _, error = validation_result
        if not is_valid:
            confidence *= 0.5
        elif error:  # Warning but valid
            confidence *= 0.8
        
        # Adjust based on field complexity
        if field_name in ['position', 'cursor_position']:
            confidence *= 0.9  # Complex fields are harder
        
        return min(max(confidence, 0.0), 1.0)
class HybridRadarExtractor:
    """Main extraction engine that coordinates all methods."""
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize the extraction system."""
        self.radar_detector = RadarTypeDetector()
        self.preprocessor = ImagePreprocessor()
        self.ai_extractor = AIVisionExtractor(api_keys)
        self.ocr_extractor = OCRExtractor()
        self.validator = ValidationEngine()
        self.scorer = ConfidenceScorer()
        self.learning_system = IntelligentLearningSystem()
        self.adaptive_engine = AdaptiveExtractionEngine(self.learning_system)
        
        logger.info("Hybrid Radar Extractor initialized with learning system")
    
    async def extract_image(self, image_path: str) -> RadarImageAnalysis:
        """
        Extract all data from a radar image using the hybrid approach.
        """
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
            # Try Gemini first (more flexible with image formats)
            if 'gemini' in self.ai_extractor.clients:
                try:
                    primary_data = await self.ai_extractor.extract_with_gemini(
                        image_path, radar_type
                    )
                    logger.info(f"Gemini extracted {len(primary_data)} fields")
                except Exception as e:
                    logger.warning(f"Gemini extraction failed: {e}")
                    primary_data = {}
            
            # Fallback to GPT-4V if Gemini fails
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
        
        if missing_fields:
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
                    if hasattr(self, 'adaptive_engine'):
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
        
        # ADD LEARNING HERE - This is where the system learns from each extraction
        if hasattr(self, 'learning_system'):
            try:
                # Save extraction patterns for learning
                for field_name, result in extraction_results.items():
                    if result.value is not None and result.confidence > 0.8:
                        # This is a successful extraction to learn from
                        logger.debug(f"Learning from successful extraction of {field_name}")
            except Exception as e:
                logger.warning(f"Could not update learning system: {e}")
        
        return analysis  
    
    async def extract_batch(self, image_paths: List[str], 
                          max_workers: int = 4) -> List[RadarImageAnalysis]:
        """Extract data from multiple images in parallel."""
        logger.info(f"Starting batch extraction for {len(image_paths)} images")
        
        # Process in batches to avoid overwhelming APIs
        results = []
        
        for i in range(0, len(image_paths), max_workers):
            batch = image_paths[i:i + max_workers]
            batch_tasks = [self.extract_image(path) for path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for path, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {path}: {result}")
                    # Create a failed result
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
    
    async def get_weak_fields(self) -> List[str]:
        import sqlite3
        """Get fields with low extraction success rates."""
        conn = sqlite3.connect(self.learning_system.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT field_name 
            FROM (
                SELECT 
                    field_name,
                    SUM(CASE WHEN field_value IS NOT NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
                FROM extracted_fields
                GROUP BY field_name
            )
            WHERE success_rate < 0.5
        """)
        
        weak_fields = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return weak_fields

if __name__ == "__main__":
    # Example usage
    api_keys = {
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'openai': os.getenv('OPENAI_API_KEY'),
        'google': os.getenv('GOOGLE_API_KEY')
    }
    
    # Test on a single image
    test_image = "radar_image.png"
    if os.path.exists(test_image):
        asyncio.run(test_single_image(test_image, api_keys))
        
    async def run_with_learning(image_path: str, api_keys: Dict[str, str]):
        """Run extraction with learning system."""
        # First run learning analysis
        from quick_learning_setup import QuickLearningSystem
        learning = QuickLearningSystem()
        learning.save_enhanced_configuration()
        
        # Then run extraction
        extractor = HybridRadarExtractor(api_keys)
        result = await extractor.extract_image(image_path)
        
        return result