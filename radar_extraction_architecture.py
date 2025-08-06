# Hybrid Radar Data Extraction System - Architecture & Configuration
# This file defines the complete system architecture

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json

class ExtractionMethod(Enum):
    """Extraction methods in order of preference."""
    AI_VISION = "ai_vision"
    SPECIALIZED_OCR = "specialized_ocr"
    TEMPLATE_MATCHING = "template_matching"
    PATTERN_MATCHING = "pattern_matching"
    MANUAL_REVIEW = "manual_review"

class RadarType(Enum):
    """Known radar display types."""
    FURUNO_CLASSIC = "furuno_classic"  # Green text on black
    MODERN_DARK = "modern_dark"        # Gray/white on dark
    FURUNO_FAR = "furuno_far"          # Newer Furuno models
    JRC = "jrc"                        # JRC radar displays
    UNKNOWN = "unknown"

@dataclass
class FieldDefinition:
    """Definition for each data field to extract."""
    name: str
    display_name: str
    data_type: str  # 'numeric', 'text', 'coordinate', 'bearing', 'speed'
    unit: Optional[str] = None
    validation_regex: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required: bool = True
    extraction_methods: List[ExtractionMethod] = field(default_factory=list)
    roi_coordinates: Dict[RadarType, Tuple[int, int, int, int]] = field(default_factory=dict)
    confidence_threshold: float = 0.8

@dataclass
class ExtractionResult:
    """Result from a single extraction attempt."""
    field_name: str
    value: Optional[Any]
    confidence: float
    method_used: ExtractionMethod
    raw_text: Optional[str] = None
    processing_time: float = 0.0
    error: Optional[str] = None

@dataclass
class RadarImageAnalysis:
    """Complete analysis result for a radar image."""
    filename: str
    radar_type: RadarType
    extraction_results: Dict[str, ExtractionResult]
    overall_confidence: float
    processing_time: float
    requires_review: bool
    validation_errors: List[str]
    metadata: Dict[str, Any]

# Field Definitions - Complete radar data schema
RADAR_FIELDS = {
    # Navigation Data
    "heading": FieldDefinition(
        name="heading",
        display_name="Heading",
        data_type="bearing",
        unit="degrees",
        validation_regex=r"^\d{1,3}\.?\d*°?$",
        min_value=0.0,
        max_value=360.0,
        required=True,
        extraction_methods=[ExtractionMethod.AI_VISION, ExtractionMethod.SPECIALIZED_OCR]
    ),
    
    "speed": FieldDefinition(
        name="speed",
        display_name="Speed",
        data_type="speed",
        unit="knots",
        validation_regex=r"^\d{1,2}\.?\d*\s*(?:kn|kt|knots)?$",
        min_value=0.0,
        max_value=50.0,
        required=True
    ),
    
    "cog": FieldDefinition(
        name="cog",
        display_name="COG",
        data_type="bearing",
        unit="degrees",
        validation_regex=r"^\d{1,3}\.?\d*°?\s*[TM]?$",
        min_value=0.0,
        max_value=360.0,
        required=True
    ),
    
    "sog": FieldDefinition(
        name="sog",
        display_name="SOG",
        data_type="speed",
        unit="knots",
        validation_regex=r"^\d{1,2}\.?\d*\s*(?:kn|kt|knots)?$",
        min_value=0.0,
        max_value=50.0,
        required=True
    ),
    
    "position": FieldDefinition(
        name="position",
        display_name="POSITION",
        data_type="coordinate",
        validation_regex=r"^\d{1,2}°?\d{1,2}\.?\d*\'?[NS]\s+\d{1,3}°?\d{1,2}\.?\d*\'?[EW]$",
        required=True
    ),
    
    "position_source": FieldDefinition(
        name="position_source",
        display_name="POSITION SOURCE",
        data_type="text",
        validation_regex=r"^(GPS|DGPS|GNSS|GLONASS|MANUAL|DR)$",
        required=True
    ),
    
    # Radar Display Settings
    "presentation_mode": FieldDefinition(
        name="presentation_mode",
        display_name="PRESENTATION MODE",
        data_type="text",
        validation_regex=r"^(HEAD UP|NORTH UP|COURSE UP)\s*(RM|TM)?$",
        required=True
    ),
    
    "gain": FieldDefinition(
        name="gain",
        display_name="GAIN",
        data_type="numeric",
        unit="percent",
        validation_regex=r"^\d{1,3}%?$",
        min_value=0,
        max_value=100,
        required=True
    ),
    
    "sea_clutter": FieldDefinition(
        name="sea_clutter",
        display_name="SEA CLUTTER",
        data_type="numeric",
        unit="percent",
        validation_regex=r"^\d{1,3}%?$",
        min_value=0,
        max_value=100,
        required=True
    ),
    
    "rain_clutter": FieldDefinition(
        name="rain_clutter",
        display_name="RAIN CLUTTER",
        data_type="numeric",
        unit="percent",
        validation_regex=r"^\d{1,3}%?$",
        min_value=0,
        max_value=100,
        required=True
    ),
    
    "tune": FieldDefinition(
        name="tune",
        display_name="TUNE",
        data_type="numeric",
        unit="percent",
        validation_regex=r"^\d{1,3}%?$",
        min_value=0,
        max_value=100,
        required=True
    ),
    
    # Range and Scale
    "range": FieldDefinition(
        name="range",
        display_name="RANGE",
        data_type="numeric",
        unit="NM",
        validation_regex=r"^\d+\.?\d*\s*(?:NM|nm)?$",
        min_value=0.125,
        max_value=96.0,
        required=True
    ),
    
    "range_rings": FieldDefinition(
        name="range_rings",
        display_name="RANGE RINGS",
        data_type="numeric",
        unit="NM",
        validation_regex=r"^\d+\.?\d*\s*(?:NM|nm)?$",
        min_value=0.0,
        max_value=96.0,
        required=True
    ),
    
    # Cursor Information
    "cursor_position": FieldDefinition(
        name="cursor_position",
        display_name="CURSOR POSITION",
        data_type="coordinate",
        validation_regex=r"^(\d{1,2}°?\d{1,2}\.?\d*\'?[NS]\s+\d{1,3}°?\d{1,2}\.?\d*\'?[EW])|(\d+\.?\d*°?\s+\d+\.?\d*\s*NM)$",
        required=False
    ),
    
    # Current/Drift
    "set": FieldDefinition(
        name="set",
        display_name="SET",
        data_type="bearing",
        unit="degrees",
        validation_regex=r"^\d{1,3}\.?\d*°?$",
        min_value=0.0,
        max_value=360.0,
        required=False
    ),
    
    "drift": FieldDefinition(
        name="drift",
        display_name="DRIFT",
        data_type="speed",
        unit="knots",
        validation_regex=r"^\d{1,2}\.?\d*\s*(?:kn|kt|knots)?$",
        min_value=0.0,
        max_value=10.0,
        required=False
    ),
    
    # Vector Settings
    "vector": FieldDefinition(
        name="vector",
        display_name="VECTOR",
        data_type="text",
        validation_regex=r"^(TRUE|REL|OFF)$",
        required=True
    ),
    
    "vector_duration": FieldDefinition(
        name="vector_duration",
        display_name="VECTOR DURATION",
        data_type="numeric",
        unit="minutes",
        validation_regex=r"^\d{1,2}\.?\d*\s*(?:MIN|min)?$",
        min_value=0.0,
        max_value=60.0,
        required=True
    ),
    
    # CPA/TCPA Settings
    "cpa_limit": FieldDefinition(
        name="cpa_limit",
        display_name="CPA LIMIT",
        data_type="numeric",
        unit="NM",
        validation_regex=r"^\d+\.?\d*\s*(?:NM|nm)?$",
        min_value=0.0,
        max_value=10.0,
        required=True
    ),
    
    "tcpa_limit": FieldDefinition(
        name="tcpa_limit",
        display_name="TCPA LIMIT",
        data_type="numeric",
        unit="minutes",
        validation_regex=r"^\d{1,2}\.?\d*\s*(?:MIN|min)?$",
        min_value=0.0,
        max_value=99.9,
        required=True
    ),
    
    # VRM (Variable Range Markers)
    "vrm1": FieldDefinition(
        name="vrm1",
        display_name="VRM1",
        data_type="numeric",
        unit="NM",
        validation_regex=r"^\d+\.?\d*\s*(?:NM|nm)?$",
        min_value=0.0,
        max_value=96.0,
        required=False
    ),
    
    "vrm2": FieldDefinition(
        name="vrm2",
        display_name="VRM2",
        data_type="numeric",
        unit="NM",
        validation_regex=r"^\d+\.?\d*\s*(?:NM|nm)?$",
        min_value=0.0,
        max_value=96.0,
        required=False
    ),
    
    # Index Lines
    "index_line_rng": FieldDefinition(
        name="index_line_rng",
        display_name="INDEX LINE RNG",
        data_type="numeric",
        unit="NM",
        validation_regex=r"^\d+\.?\d*\s*(?:NM|nm)?$",
        min_value=0.0,
        max_value=96.0,
        required=False
    ),
    
    "index_line_brg": FieldDefinition(
        name="index_line_brg",
        display_name="INDEX LINE BRG",
        data_type="bearing",
        unit="degrees",
        validation_regex=r"^\d{1,3}\.?\d*°?$",
        min_value=0.0,
        max_value=360.0,
        required=False
    ),
    
    # AIS Status
    "ais_on_off": FieldDefinition(
        name="ais_on_off",
        display_name="AIS ON/OFF",
        data_type="text",
        validation_regex=r"^(ON|OFF|AIS ON|AIS OFF)$",
        required=True
    ),
    
    # Depth
    "depth": FieldDefinition(
        name="depth",
        display_name="DEPTH",
        data_type="numeric",
        unit="meters",
        validation_regex=r"^\d{1,4}\.?\d*\s*[mM]?$",
        min_value=0.0,
        max_value=9999.0,
        required=False
    )
}

# System Configuration
class SystemConfig:
    """Central configuration for the extraction system."""
    
    # API Configuration
    AI_VISION_PROVIDERS = {
        "primary": "claude",  # Primary AI vision provider
        "fallback": "gpt4v",  # Fallback provider
        "tertiary": "gemini"  # Third option
    }
    
    # Performance Settings
    MAX_RETRIES = 3
    CONFIDENCE_THRESHOLD_GLOBAL = 0.85
    REQUIRE_REVIEW_THRESHOLD = 0.7
    
    # OCR Settings
    OCR_ENGINES = {
        "primary": "tesseract",
        "secondary": "easyocr",
        "specialized": "paddleocr"  # Good for maritime displays
    }
    
    # Processing Settings
    PARALLEL_WORKERS = 4
    BATCH_SIZE = 10
    
    # Image Preprocessing
    IMAGE_ENHANCEMENT = {
        "contrast_factor": 2.0,
        "brightness_delta": 10,
        "denoise_strength": 5,
        "sharpen_kernel_size": 3
    }
    
    # Output Settings
    OUTPUT_FORMAT = "json"  # json, csv, database
    SAVE_INTERMEDIATE_RESULTS = True
    SAVE_CONFIDENCE_MAPS = True
    
    # Validation Rules
    VALIDATION_RULES = {
        "cross_field_validation": True,  # e.g., COG should be close to heading when moving
        "temporal_validation": True,     # Check against previous/next frames
        "range_validation": True,        # Ensure values are within expected ranges
    }

# Radar Type Detection Configuration
RADAR_TYPE_FEATURES = {
    RadarType.FURUNO_CLASSIC: {
        "dominant_colors": [(60, 255, 255)],  # Green in HSV
        "text_color": "green",
        "background": "black",
        "layout": "circular_display_with_peripherals",
        "identifying_text": ["FURUNO", "AUTO RAIN"]
    },
    RadarType.MODERN_DARK: {
        "dominant_colors": [(0, 0, 50)],  # Dark gray
        "text_color": "white",
        "background": "dark_gray",
        "layout": "panel_based",
        "identifying_text": ["North Up", "REL Motion", "Radar"]
    }
}

# Extraction Pipeline Configuration
EXTRACTION_PIPELINE = [
    {
        "stage": 1,
        "name": "Image Preprocessing",
        "steps": [
            "radar_type_detection",
            "image_enhancement",
            "region_identification",
            "text_area_isolation"
        ]
    },
    {
        "stage": 2,
        "name": "Primary Extraction",
        "steps": [
            "ai_vision_extraction",
            "confidence_scoring",
            "initial_validation"
        ]
    },
    {
        "stage": 3,
        "name": "Secondary Extraction",
        "steps": [
            "identify_missing_fields",
            "targeted_ocr_extraction",
            "template_matching",
            "pattern_based_extraction"
        ]
    },
    {
        "stage": 4,
        "name": "Validation & Correction",
        "steps": [
            "cross_field_validation",
            "range_validation",
            "format_standardization",
            "confidence_adjustment"
        ]
    },
    {
        "stage": 5,
        "name": "Quality Assurance",
        "steps": [
            "overall_confidence_calculation",
            "review_flag_setting",
            "result_packaging",
            "audit_log_creation"
        ]
    }
]

# Initialize system configuration
def initialize_system():
    """Initialize the extraction system with all configurations."""
    config = {
        "version": "2.0",
        "fields": RADAR_FIELDS,
        "radar_types": RADAR_TYPE_FEATURES,
        "pipeline": EXTRACTION_PIPELINE,
        "system_config": SystemConfig.__dict__
    }
    
    # Save configuration
    with open("radar_extraction_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    print("System Configuration Initialized!")
    print(f"Total Fields Defined: {len(RADAR_FIELDS)}")
    print(f"Radar Types Supported: {len(RADAR_TYPE_FEATURES)}")
    print(f"Pipeline Stages: {len(EXTRACTION_PIPELINE)}")
    
    return config

if __name__ == "__main__":
    config = initialize_system()