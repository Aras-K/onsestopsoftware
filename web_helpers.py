# web_helpers.py - Helper functions to bridge web interface with existing code

import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import base64
from PIL import Image
import io

# Import your existing components
from radar_extraction_architecture import RadarType, RADAR_FIELDS
from radar_extraction_engine import HybridRadarExtractor, RadarImageAnalysis
from radar_database_management import DatabaseManager, ResultManager, ReviewStatus

# Import the image storage manager
try:
    from image_storage import ImageStorageManager
except ImportError:
    print("Warning: ImageStorageManager not found. Image storage will be disabled.")
    ImageStorageManager = None

class WebProcessingHelper:
    """Helper class to handle processing for web interface."""
    
    def __init__(self, api_keys: Dict[str, str], db_path: str = "radar_extraction_system.db"):
        """Initialize the processing helper."""
        self.api_keys = api_keys
        self.db_manager = DatabaseManager(db_path)
        self.result_manager = ResultManager(self.db_manager)
        self.extractor = HybridRadarExtractor(api_keys)
        
        # Initialize image storage manager
        if ImageStorageManager:
            self.image_storage = ImageStorageManager()
        else:
            self.image_storage = None
    
    async def process_single_image_async(self, image_path: str) -> Dict[str, Any]:
        """Process a single image and return results for web display."""
        try:
            # Process the image
            analysis = await self.extractor.extract_image(image_path)
            
            # Save to database
            extraction_id = self.result_manager.process_extraction_result(
                analysis, image_path
            )
            
            # Store the image for later review if storage is available
            if self.image_storage and extraction_id:
                try:
                    self.image_storage.store_image(image_path, extraction_id)
                except Exception as e:
                    print(f"Warning: Failed to store image for extraction {extraction_id}: {e}")
            
            # Prepare web-friendly result
            result = {
                'success': True,
                'extraction_id': extraction_id,
                'filename': analysis.filename,
                'radar_type': analysis.radar_type.value,
                'overall_confidence': analysis.overall_confidence,
                'processing_time': analysis.processing_time,
                'requires_review': analysis.requires_review,
                'fields_extracted': {},
                'field_count': 0,
                'validation_warnings': analysis.validation_errors,
                'status': self._get_status(analysis.overall_confidence)
            }
            
            # Add extracted fields
            for field_name, extraction in analysis.extraction_results.items():
                if extraction.value is not None:
                    result['fields_extracted'][field_name] = {
                        'value': extraction.value,
                        'confidence': extraction.confidence,
                        'method': extraction.method_used.value
                    }
                    result['field_count'] += 1
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': os.path.basename(image_path),
                'status': 'error'
            }
    
    def _get_status(self, confidence: float) -> str:
        """Get status based on confidence level."""
        if confidence > 0.8:
            return 'success'
        elif confidence > 0.5:
            return 'review'
        else:
            return 'failed'
    
    def get_review_items_for_web(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get review items formatted for web display."""
        queue = self.db_manager.get_review_queue(limit=limit)
        
        web_items = []
        for item in queue:
            # Get full extraction details
            extraction = self.db_manager.get_extraction_details(item['extraction_id'])
            
            # Format for web
            web_item = {
                'extraction_id': item['extraction_id'],
                'filename': extraction['filename'],
                'radar_type': extraction['radar_type'],
                'overall_confidence': extraction['overall_confidence'],
                'timestamp': extraction['extraction_timestamp'],
                'priority': item['priority'],
                'fields': {}
            }
            
            # Organize fields by confidence
            for field in extraction['fields']:
                web_item['fields'][field['field_name']] = {
                    'value': field['field_value'],
                    'confidence': field['confidence'],
                    'is_valid': field['is_valid'],
                    'method': field['extraction_method']
                }
            
            web_items.append(web_item)
        
        return web_items
    
    def submit_review_from_web(self, extraction_id: int, reviewer: str, 
                              action: str, corrections: Dict[str, Any] = None,
                              notes: str = "") -> bool:
        """Submit review from web interface."""
        status_map = {
            'approve': ReviewStatus.APPROVED,
            'reject': ReviewStatus.REJECTED,
            'correct': ReviewStatus.CORRECTED
        }
        
        review_status = status_map.get(action, ReviewStatus.APPROVED)
        
        return self.db_manager.submit_review(
            extraction_id=extraction_id,
            reviewer_name=reviewer,
            status=review_status,
            notes=notes,
            corrections=corrections
        )
    
    def get_image_for_display(self, extraction_id: int) -> Optional[bytes]:
        """Get image data for display in review interface."""
        if not self.image_storage:
            return None
        
        # Try to get from image storage
        image_data = self.image_storage.get_image_data(extraction_id)
        if image_data:
            return image_data
        
        # If no image storage or image not found, return None
        return None
    
    def get_image_as_base64(self, extraction_id: int) -> Optional[str]:
        """Get image as base64 string for web display."""
        image_data = self.get_image_for_display(extraction_id)
        if image_data:
            return base64.b64encode(image_data).decode('utf-8')
        return None
    
    def get_analytics_data(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics data for web display."""
        from datetime import timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stats = self.db_manager.get_statistics(start_date, end_date)
        
        # Prepare data for charts
        analytics = {
            'overall_stats': stats['overall'],
            'field_performance': [],
            'daily_trend': [],
            'radar_type_distribution': stats.get('radar_type_distribution', []),
            'method_effectiveness': stats.get('method_effectiveness', [])
        }
        
        # Field performance data
        for field_stat in stats.get('field_statistics', []):
            analytics['field_performance'].append({
                'field': field_stat['field_name'],
                'success_rate': (field_stat['valid_count'] / field_stat['extraction_count'] * 100 
                               if field_stat['extraction_count'] > 0 else 0),
                'avg_confidence': field_stat['avg_confidence'] * 100 if field_stat['avg_confidence'] else 0,
                'count': field_stat['extraction_count']
            })
        
        # Daily trend data
        for i in range(days):
            date = start_date + timedelta(days=i)
            day_stats = self.db_manager.get_statistics(
                start_date=date.replace(hour=0, minute=0),
                end_date=date.replace(hour=23, minute=59)
            )
            
            if day_stats and day_stats['overall']:
                analytics['daily_trend'].append({
                    'date': date.strftime('%Y-%m-%d'),
                    'total': day_stats['overall']['total_images'] or 0,
                    'successful': day_stats['overall']['successful'] or 0,
                    'failed': day_stats['overall']['failed'] or 0
                })
        
        return analytics
    
    def export_data_for_web(self, format: str = 'csv', days: int = 30) -> Dict[str, Any]:
        """Export data and return file info for web download."""
        from datetime import timedelta
        import tempfile
        import pandas as pd
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}')
        
        try:
            if format == 'csv':
                self.db_manager.export_to_csv(temp_file.name, start_date, end_date)
                mime_type = 'text/csv'
            elif format == 'json':
                # Get data and export as JSON
                with self.db_manager.get_connection() as conn:
                    query = """
                        SELECT * FROM extractions e
                        LEFT JOIN extracted_fields ef ON e.extraction_id = ef.extraction_id
                        WHERE e.extraction_timestamp BETWEEN ? AND ?
                    """
                    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
                    df.to_json(temp_file.name, orient='records', indent=2)
                mime_type = 'application/json'
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Read file content
            with open(temp_file.name, 'rb') as f:
                content = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
            
            return {
                'success': True,
                'content': content,
                'filename': f'radar_export_{datetime.now():%Y%m%d_%H%M%S}.{format}',
                'mime_type': mime_type
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if self.image_storage:
            return self.image_storage.get_storage_stats()
        return {
            'total_images': 0,
            'valid_images': 0,
            'total_size_mb': 0,
            'storage_enabled': False
        }
    
    def cleanup_old_images(self, days: int = 7) -> int:
        """Clean up old images."""
        if self.image_storage:
            return self.image_storage.cleanup_old_images(days)
        return 0

# Singleton instance for web app
_web_helper = None

def get_web_helper(api_keys: Dict[str, str]) -> WebProcessingHelper:
    """Get or create web helper instance."""
    global _web_helper
    if _web_helper is None:
        _web_helper = WebProcessingHelper(api_keys)
    return _web_helper