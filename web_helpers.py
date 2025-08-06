# web_helpers.py - Helper functions to bridge web interface with existing code (Azure Compatible)

import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import base64
from PIL import Image
import io
import logging
import tempfile
import pandas as pd

# Import your existing components
from radar_extraction_architecture import RadarType, RADAR_FIELDS
from radar_extraction_engine import HybridRadarExtractor, RadarImageAnalysis
from radar_database_management import DatabaseManager, ResultManager, ReviewStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the image storage manager (optional)
try:
    from image_storage import ImageStorageManager
    IMAGE_STORAGE_AVAILABLE = True
except ImportError:
    logger.warning("ImageStorageManager not found. Image storage will be disabled.")
    ImageStorageManager = None
    IMAGE_STORAGE_AVAILABLE = False

class WebProcessingHelper:
    """Helper class to handle processing for web interface."""
    
    def __init__(self, api_keys: Dict[str, str] = None, db_connection_string: str = None):
        """
        Initialize the processing helper.
        
        Args:
            api_keys: Dictionary of API keys or None to use environment variables
            db_connection_string: PostgreSQL connection string or None to use environment variable
        """
        # Use environment variables if not provided
        if api_keys is None:
            api_keys = {
                'openai': os.environ.get('OPENAI_API_KEY'),
                'google': os.environ.get('GOOGLE_API_KEY'),
                'anthropic': os.environ.get('ANTHROPIC_API_KEY')
            }
        
        self.api_keys = api_keys
        
        # Initialize database with PostgreSQL
        self.db_manager = DatabaseManager(connection_string=db_connection_string)
        self.result_manager = ResultManager(self.db_manager)
        self.extractor = HybridRadarExtractor(api_keys)
        
        # Initialize image storage manager if available
        if IMAGE_STORAGE_AVAILABLE:
            try:
                # For Azure, you might want to use Azure Blob Storage
                storage_type = os.environ.get('IMAGE_STORAGE_TYPE', 'local')
                if storage_type == 'azure':
                    # Initialize with Azure Blob Storage connection
                    connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
                    self.image_storage = ImageStorageManager(storage_type='azure', 
                                                            connection_string=connection_string)
                else:
                    # Use local storage
                    self.image_storage = ImageStorageManager()
            except Exception as e:
                logger.warning(f"Failed to initialize image storage: {e}")
                self.image_storage = None
        else:
            self.image_storage = None
        
        logger.info("WebProcessingHelper initialized successfully")
    
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
                    logger.warning(f"Failed to store image for extraction {extraction_id}: {e}")
            
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
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'filename': os.path.basename(image_path),
                'status': 'error'
            }
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """Synchronous wrapper for process_single_image_async."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_single_image_async(image_path))
        finally:
            loop.close()
    
    async def process_batch_async(self, image_paths: List[str], 
                                 max_workers: int = 4) -> List[Dict[str, Any]]:
        """Process multiple images in batch."""
        results = []
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(image_paths), max_workers):
            batch = image_paths[i:i + max_workers]
            batch_tasks = [self.process_single_image_async(path) for path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for path, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results.append({
                        'success': False,
                        'error': str(result),
                        'filename': os.path.basename(path),
                        'status': 'error'
                    })
                else:
                    results.append(result)
        
        return results
    
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
        try:
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
                for field in extraction.get('fields', []):
                    web_item['fields'][field['field_name']] = {
                        'value': field['field_value'],
                        'confidence': field['confidence'],
                        'is_valid': field['is_valid'],
                        'method': field['extraction_method']
                    }
                
                # Add image availability flag
                web_item['has_image'] = self.image_storage is not None
                
                web_items.append(web_item)
            
            return web_items
            
        except Exception as e:
            logger.error(f"Error getting review items: {e}")
            return []
    
    def submit_review_from_web(self, extraction_id: int, reviewer: str, 
                              action: str, corrections: Dict[str, Any] = None,
                              notes: str = "") -> bool:
        """Submit review from web interface."""
        try:
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
        except Exception as e:
            logger.error(f"Error submitting review: {e}")
            return False
    
    def get_image_for_display(self, extraction_id: int) -> Optional[bytes]:
        """Get image data for display in review interface."""
        if not self.image_storage:
            return None
        
        try:
            # Try to get from image storage
            image_data = self.image_storage.get_image_data(extraction_id)
            return image_data
        except Exception as e:
            logger.warning(f"Failed to get image for extraction {extraction_id}: {e}")
            return None
    
    def get_image_as_base64(self, extraction_id: int) -> Optional[str]:
        """Get image as base64 string for web display."""
        try:
            image_data = self.get_image_for_display(extraction_id)
            if image_data:
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to encode image as base64: {e}")
        return None
    
    def get_analytics_data(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics data for web display."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            stats = self.db_manager.get_statistics(start_date, end_date)
            
            # Prepare data for charts
            analytics = {
                'overall_stats': stats.get('overall', {}),
                'field_performance': [],
                'daily_trend': [],
                'radar_type_distribution': stats.get('radar_type_distribution', []),
                'method_effectiveness': stats.get('method_effectiveness', [])
            }
            
            # Field performance data
            for field_stat in stats.get('field_statistics', []):
                if field_stat.get('extraction_count', 0) > 0:
                    analytics['field_performance'].append({
                        'field': field_stat['field_name'],
                        'success_rate': (field_stat.get('valid_count', 0) / 
                                       field_stat['extraction_count'] * 100),
                        'avg_confidence': (field_stat.get('avg_confidence', 0) * 100 
                                         if field_stat.get('avg_confidence') else 0),
                        'count': field_stat['extraction_count']
                    })
            
            # Sort by success rate
            analytics['field_performance'].sort(key=lambda x: x['success_rate'], reverse=True)
            
            # Get daily trend data (simplified for performance)
            # Instead of making 30 queries, aggregate in the database
            analytics['daily_trend'] = self._get_daily_trend_data(start_date, end_date)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return {
                'overall_stats': {},
                'field_performance': [],
                'daily_trend': [],
                'radar_type_distribution': [],
                'method_effectiveness': []
            }
    
    def _get_daily_trend_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get daily trend data efficiently."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        DATE(extraction_timestamp) as date,
                        COUNT(*) as total,
                        SUM(CASE WHEN extraction_status = 'success' THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN extraction_status = 'failed' THEN 1 ELSE 0 END) as failed,
                        AVG(overall_confidence) as avg_confidence
                    FROM extractions
                    WHERE extraction_timestamp BETWEEN %s AND %s
                    GROUP BY DATE(extraction_timestamp)
                    ORDER BY date
                """, (start_date, end_date))
                
                trend_data = []
                for row in cursor.fetchall():
                    trend_data.append({
                        'date': row[0].strftime('%Y-%m-%d') if row[0] else '',
                        'total': row[1] or 0,
                        'successful': row[2] or 0,
                        'failed': row[3] or 0,
                        'avg_confidence': (row[4] * 100) if row[4] else 0
                    })
                
                return trend_data
                
        except Exception as e:
            logger.error(f"Error getting daily trend data: {e}")
            return []
    
    def export_data_for_web(self, format: str = 'csv', days: int = 30) -> Dict[str, Any]:
        """Export data and return file info for web download."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}') as temp_file:
                temp_filename = temp_file.name
            
            try:
                if format == 'csv':
                    self.db_manager.export_to_csv(temp_filename, start_date, end_date)
                    mime_type = 'text/csv'
                    
                elif format == 'json':
                    # Get data and export as JSON
                    with self.db_manager.get_connection() as conn:
                        # Simpler query that works with both PostgreSQL and pandas
                        query = """
                            SELECT 
                                e.extraction_id,
                                e.filename,
                                e.radar_type,
                                e.extraction_status,
                                e.overall_confidence,
                                e.processing_time,
                                e.extraction_timestamp,
                                ef.field_name,
                                ef.field_value,
                                ef.confidence as field_confidence,
                                ef.extraction_method
                            FROM extractions e
                            LEFT JOIN extracted_fields ef ON e.extraction_id = ef.extraction_id
                            WHERE e.extraction_timestamp BETWEEN %s AND %s
                            ORDER BY e.extraction_id, ef.field_name
                        """
                        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
                        
                        # Convert to JSON
                        json_data = df.to_dict('records')
                        with open(temp_filename, 'w') as f:
                            json.dump(json_data, f, indent=2, default=str)
                    
                    mime_type = 'application/json'
                    
                elif format == 'excel':
                    # Export as Excel
                    self.db_manager.export_to_csv(temp_filename.replace('.excel', '.csv'), 
                                                 start_date, end_date)
                    df = pd.read_csv(temp_filename.replace('.excel', '.csv'))
                    df.to_excel(temp_filename, index=False)
                    os.unlink(temp_filename.replace('.excel', '.csv'))
                    mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                # Read file content
                with open(temp_filename, 'rb') as f:
                    content = f.read()
                
                # Clean up
                os.unlink(temp_filename)
                
                return {
                    'success': True,
                    'content': content,
                    'filename': f'radar_export_{datetime.now():%Y%m%d_%H%M%S}.{format}',
                    'mime_type': mime_type,
                    'size': len(content)
                }
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                raise e
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if self.image_storage:
            try:
                return self.image_storage.get_storage_stats()
            except Exception as e:
                logger.warning(f"Failed to get storage stats: {e}")
        
        return {
            'total_images': 0,
            'valid_images': 0,
            'total_size_mb': 0,
            'storage_enabled': False
        }
    
    def cleanup_old_images(self, days: int = 7) -> int:
        """Clean up old images."""
        if self.image_storage:
            try:
                return self.image_storage.cleanup_old_images(days)
            except Exception as e:
                logger.warning(f"Failed to cleanup images: {e}")
        return 0
    
    def get_recent_extractions(self, limit: int = 10) -> List[Dict]:
        """Get recent extractions for dashboard."""
        try:
            return self.db_manager.get_recent_extractions(limit=limit)
        except Exception as e:
            logger.error(f"Error getting recent extractions: {e}")
            return []
    
    def close(self):
        """Clean up resources."""
        try:
            if hasattr(self.db_manager, 'close'):
                self.db_manager.close()
            logger.info("WebProcessingHelper resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

# Singleton instance for web app
_web_helper = None

def get_web_helper(api_keys: Dict[str, str] = None) -> WebProcessingHelper:
    """Get or create web helper instance."""
    global _web_helper
    if _web_helper is None:
        _web_helper = WebProcessingHelper(api_keys)
    return _web_helper

def reset_web_helper():
    """Reset the web helper instance (useful for testing or reinitialization)."""
    global _web_helper
    if _web_helper:
        _web_helper.close()
    _web_helper = None