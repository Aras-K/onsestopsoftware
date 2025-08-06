# Hybrid Radar Data Extraction System - Database & Result Management
# This implements comprehensive database storage, analytics, and review management

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import logging
from contextlib import contextmanager
import hashlib
import shutil
from enum import Enum

# Import our previous modules
from radar_extraction_architecture import (
    ExtractionMethod, RadarType, ExtractionResult, 
    RadarImageAnalysis, RADAR_FIELDS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewStatus(Enum):
    """Status for manual review process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    CORRECTED = "corrected"

class ExtractionStatus(Enum):
    """Overall extraction status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    REPROCESSING = "reprocessing"

@dataclass
class ReviewRecord:
    """Record for manual review actions."""
    review_id: int
    extraction_id: int
    reviewer_name: str
    review_status: ReviewStatus
    review_notes: Optional[str]
    corrections: Optional[Dict[str, Any]]
    review_timestamp: datetime

class DatabaseManager:
    """Manages all database operations for the radar extraction system."""
    
    def __init__(self, db_path: str = "radar_extraction_system.db"):
        """Initialize database connection and create tables."""
        self.db_path = db_path
        self.init_database()
        logger.info(f"Database initialized at: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def init_database(self):
        """Create all necessary tables for the extraction system."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Main extraction results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extractions (
                    extraction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    radar_type TEXT NOT NULL,
                    extraction_status TEXT NOT NULL,
                    overall_confidence REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    requires_review BOOLEAN NOT NULL,
                    extraction_timestamp TIMESTAMP NOT NULL,
                    image_path TEXT,
                    metadata TEXT,
                    UNIQUE(file_hash)
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_extraction_status 
                ON extractions(extraction_status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_requires_review 
                ON extractions(requires_review)
            """)
            
            # Extracted fields table (normalized)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extracted_fields (
                    field_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    extraction_id INTEGER NOT NULL,
                    field_name TEXT NOT NULL,
                    field_value TEXT,
                    confidence REAL NOT NULL,
                    extraction_method TEXT NOT NULL,
                    raw_text TEXT,
                    is_valid BOOLEAN NOT NULL,
                    validation_error TEXT,
                    processing_time REAL,
                    FOREIGN KEY (extraction_id) REFERENCES extractions(extraction_id),
                    UNIQUE(extraction_id, field_name)
                )
            """)
            
            # Review queue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    extraction_id INTEGER NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 5,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    assigned_to TEXT,
                    created_timestamp TIMESTAMP NOT NULL,
                    updated_timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (extraction_id) REFERENCES extractions(extraction_id)
                )
            """)
            
            # Review history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_history (
                    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    extraction_id INTEGER NOT NULL,
                    reviewer_name TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    review_notes TEXT,
                    corrections TEXT,
                    review_timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (extraction_id) REFERENCES extractions(extraction_id)
                )
            """)
            
            # Field corrections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_corrections (
                    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_id INTEGER NOT NULL,
                    original_value TEXT,
                    corrected_value TEXT NOT NULL,
                    correction_reason TEXT,
                    corrected_by TEXT NOT NULL,
                    correction_timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (field_id) REFERENCES extracted_fields(field_id)
                )
            """)
            
            # Processing statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_stats (
                    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_processed INTEGER DEFAULT 0,
                    successful_extractions INTEGER DEFAULT 0,
                    partial_extractions INTEGER DEFAULT 0,
                    failed_extractions INTEGER DEFAULT 0,
                    total_fields_extracted INTEGER DEFAULT 0,
                    average_confidence REAL,
                    average_processing_time REAL,
                    reviews_completed INTEGER DEFAULT 0,
                    UNIQUE(date)
                )
            """)
            
            # Confidence thresholds configuration
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS confidence_thresholds (
                    threshold_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_name TEXT NOT NULL,
                    min_confidence REAL NOT NULL DEFAULT 0.7,
                    review_required_below REAL NOT NULL DEFAULT 0.8,
                    auto_approve_above REAL NOT NULL DEFAULT 0.95,
                    updated_timestamp TIMESTAMP NOT NULL,
                    UNIQUE(field_name)
                )
            """)
            
            # Initialize default confidence thresholds
            self._init_confidence_thresholds(cursor)
            
            logger.info("Database tables created successfully")
    
    def _init_confidence_thresholds(self, cursor):
        """Initialize default confidence thresholds for each field."""
        for field_name, field_def in RADAR_FIELDS.items():
            cursor.execute("""
                INSERT OR IGNORE INTO confidence_thresholds 
                (field_name, min_confidence, review_required_below, auto_approve_above, updated_timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                field_name,
                field_def.confidence_threshold,
                field_def.confidence_threshold + 0.1,
                0.95,
                datetime.now()
            ))
    def get_recent_extractions(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent extractions from the database.
        
        Args:
            limit: Maximum number of extractions to return
            
        Returns:
            List of extraction dictionaries with basic information
        """
        results = []
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    extraction_id,
                    filename,
                    radar_type,
                    extraction_status,
                    overall_confidence,
                    processing_time,
                    requires_review,
                    extraction_timestamp
                FROM extractions
                ORDER BY extraction_timestamp DESC
                LIMIT ?
            """, (limit,))
            
            # Fetch and convert each row inside the connection context
            for row in cursor.fetchall():
                results.append({
                    'extraction_id': row['extraction_id'],
                    'filename': row['filename'],
                    'radar_type': row['radar_type'],
                    'extraction_status': row['extraction_status'],
                    'overall_confidence': row['overall_confidence'],
                    'processing_time': row['processing_time'],
                    'requires_review': row['requires_review'],
                    'extraction_timestamp': row['extraction_timestamp']
                })
        
        return results
    
    def save_extraction_result(self, analysis: RadarImageAnalysis, 
                             image_path: str = None) -> int:
        """
        Save extraction results to database.
        Returns: extraction_id
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Calculate file hash for duplicate detection
            file_hash = self._calculate_file_hash(image_path) if image_path else ""
            
            # Determine extraction status
            required_fields = [f for f in RADAR_FIELDS.values() if f.required]
            extracted_required = sum(
                1 for f in required_fields 
                if f.name in analysis.extraction_results 
                and analysis.extraction_results[f.name].value is not None
            )
            
            if extracted_required == len(required_fields):
                status = ExtractionStatus.SUCCESS
            elif extracted_required > 0:
                status = ExtractionStatus.PARTIAL
            else:
                status = ExtractionStatus.FAILED
            
            # Insert main extraction record
            cursor.execute("""
                INSERT OR REPLACE INTO extractions (
                    filename, file_hash, radar_type, extraction_status,
                    overall_confidence, processing_time, requires_review,
                    extraction_timestamp, image_path, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.filename,
                file_hash,
                analysis.radar_type.value,
                status.value,
                analysis.overall_confidence,
                analysis.processing_time,
                analysis.requires_review,
                datetime.now(),
                image_path,
                json.dumps(analysis.metadata)
            ))
            
            extraction_id = cursor.lastrowid
            
            # Insert extracted fields
            for field_name, result in analysis.extraction_results.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO extracted_fields (
                        extraction_id, field_name, field_value, confidence,
                        extraction_method, raw_text, is_valid, validation_error,
                        processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    extraction_id,
                    field_name,
                    str(result.value) if result.value is not None else None,
                    result.confidence,
                    result.method_used.value,
                    result.raw_text,
                    result.error is None,
                    result.error,
                    result.processing_time
                ))
            
            # Add to review queue if needed
            if analysis.requires_review:
                priority = self._calculate_review_priority(analysis)
                cursor.execute("""
                    INSERT INTO review_queue (
                        extraction_id, priority, review_status,
                        created_timestamp, updated_timestamp
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    extraction_id,
                    priority,
                    ReviewStatus.PENDING.value,
                    datetime.now(),
                    datetime.now()
                ))
            
            # Update daily statistics
            self._update_daily_stats(cursor, analysis)
            
            logger.info(f"Saved extraction result: ID={extraction_id}, "
                       f"Status={status.value}, Confidence={analysis.overall_confidence:.2f}")
            
            return extraction_id
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for duplicate detection."""
        if not file_path or not os.path.exists(file_path):
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _calculate_review_priority(self, analysis: RadarImageAnalysis) -> int:
        """
        Calculate review priority (1-10, 1 being highest priority).
        """
        priority = 5  # Default medium priority
        
        # Higher priority for lower confidence
        if analysis.overall_confidence < 0.5:
            priority = 1
        elif analysis.overall_confidence < 0.7:
            priority = 2
        elif analysis.overall_confidence < 0.8:
            priority = 3
        
        # Higher priority if critical fields are missing
        critical_fields = ['heading', 'speed', 'position']
        missing_critical = sum(
            1 for field in critical_fields
            if field not in analysis.extraction_results
            or analysis.extraction_results[field].value is None
        )
        if missing_critical > 0:
            priority = min(priority - missing_critical, 1)
        
        return priority
    
    def _update_daily_stats(self, cursor, analysis: RadarImageAnalysis):
        """Update daily processing statistics."""
        today = datetime.now().date()
        
        # Get current stats
        cursor.execute("""
            SELECT * FROM processing_stats WHERE date = ?
        """, (today,))
        
        row = cursor.fetchone()
        
        if row:
            # Update existing stats
            total_fields = sum(
                1 for r in analysis.extraction_results.values() 
                if r.value is not None
            )
            
            cursor.execute("""
                UPDATE processing_stats SET
                    total_processed = total_processed + 1,
                    successful_extractions = successful_extractions + ?,
                    partial_extractions = partial_extractions + ?,
                    failed_extractions = failed_extractions + ?,
                    total_fields_extracted = total_fields_extracted + ?,
                    average_confidence = 
                        (average_confidence * total_processed + ?) / (total_processed + 1),
                    average_processing_time = 
                        (average_processing_time * total_processed + ?) / (total_processed + 1)
                WHERE date = ?
            """, (
                1 if analysis.overall_confidence > 0.8 else 0,
                1 if 0.3 < analysis.overall_confidence <= 0.8 else 0,
                1 if analysis.overall_confidence <= 0.3 else 0,
                total_fields,
                analysis.overall_confidence,
                analysis.processing_time,
                today
            ))
        else:
            # Insert new stats
            total_fields = sum(
                1 for r in analysis.extraction_results.values() 
                if r.value is not None
            )
            
            cursor.execute("""
                INSERT INTO processing_stats (
                    date, total_processed, successful_extractions,
                    partial_extractions, failed_extractions,
                    total_fields_extracted, average_confidence,
                    average_processing_time
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?)
            """, (
                today,
                1 if analysis.overall_confidence > 0.8 else 0,
                1 if 0.3 < analysis.overall_confidence <= 0.8 else 0,
                1 if analysis.overall_confidence <= 0.3 else 0,
                total_fields,
                analysis.overall_confidence,
                analysis.processing_time
            ))
    
    def get_review_queue(self, limit: int = 50, 
                        status: ReviewStatus = ReviewStatus.PENDING) -> List[Dict]:
        """Get items from review queue ordered by priority."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    rq.*,
                    e.filename,
                    e.radar_type,
                    e.overall_confidence,
                    e.extraction_timestamp
                FROM review_queue rq
                JOIN extractions e ON rq.extraction_id = e.extraction_id
                WHERE rq.review_status = ?
                ORDER BY rq.priority ASC, rq.created_timestamp ASC
                LIMIT ?
            """, (status.value, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def submit_review(self, extraction_id: int, reviewer_name: str,
                     status: ReviewStatus, notes: str = None,
                     corrections: Dict[str, Any] = None) -> bool:
        """Submit a review for an extraction."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert review history
            cursor.execute("""
                INSERT INTO review_history (
                    extraction_id, reviewer_name, review_status,
                    review_notes, corrections, review_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                extraction_id,
                reviewer_name,
                status.value,
                notes,
                json.dumps(corrections) if corrections else None,
                datetime.now()
            ))
            
            # Update review queue
            cursor.execute("""
                UPDATE review_queue 
                SET review_status = ?, updated_timestamp = ?
                WHERE extraction_id = ?
            """, (status.value, datetime.now(), extraction_id))
            
            # Apply corrections if provided
            if corrections and status == ReviewStatus.CORRECTED:
                for field_name, corrected_value in corrections.items():
                    # Get field_id
                    cursor.execute("""
                        SELECT field_id, field_value 
                        FROM extracted_fields 
                        WHERE extraction_id = ? AND field_name = ?
                    """, (extraction_id, field_name))
                    
                    row = cursor.fetchone()
                    if row:
                        field_id, original_value = row
                        
                        # Insert correction record
                        cursor.execute("""
                            INSERT INTO field_corrections (
                                field_id, original_value, corrected_value,
                                correction_reason, corrected_by, correction_timestamp
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            field_id,
                            original_value,
                            str(corrected_value),
                            notes,
                            reviewer_name,
                            datetime.now()
                        ))
                        
                        # Update extracted field
                        cursor.execute("""
                            UPDATE extracted_fields 
                            SET field_value = ?, confidence = 1.0, is_valid = 1
                            WHERE field_id = ?
                        """, (str(corrected_value), field_id))
            
            # Update extraction status if approved
            if status == ReviewStatus.APPROVED:
                cursor.execute("""
                    UPDATE extractions 
                    SET requires_review = 0 
                    WHERE extraction_id = ?
                """, (extraction_id,))
            
            # Update daily stats
            cursor.execute("""
                UPDATE processing_stats 
                SET reviews_completed = reviews_completed + 1 
                WHERE date = ?
            """, (datetime.now().date(),))
            
            logger.info(f"Review submitted: extraction_id={extraction_id}, "
                       f"status={status.value}, reviewer={reviewer_name}")
            
            return True
    
    def get_extraction_details(self, extraction_id: int) -> Dict:
        """Get complete details for an extraction including all fields."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get main extraction info
            cursor.execute("""
                SELECT * FROM extractions WHERE extraction_id = ?
            """, (extraction_id,))
            
            extraction = dict(cursor.fetchone())
            
            # Get extracted fields
            cursor.execute("""
                SELECT * FROM extracted_fields WHERE extraction_id = ?
                ORDER BY field_name
            """, (extraction_id,))
            
            extraction['fields'] = [dict(row) for row in cursor.fetchall()]
            
            # Get review history
            cursor.execute("""
                SELECT * FROM review_history WHERE extraction_id = ?
                ORDER BY review_timestamp DESC
            """, (extraction_id,))
            
            extraction['reviews'] = [dict(row) for row in cursor.fetchall()]
            
            return extraction
    
    def get_statistics(self, start_date: datetime = None, 
                      end_date: datetime = None) -> Dict:
        """Get processing statistics for a date range."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # Overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_images,
                    AVG(overall_confidence) as avg_confidence,
                    AVG(processing_time) as avg_processing_time,
                    SUM(CASE WHEN extraction_status = 'success' THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN extraction_status = 'partial' THEN 1 ELSE 0 END) as partial,
                    SUM(CASE WHEN extraction_status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN requires_review = 1 THEN 1 ELSE 0 END) as pending_review
                FROM extractions
                WHERE extraction_timestamp BETWEEN ? AND ?
            """, (start_date, end_date))
            
            overall = dict(cursor.fetchone())
            
            # Field-level statistics
            cursor.execute("""
                SELECT 
                    field_name,
                    COUNT(*) as extraction_count,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN is_valid = 1 THEN 1 ELSE 0 END) as valid_count,
                    COUNT(DISTINCT extraction_method) as methods_used
                FROM extracted_fields ef
                JOIN extractions e ON ef.extraction_id = e.extraction_id
                WHERE e.extraction_timestamp BETWEEN ? AND ?
                GROUP BY field_name
                ORDER BY avg_confidence DESC
            """, (start_date, end_date))
            
            field_stats = [dict(row) for row in cursor.fetchall()]
            
            # Method effectiveness
            cursor.execute("""
                SELECT 
                    extraction_method,
                    COUNT(*) as usage_count,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN is_valid = 1 THEN 1 ELSE 0 END) as success_count
                FROM extracted_fields ef
                JOIN extractions e ON ef.extraction_id = e.extraction_id
                WHERE e.extraction_timestamp BETWEEN ? AND ?
                GROUP BY extraction_method
            """, (start_date, end_date))
            
            method_stats = [dict(row) for row in cursor.fetchall()]
            
            # Radar type distribution
            cursor.execute("""
                SELECT 
                    radar_type,
                    COUNT(*) as count,
                    AVG(overall_confidence) as avg_confidence
                FROM extractions
                WHERE extraction_timestamp BETWEEN ? AND ?
                GROUP BY radar_type
            """, (start_date, end_date))
            
            radar_type_stats = [dict(row) for row in cursor.fetchall()]
            
            return {
                'overall': overall,
                'field_statistics': field_stats,
                'method_effectiveness': method_stats,
                'radar_type_distribution': radar_type_stats,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            }
    
    def get_failed_extractions(self, limit: int = 100) -> List[Dict]:
        """Get failed or low-confidence extractions for reprocessing."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    extraction_id,
                    filename,
                    file_hash,
                    radar_type,
                    overall_confidence,
                    extraction_timestamp,
                    image_path
                FROM extractions
                WHERE extraction_status = 'failed' 
                   OR extraction_status = 'partial'
                   OR overall_confidence < 0.7
                ORDER BY overall_confidence ASC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def mark_for_reprocessing(self, extraction_ids: List[int]):
        """Mark extractions for reprocessing."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            placeholders = ','.join('?' * len(extraction_ids))
            cursor.execute(f"""
                UPDATE extractions 
                SET extraction_status = 'reprocessing' 
                WHERE extraction_id IN ({placeholders})
            """, extraction_ids)
            
            logger.info(f"Marked {len(extraction_ids)} extractions for reprocessing")
    
    def export_to_csv(self, output_path: str, start_date: datetime = None,
                     end_date: datetime = None):
        """Export extraction results to CSV for analysis."""
        with self.get_connection() as conn:
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # Create a denormalized view for export
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
                    ef.extraction_method,
                    ef.is_valid
                FROM extractions e
                LEFT JOIN extracted_fields ef ON e.extraction_id = ef.extraction_id
                WHERE e.extraction_timestamp BETWEEN ? AND ?
                ORDER BY e.extraction_id, ef.field_name
            """
            
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            
            # Pivot to have one row per extraction with fields as columns
            pivot_df = df.pivot_table(
                index=['extraction_id', 'filename', 'radar_type', 
                       'extraction_status', 'overall_confidence', 
                       'processing_time', 'extraction_timestamp'],
                columns='field_name',
                values='field_value',
                aggfunc='first'
            ).reset_index()
            
            # Save to CSV
            pivot_df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(pivot_df)} extractions to {output_path}")

class ResultManager:
    """High-level manager for extraction results and workflows."""
    
    def __init__(self, db_manager: DatabaseManager, 
                 output_dir: str = "extraction_results"):
        """Initialize result manager."""
        self.db = db_manager
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'successful': os.path.join(output_dir, 'successful'),
            'review': os.path.join(output_dir, 'needs_review'),
            'failed': os.path.join(output_dir, 'failed'),
            'reports': os.path.join(output_dir, 'reports'),
            'exports': os.path.join(output_dir, 'exports')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def process_extraction_result(self, analysis: RadarImageAnalysis, 
                                image_path: str) -> int:
        """
        Process and save extraction result with appropriate filing.
        Returns: extraction_id
        """
        # Save to database
        extraction_id = self.db.save_extraction_result(analysis, image_path)
        
        # Save detailed JSON result
        self._save_json_result(analysis, extraction_id)
        
        # Copy image to appropriate directory
        if analysis.overall_confidence > 0.8 and not analysis.requires_review:
            dest_dir = self.dirs['successful']
        elif analysis.requires_review:
            dest_dir = self.dirs['review']
        else:
            dest_dir = self.dirs['failed']
        
        if image_path and os.path.exists(image_path):
            dest_path = os.path.join(dest_dir, os.path.basename(image_path))
            shutil.copy2(image_path, dest_path)
        
        return extraction_id
    
    def _save_json_result(self, analysis: RadarImageAnalysis, extraction_id: int):
        """Save detailed JSON result file."""
        result_data = {
            'extraction_id': extraction_id,
            'timestamp': datetime.now().isoformat(),
            'analysis': asdict(analysis),
            'extracted_values': {}
        }
        
        # Simplify extracted values for quick reference
        for field_name, result in analysis.extraction_results.items():
            result_data['extracted_values'][field_name] = {
                'value': result.value,
                'confidence': result.confidence,
                'method': result.method_used.value
            }
        
        # Save to appropriate directory
        if analysis.overall_confidence > 0.8 and not analysis.requires_review:
            json_dir = self.dirs['successful']
        elif analysis.requires_review:
            json_dir = self.dirs['review']
        else:
            json_dir = self.dirs['failed']
        
        json_path = os.path.join(
            json_dir, 
            f"{analysis.filename}_result_{extraction_id}.json"
        )
        
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
    
    def generate_daily_report(self, date: datetime = None) -> str:
        """Generate daily processing report."""
        if not date:
            date = datetime.now()
        
        report_date = date.date()
        stats = self.db.get_statistics(
            start_date=datetime.combine(report_date, datetime.min.time()),
            end_date=datetime.combine(report_date, datetime.max.time())
        )
        
        report_lines = [
            f"# Daily Extraction Report - {report_date}",
            f"\n## Overall Statistics",
            f"- Total Images Processed: {stats['overall']['total_images']}",
            f"- Average Confidence: {stats['overall']['avg_confidence']:.2%}",
            f"- Average Processing Time: {stats['overall']['avg_processing_time']:.2f}s",
            f"- Successful: {stats['overall']['successful']}",
            f"- Partial: {stats['overall']['partial']}",
            f"- Failed: {stats['overall']['failed']}",
            f"- Pending Review: {stats['overall']['pending_review']}",
            f"\n## Field Performance"
        ]
        
        # Add field statistics
        for field_stat in stats['field_statistics'][:10]:  # Top 10 fields
            report_lines.append(
                f"- {field_stat['field_name']}: "
                f"{field_stat['avg_confidence']:.2%} confidence, "
                f"{field_stat['valid_count']}/{field_stat['extraction_count']} valid"
            )
        
        report_lines.extend([
            f"\n## Method Effectiveness"
        ])
        
        # Add method statistics
        for method_stat in stats['method_effectiveness']:
            success_rate = (method_stat['success_count'] / method_stat['usage_count'] 
                          if method_stat['usage_count'] > 0 else 0)
            report_lines.append(
                f"- {method_stat['extraction_method']}: "
                f"{method_stat['usage_count']} uses, "
                f"{method_stat['avg_confidence']:.2%} avg confidence, "
                f"{success_rate:.2%} success rate"
            )
        
        # Save report
        report_content = '\n'.join(report_lines)
        report_path = os.path.join(
            self.dirs['reports'],
            f"daily_report_{report_date}.md"
        )
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Daily report generated: {report_path}")
        return report_path
    
    def export_for_analysis(self, start_date: datetime = None, 
                          end_date: datetime = None) -> str:
        """Export data for external analysis."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(
            self.dirs['exports'],
            f"radar_extractions_{timestamp}.csv"
        )
        
        self.db.export_to_csv(export_path, start_date, end_date)
        
        return export_path

# Example usage and testing
def test_database_system():
    """Test the database system with sample data."""
    # Initialize database
    db = DatabaseManager("test_radar_extraction.db")
    result_manager = ResultManager(db)
    
    # Create sample extraction result
    from radar_extraction_architecture import ExtractionMethod
    
    sample_analysis = RadarImageAnalysis(
        filename="test_radar_001.png",
        radar_type=RadarType.FURUNO_CLASSIC,
        extraction_results={
            "heading": ExtractionResult(
                field_name="heading",
                value=327.5,
                confidence=0.95,
                method_used=ExtractionMethod.AI_VISION,
                raw_text="327.5°"
            ),
            "speed": ExtractionResult(
                field_name="speed",
                value=9.7,
                confidence=0.88,
                method_used=ExtractionMethod.AI_VISION,
                raw_text="9.7kn"
            )
        },
        overall_confidence=0.91,
        processing_time=3.45,
        requires_review=False,
        validation_errors=[],
        metadata={"test": True}
    )
    
    # Save result
    extraction_id = result_manager.process_extraction_result(
        sample_analysis, 
        "test_image.png"
    )
    
    print(f"Saved extraction with ID: {extraction_id}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")
    
    # Generate report
    report_path = result_manager.generate_daily_report()
    print(f"\nGenerated report: {report_path}")

if __name__ == "__main__":
    test_database_system()