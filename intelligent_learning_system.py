# intelligent_learning_system.py
# Self-learning and retraining system for radar extraction (Azure Compatible)

import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
from collections import defaultdict
import re
import logging
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2 import pool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentLearningSystem:
    """Self-learning system that improves extraction accuracy over time."""
    
    def __init__(self, connection_string: str = None):
        """
        Initialize learning system with PostgreSQL connection.
        
        Args:
            connection_string: PostgreSQL connection string or None to use environment variable
        """
        # Get connection string from environment if not provided
        if connection_string is None:
            connection_string = os.environ.get('DATABASE_URL')
            if not connection_string:
                connection_string = self._build_connection_string()
        
        self.connection_string = connection_string
        
        # Initialize connection pool for better performance
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=self.connection_string
        )
        
        self.init_learning_tables()
        self.knowledge_base = self.load_knowledge_base()
        
        logger.info("Intelligent Learning System initialized with PostgreSQL")
    
    def _build_connection_string(self) -> str:
        """Build connection string from environment variables."""
        host = os.environ.get('POSTGRES_HOST', 'localhost')
        port = os.environ.get('POSTGRES_PORT', '5432')
        database = os.environ.get('POSTGRES_DB', 'radar_extraction')
        user = os.environ.get('POSTGRES_USER', 'postgres')
        password = os.environ.get('POSTGRES_PASSWORD', '')
        sslmode = os.environ.get('POSTGRES_SSLMODE', 'require')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode={sslmode}"
    
    def get_connection(self):
        """Get a connection from the pool."""
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return a connection to the pool."""
        self.connection_pool.putconn(conn)
    
    def init_learning_tables(self):
        """Initialize learning system tables in PostgreSQL."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Learning patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    pattern_id SERIAL PRIMARY KEY,
                    radar_type TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    successful_pattern JSONB,
                    failed_pattern JSONB,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    confidence_boost REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(radar_type, field_name)
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_learning_patterns_radar_field 
                ON learning_patterns(radar_type, field_name)
            """)
            
            # Prompt improvements table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_improvements (
                    improvement_id SERIAL PRIMARY KEY,
                    field_name TEXT NOT NULL,
                    original_prompt TEXT,
                    improved_prompt TEXT,
                    improvement_reason TEXT,
                    success_rate_before REAL,
                    success_rate_after REAL,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Field synonyms learned
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_synonyms (
                    synonym_id SERIAL PRIMARY KEY,
                    field_name TEXT NOT NULL,
                    synonym TEXT NOT NULL,
                    radar_type TEXT,
                    confidence REAL DEFAULT 1.0,
                    UNIQUE(field_name, synonym, radar_type)
                )
            """)
            
            # Performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_tracking (
                    tracking_id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    field_name TEXT NOT NULL,
                    radar_type TEXT,
                    attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    avg_confidence REAL,
                    UNIQUE(date, field_name, radar_type)
                )
            """)
            
            # Create performance index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_performance_tracking_date 
                ON performance_tracking(date DESC)
            """)
            
            conn.commit()
            logger.info("Learning tables initialized successfully")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error initializing learning tables: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def analyze_and_learn(self):
        """Analyze recent extractions and learn from them."""
        print("🧠 INTELLIGENT LEARNING SYSTEM - Analysis Started")
        print("="*70)
        
        try:
            # 1. Analyze success patterns
            self.analyze_success_patterns()
            
            # 2. Learn from failures
            self.learn_from_failures()
            
            # 3. Discover field synonyms
            self.discover_field_synonyms()
            
            # 4. Generate improved prompts
            self.generate_improved_prompts()
            
            # 5. Update performance metrics
            self.update_performance_metrics()
            
            # 6. Save updated knowledge base
            self.save_knowledge_base()
            
            print("\n✅ Learning complete! System intelligence updated.")
            
        except Exception as e:
            logger.error(f"Error during learning analysis: {e}")
            print(f"\n❌ Learning failed: {e}")
    
    def analyze_success_patterns(self):
        """Analyze what patterns lead to successful extractions."""
        conn = self.get_connection()
        
        print("\n📊 Analyzing Success Patterns...")
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get successful extractions
            query = """
                SELECT 
                    e.radar_type,
                    ef.field_name,
                    ef.field_value,
                    ef.raw_text,
                    ef.confidence
                FROM extracted_fields ef
                JOIN extractions e ON ef.extraction_id = e.extraction_id
                WHERE ef.field_value IS NOT NULL
                AND ef.confidence > 0.8
                AND e.extraction_timestamp >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY ef.confidence DESC
            """
            
            cursor.execute(query)
            
            patterns = defaultdict(lambda: defaultdict(list))
            
            for row in cursor.fetchall():
                radar_type = row['radar_type']
                field_name = row['field_name']
                field_value = row['field_value']
                raw_text = row['raw_text']
                confidence = row['confidence']
                
                # Extract patterns from successful values
                pattern = self.extract_value_pattern(field_name, field_value, raw_text)
                patterns[radar_type][field_name].append({
                    'pattern': pattern,
                    'confidence': confidence,
                    'example': field_value
                })
            
            # Save learned patterns
            for radar_type, fields in patterns.items():
                for field_name, field_patterns in fields.items():
                    if field_patterns:
                        best_pattern = max(field_patterns, key=lambda x: x['confidence'])
                        avg_confidence = sum(p['confidence'] for p in field_patterns) / len(field_patterns)
                        
                        cursor.execute("""
                            INSERT INTO learning_patterns
                            (radar_type, field_name, successful_pattern, success_count, confidence_boost, last_updated)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (radar_type, field_name)
                            DO UPDATE SET
                                successful_pattern = EXCLUDED.successful_pattern,
                                success_count = learning_patterns.success_count + EXCLUDED.success_count,
                                confidence_boost = EXCLUDED.confidence_boost,
                                last_updated = EXCLUDED.last_updated
                        """, (
                            radar_type,
                            field_name,
                            Json(best_pattern),
                            len(field_patterns),
                            avg_confidence - 0.5,
                            datetime.now()
                        ))
            
            conn.commit()
            print(f"✓ Learned patterns for {len(patterns)} radar types")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error analyzing success patterns: {e}")
        finally:
            self.return_connection(conn)
    
    def learn_from_failures(self):
        """Learn why certain fields fail to extract."""
        conn = self.get_connection()
        
        print("\n🔍 Learning from Failures...")
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Analyze failed extractions
            query = """
                SELECT 
                    e.radar_type,
                    ef.field_name,
                    ef.validation_error,
                    COUNT(*) as failure_count
                FROM extracted_fields ef
                JOIN extractions e ON ef.extraction_id = e.extraction_id
                WHERE (ef.field_value IS NULL OR ef.confidence < 0.5)
                AND e.extraction_timestamp >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY e.radar_type, ef.field_name, ef.validation_error
                ORDER BY failure_count DESC
            """
            
            cursor.execute(query)
            
            failure_patterns = defaultdict(lambda: defaultdict(list))
            
            for row in cursor.fetchall():
                field_name = row['field_name']
                failure_patterns[field_name]['errors'].append({
                    'radar_type': row['radar_type'],
                    'error': row['validation_error'],
                    'count': row['failure_count']
                })
            
            # Generate improvement suggestions
            improvements = []
            for field_name, data in failure_patterns.items():
                if field_name in ['gain', 'sea_clutter', 'rain_clutter']:
                    improvements.append({
                        'field': field_name,
                        'suggestion': 'Look for bar graphs or percentage indicators',
                        'prompt_addition': f'Look for {field_name} as a bar graph, slider, or percentage value (0-100)'
                    })
                elif field_name in ['vrm1', 'vrm2']:
                    improvements.append({
                        'field': field_name,
                        'suggestion': 'Check if VRM circles are enabled',
                        'prompt_addition': 'Look for VRM (Variable Range Marker) values, may be OFF or disabled'
                    })
                elif field_name in ['tune']:
                    improvements.append({
                        'field': field_name,
                        'suggestion': 'May be set to AUTO',
                        'prompt_addition': 'Look for TUNE value (may show as AUTO or a percentage)'
                    })
            
            # Store improvements
            for improvement in improvements:
                cursor.execute("""
                    INSERT INTO prompt_improvements
                    (field_name, improved_prompt, improvement_reason, created_timestamp)
                    VALUES (%s, %s, %s, %s)
                """, (
                    improvement['field'],
                    improvement['prompt_addition'],
                    improvement['suggestion'],
                    datetime.now()
                ))
            
            conn.commit()
            print(f"✓ Identified {len(improvements)} improvement opportunities")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error learning from failures: {e}")
        finally:
            self.return_connection(conn)
        
        return improvements
    
    def discover_field_synonyms(self):
        """Discover alternative names for fields across different radar types."""
        conn = self.get_connection()
        
        print("\n🔤 Discovering Field Synonyms...")
        
        try:
            cursor = conn.cursor()
            
            # Common synonyms to check
            synonym_mappings = {
                'heading': ['HDG', 'HEAD', 'Ship Head', 'Compass', 'GYRO'],
                'speed': ['SPD', 'STW', 'Speed Through Water', 'Vessel Speed', 'SPEED LOG'],
                'cog': ['Course', 'CSE', 'Course Over Ground', 'GND CRS', 'Track'],
                'sog': ['Speed OG', 'Speed Over Ground', 'GPS Speed', 'GND SPD'],
                'position': ['POS', 'LAT/LON', 'GPS POS', 'Position', 'L/L'],
                'gain': ['GAIN', 'Sensitivity', 'RX Gain', 'Receiver Gain'],
                'sea_clutter': ['SEA', 'A/C SEA', 'Sea Clutter', 'STC', 'SEA CLT'],
                'rain_clutter': ['RAIN', 'A/C RAIN', 'Rain Clutter', 'FTC', 'RAIN CLT'],
                'presentation_mode': ['MODE', 'Display Mode', 'Orientation', 'VIEW'],
                'range': ['RNG', 'Range Scale', 'Display Range'],
                'vector': ['VECT', 'Vector Mode', 'Target Vectors'],
                'ais_on_off': ['AIS', 'AIS Status', 'AIS Display']
            }
            
            for field_name, synonyms in synonym_mappings.items():
                for synonym in synonyms:
                    cursor.execute("""
                        INSERT INTO field_synonyms
                        (field_name, synonym, confidence)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (field_name, synonym, radar_type) DO NOTHING
                    """, (field_name, synonym, 0.9))
            
            conn.commit()
            print(f"✓ Loaded {sum(len(v) for v in synonym_mappings.values())} field synonyms")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error discovering synonyms: {e}")
        finally:
            self.return_connection(conn)
    
    def generate_improved_prompts(self):
        """Generate improved prompts based on learning."""
        conn = self.get_connection()
        
        print("\n📝 Generating Improved Prompts...")
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get all synonyms
            cursor.execute("""
                SELECT field_name, STRING_AGG(synonym, ' or ') as synonyms
                FROM field_synonyms
                GROUP BY field_name
            """)
            
            field_synonyms = {row['field_name']: row['synonyms'] for row in cursor.fetchall()}
            
            # Get recent improvements
            cursor.execute("""
                SELECT DISTINCT field_name, improved_prompt
                FROM prompt_improvements
                WHERE created_timestamp >= CURRENT_DATE - INTERVAL '30 days'
            """)
            
            improvements = {row['field_name']: row['improved_prompt'] for row in cursor.fetchall()}
            
            # Generate improved prompt template
            improved_prompt = """Analyze this marine radar display and extract ALL the following data fields.

IMPORTANT: Look everywhere on the screen including corners, side panels, menus, and overlay windows.

Based on learning from thousands of extractions, here are the key areas to check:

"""
            
            # Add improvements
            if improvements:
                improved_prompt += "SPECIAL ATTENTION REQUIRED:\n"
                for field, improvement in improvements.items():
                    improved_prompt += f"- {field}: {improvement}\n"
                improved_prompt += "\n"
            
            improved_prompt += "Extract these fields (use the exact field names as keys in your JSON response):\n"
            
            # List all fields with synonyms
            all_fields = [
                "presentation_mode", "gain", "sea_clutter", "rain_clutter", "tune",
                "heading", "speed", "cog", "sog", "position", "position_source",
                "range", "range_rings", "cursor_position", "set", "drift",
                "vector", "vector_duration", "cpa_limit", "tcpa_limit",
                "vrm1", "vrm2", "index_line_rng", "index_line_brg", "ais_on_off", "depth"
            ]
            
            for field_name in all_fields:
                if field_name in field_synonyms:
                    improved_prompt += f"\n- {field_name}: Look for {field_synonyms[field_name]}"
                else:
                    improved_prompt += f"\n- {field_name}: Look for {field_name.upper().replace('_', ' ')}"
            
            improved_prompt += "\n\nReturn ONLY a JSON object with these exact field names as keys. Use null for fields not visible."
            
            # Save improved prompt
            output_path = 'improved_extraction_prompt.txt'
            with open(output_path, 'w') as f:
                f.write(improved_prompt)
            
            print(f"✓ Generated improved extraction prompt saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating prompts: {e}")
        finally:
            self.return_connection(conn)
        
        return improved_prompt
    
    def extract_value_pattern(self, field_name: str, value: str, raw_text: str) -> Dict:
        """Extract patterns from successful values."""
        pattern = {
            'field_type': 'unknown',
            'format': None,
            'location_hint': None,
            'value_example': str(value)
        }
        
        if field_name in ['heading', 'cog', 'set', 'index_line_brg']:
            if re.match(r'^\d{1,3}\.?\d*$', str(value)):
                pattern['field_type'] = 'bearing'
                pattern['format'] = 'numeric_0_360'
        
        elif field_name in ['speed', 'sog', 'drift']:
            if re.match(r'^\d{1,2}\.?\d*$', str(value)):
                pattern['field_type'] = 'speed'
                pattern['format'] = 'numeric_knots'
        
        elif field_name == 'position':
            pattern['field_type'] = 'coordinate'
            pattern['format'] = 'lat_lon'
        
        elif field_name in ['gain', 'sea_clutter', 'rain_clutter', 'tune']:
            pattern['field_type'] = 'percentage'
            pattern['format'] = 'numeric_0_100'
        
        elif field_name == 'presentation_mode':
            pattern['field_type'] = 'enum'
            pattern['format'] = 'HEAD_UP|NORTH_UP|COURSE_UP'
        
        return pattern
    
    def update_performance_metrics(self):
        """Track performance improvements over time."""
        conn = self.get_connection()
        
        print("\n📈 Updating Performance Metrics...")
        
        try:
            cursor = conn.cursor()
            
            # Calculate today's performance
            today = datetime.now().date()
            
            query = """
                SELECT 
                    ef.field_name,
                    e.radar_type,
                    COUNT(*) as attempts,
                    SUM(CASE WHEN ef.field_value IS NOT NULL THEN 1 ELSE 0 END) as successes,
                    AVG(CASE WHEN ef.field_value IS NOT NULL THEN ef.confidence ELSE 0 END) as avg_conf
                FROM extracted_fields ef
                JOIN extractions e ON ef.extraction_id = e.extraction_id
                WHERE DATE(e.extraction_timestamp) = %s
                GROUP BY ef.field_name, e.radar_type
            """
            
            cursor.execute(query, (today,))
            
            for row in cursor.fetchall():
                field_name, radar_type, attempts, successes, avg_conf = row
                
                cursor.execute("""
                    INSERT INTO performance_tracking
                    (date, field_name, radar_type, attempts, successes, avg_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, field_name, radar_type)
                    DO UPDATE SET
                        attempts = EXCLUDED.attempts,
                        successes = EXCLUDED.successes,
                        avg_confidence = EXCLUDED.avg_confidence
                """, (today, field_name, radar_type, attempts, successes, avg_conf))
            
            conn.commit()
            print("✓ Performance metrics updated")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating metrics: {e}")
        finally:
            self.return_connection(conn)
    
    def load_knowledge_base(self) -> Dict:
        """Load the accumulated knowledge base."""
        kb_path = 'radar_knowledge_base.json'
        
        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load knowledge base: {e}")
        
        return {
            'version': 2.0,
            'created': datetime.now().isoformat(),
            'patterns': {},
            'synonyms': {},
            'improvements': [],
            'database_type': 'postgresql'
        }
    
    def save_knowledge_base(self):
        """Save the updated knowledge base."""
        self.knowledge_base['last_updated'] = datetime.now().isoformat()
        
        # Save JSON version (pickle not recommended for Azure)
        kb_path = 'radar_knowledge_base.json'
        try:
            with open(kb_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2, default=str)
            logger.info(f"Knowledge base saved to {kb_path}")
        except Exception as e:
            logger.error(f"Could not save knowledge base: {e}")
    
    def close(self):
        """Close all database connections."""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.closeall()
            logger.info("Connection pool closed")

class AdaptiveExtractionEngine:
    """Enhanced extraction engine that uses learned knowledge."""
    
    def __init__(self, learning_system: IntelligentLearningSystem):
        self.learning = learning_system
        self.performance_cache = {}
    
    def get_enhanced_prompt(self, radar_type: str, weak_fields: List[str] = None) -> str:
        """Generate an enhanced prompt using learned knowledge."""
        conn = self.learning.get_connection()
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get synonyms for all fields
            cursor.execute("SELECT field_name, synonym FROM field_synonyms")
            synonyms = defaultdict(list)
            for row in cursor.fetchall():
                synonyms[row['field_name']].append(row['synonym'])
            
            # Get successful patterns for this radar type
            cursor.execute("""
                SELECT field_name, successful_pattern, confidence_boost
                FROM learning_patterns
                WHERE radar_type = %s
                AND confidence_boost > 0
            """, (radar_type,))
            
            patterns = {}
            for row in cursor.fetchall():
                if row['successful_pattern']:
                    patterns[row['field_name']] = row['successful_pattern']
            
            # Build enhanced prompt
            prompt = f"""Analyze this {radar_type} marine radar display image.

Based on previous successful extractions, pay special attention to these patterns:
"""
            
            # Add learned patterns
            for field, pattern in patterns.items():
                if isinstance(pattern, dict) and 'value_example' in pattern:
                    prompt += f"\n- {field}: Usually appears as '{pattern['value_example']}'"
            
            prompt += "\n\nExtract these fields (look for these exact names or variations):\n"
            
            # Add fields with synonyms
            all_fields = [
                "presentation_mode", "gain", "sea_clutter", "rain_clutter", "tune",
                "heading", "speed", "cog", "sog", "position", "position_source",
                "range", "range_rings", "cursor_position", "set", "drift",
                "vector", "vector_duration", "cpa_limit", "tcpa_limit",
                "vrm1", "vrm2", "index_line_rng", "index_line_brg", "ais_on_off", "depth"
            ]
            
            for field in all_fields:
                if field in synonyms:
                    prompt += f"\n- {field}: Look for {field.upper()} or {', '.join(synonyms[field])}"
                else:
                    prompt += f"\n- {field}: Look for {field.upper().replace('_', ' ')}"
                
                # Add special instructions for weak fields
                if weak_fields and field in weak_fields:
                    if field in ['gain', 'sea_clutter', 'rain_clutter']:
                        prompt += " (may be shown as a bar graph or percentage 0-100)"
                    elif field in ['tune']:
                        prompt += " (may be a small indicator or show as AUTO)"
                    elif field in ['vrm1', 'vrm2']:
                        prompt += " (Variable Range Markers - may be OFF or not enabled)"
            
            prompt += "\n\nReturn ONLY a JSON object with these field names as keys. Use null for fields not visible."
            
        except Exception as e:
            logger.error(f"Error generating enhanced prompt: {e}")
            prompt = None
        finally:
            self.learning.return_connection(conn)
        
        return prompt
    
    def apply_confidence_boost(self, field_name: str, radar_type: str, 
                             base_confidence: float) -> float:
        """Apply learned confidence boost to extraction results."""
        conn = self.learning.get_connection()
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT confidence_boost 
                FROM learning_patterns
                WHERE field_name = %s AND radar_type = %s
            """, (field_name, radar_type))
            
            result = cursor.fetchone()
            
            if result:
                boost = result[0]
                return min(base_confidence + boost, 1.0)
            
        except Exception as e:
            logger.error(f"Error applying confidence boost: {e}")
        finally:
            self.learning.return_connection(conn)
        
        return base_confidence

def run_learning_cycle():
    """Run a complete learning cycle."""
    print("\n🚀 Starting Intelligent Learning Cycle\n")
    
    try:
        # Initialize learning system
        learning_system = IntelligentLearningSystem()
        
        # Run analysis and learning
        learning_system.analyze_and_learn()
        
        # Show performance improvements
        conn = learning_system.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("\n📊 Performance Trend (Last 7 Days):")
        cursor.execute("""
            SELECT 
                date,
                SUM(successes)::float / NULLIF(SUM(attempts), 0) as success_rate
            FROM performance_tracking
            WHERE date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY date
            ORDER BY date
        """)
        
        for row in cursor.fetchall():
            if row['success_rate']:
                print(f"  {row['date']}: {row['success_rate']*100:.1f}% success rate")
        
        learning_system.return_connection(conn)
        
        print("\n✅ Learning cycle complete!")
        print("\nTo use enhanced extraction:")
        print("1. The system will now use learned patterns")
        print("2. Weak fields will get special attention")
        print("3. Confidence scores will be boosted based on patterns")
        
        # Clean up
        learning_system.close()
        
    except Exception as e:
        logger.error(f"Learning cycle failed: {e}")
        print(f"\n❌ Learning cycle failed: {e}")

if __name__ == "__main__":
    run_learning_cycle()