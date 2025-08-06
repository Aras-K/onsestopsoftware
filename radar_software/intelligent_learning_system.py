# intelligent_learning_system.py
# Self-learning and retraining system for radar extraction

import json
import sqlite3
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
from collections import defaultdict
import re

class IntelligentLearningSystem:
    """Self-learning system that improves extraction accuracy over time."""
    
    def __init__(self, db_path: str = "radar_extraction_system.db"):
        self.db_path = db_path
        self.learning_db = "radar_learning_system.db"
        self.init_learning_database()
        self.knowledge_base = self.load_knowledge_base()
        
    def init_learning_database(self):
        """Initialize learning system database."""
        conn = sqlite3.connect(self.learning_db)
        cursor = conn.cursor()
        
        # Learning patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                radar_type TEXT NOT NULL,
                field_name TEXT NOT NULL,
                successful_pattern TEXT,
                failed_pattern TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                confidence_boost REAL DEFAULT 0.0,
                last_updated TIMESTAMP
            )
        """)
        
        # Prompt improvements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_improvements (
                improvement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                field_name TEXT NOT NULL,
                original_prompt TEXT,
                improved_prompt TEXT,
                improvement_reason TEXT,
                success_rate_before REAL,
                success_rate_after REAL,
                created_timestamp TIMESTAMP
            )
        """)
        
        # Field synonyms learned
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS field_synonyms (
                synonym_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                tracking_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                field_name TEXT NOT NULL,
                radar_type TEXT,
                attempts INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                avg_confidence REAL,
                UNIQUE(date, field_name, radar_type)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def analyze_and_learn(self):
        """Analyze recent extractions and learn from them."""
        print("🧠 INTELLIGENT LEARNING SYSTEM - Analysis Started")
        print("="*70)
        
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
    
    def analyze_success_patterns(self):
        """Analyze what patterns lead to successful extractions."""
        conn = sqlite3.connect(self.db_path)
        learning_conn = sqlite3.connect(self.learning_db)
        
        print("\n📊 Analyzing Success Patterns...")
        
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
            ORDER BY ef.confidence DESC
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        patterns = defaultdict(lambda: defaultdict(list))
        
        for row in cursor.fetchall():
            radar_type, field_name, field_value, raw_text, confidence = row
            
            # Extract patterns from successful values
            pattern = self.extract_value_pattern(field_name, field_value, raw_text)
            patterns[radar_type][field_name].append({
                'pattern': pattern,
                'confidence': confidence,
                'example': field_value
            })
        
        # Save learned patterns
        learning_cursor = learning_conn.cursor()
        
        for radar_type, fields in patterns.items():
            for field_name, field_patterns in fields.items():
                # Find most common successful pattern
                if field_patterns:
                    best_pattern = max(field_patterns, key=lambda x: x['confidence'])
                    
                    learning_cursor.execute("""
                        INSERT OR REPLACE INTO learning_patterns
                        (radar_type, field_name, successful_pattern, success_count, confidence_boost, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        radar_type,
                        field_name,
                        json.dumps(best_pattern),
                        len(field_patterns),
                        sum(p['confidence'] for p in field_patterns) / len(field_patterns) - 0.5,
                        datetime.now()
                    ))
        
        learning_conn.commit()
        conn.close()
        learning_conn.close()
        
        print(f"✓ Learned patterns for {len(patterns)} radar types")
    
    def learn_from_failures(self):
        """Learn why certain fields fail to extract."""
        conn = sqlite3.connect(self.db_path)
        
        print("\n🔍 Learning from Failures...")
        
        # Analyze failed extractions
        query = """
            SELECT 
                e.radar_type,
                ef.field_name,
                ef.validation_error,
                COUNT(*) as failure_count
            FROM extracted_fields ef
            JOIN extractions e ON ef.extraction_id = e.extraction_id
            WHERE ef.field_value IS NULL
            OR ef.confidence < 0.5
            GROUP BY e.radar_type, ef.field_name, ef.validation_error
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        failure_patterns = defaultdict(lambda: defaultdict(list))
        
        for row in cursor.fetchall():
            radar_type, field_name, error, count = row
            failure_patterns[field_name]['errors'].append({
                'radar_type': radar_type,
                'error': error,
                'count': count
            })
        
        # Generate improvement suggestions
        improvements = []
        for field_name, data in failure_patterns.items():
            if field_name in ['gain', 'sea_clutter', 'rain_clutter']:
                improvements.append({
                    'field': field_name,
                    'suggestion': 'Look for bar graphs or percentage indicators',
                    'prompt_addition': f'Look for {field_name} as a bar graph, slider, or percentage value'
                })
            elif field_name in ['vrm1', 'vrm2']:
                improvements.append({
                    'field': field_name,
                    'suggestion': 'Check if VRM circles are enabled',
                    'prompt_addition': 'Look for VRM (Variable Range Marker) values, may be disabled if not shown'
                })
        
        conn.close()
        
        print(f"✓ Identified {len(improvements)} improvement opportunities")
        return improvements
    
    def discover_field_synonyms(self):
        """Discover alternative names for fields across different radar types."""
        conn = sqlite3.connect(self.db_path)
        learning_conn = sqlite3.connect(self.learning_db)
        
        print("\n🔤 Discovering Field Synonyms...")
        
        # Common synonyms to check
        synonym_mappings = {
            'heading': ['HDG', 'HEAD', 'Ship Head', 'Compass'],
            'speed': ['SPD', 'STW', 'Speed Through Water', 'Vessel Speed'],
            'cog': ['Course', 'CSE', 'Course Over Ground'],
            'sog': ['Speed OG', 'Speed Over Ground', 'GPS Speed'],
            'position': ['POS', 'LAT/LON', 'GPS POS', 'Position'],
            'gain': ['GAIN', 'Sensitivity', 'RX Gain'],
            'sea_clutter': ['SEA', 'A/C SEA', 'Sea Clutter', 'STC'],
            'rain_clutter': ['RAIN', 'A/C RAIN', 'Rain Clutter', 'FTC']
        }
        
        learning_cursor = learning_conn.cursor()
        
        for field_name, synonyms in synonym_mappings.items():
            for synonym in synonyms:
                learning_cursor.execute("""
                    INSERT OR IGNORE INTO field_synonyms
                    (field_name, synonym, confidence)
                    VALUES (?, ?, ?)
                """, (field_name, synonym, 0.9))
        
        learning_conn.commit()
        conn.close()
        learning_conn.close()
        
        print(f"✓ Loaded {sum(len(v) for v in synonym_mappings.values())} field synonyms")
    
    def generate_improved_prompts(self):
        """Generate improved prompts based on learning."""
        learning_conn = sqlite3.connect(self.learning_db)
        
        print("\n📝 Generating Improved Prompts...")
        
        # Get all synonyms
        cursor = learning_conn.cursor()
        cursor.execute("""
            SELECT field_name, GROUP_CONCAT(synonym, ' or ') as synonyms
            FROM field_synonyms
            GROUP BY field_name
        """)
        
        field_synonyms = dict(cursor.fetchall())
        
        # Generate improved prompt template
        improved_prompt = """Analyze this marine radar display and extract ALL the following data fields.

IMPORTANT: Look everywhere on the screen including corners, side panels, and overlay windows.

Extract these fields (use the exact field names as keys in your JSON response):
"""
        
        for field_name, synonyms in field_synonyms.items():
            improved_prompt += f"\n- {field_name}: Look for {synonyms}"
        
        # Save improved prompt
        with open('improved_extraction_prompt.txt', 'w') as f:
            f.write(improved_prompt)
        
        learning_conn.close()
        
        print("✓ Generated improved extraction prompt")
        return improved_prompt
    
    def extract_value_pattern(self, field_name: str, value: str, raw_text: str) -> Dict:
        """Extract patterns from successful values."""
        pattern = {
            'field_type': 'unknown',
            'format': None,
            'location_hint': None
        }
        
        if field_name in ['heading', 'cog']:
            if re.match(r'^\d{1,3}\.?\d*$', str(value)):
                pattern['field_type'] = 'bearing'
                pattern['format'] = 'numeric_0_360'
        
        elif field_name in ['speed', 'sog']:
            if re.match(r'^\d{1,2}\.?\d*$', str(value)):
                pattern['field_type'] = 'speed'
                pattern['format'] = 'numeric_knots'
        
        elif field_name == 'position':
            pattern['field_type'] = 'coordinate'
            pattern['format'] = 'lat_lon'
        
        return pattern
    
    def update_performance_metrics(self):
        """Track performance improvements over time."""
        conn = sqlite3.connect(self.db_path)
        learning_conn = sqlite3.connect(self.learning_db)
        
        print("\n📈 Updating Performance Metrics...")
        
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
            WHERE DATE(e.extraction_timestamp) = ?
            GROUP BY ef.field_name, e.radar_type
        """
        
        cursor = conn.cursor()
        cursor.execute(query, (today,))
        
        learning_cursor = learning_conn.cursor()
        
        for row in cursor.fetchall():
            field_name, radar_type, attempts, successes, avg_conf = row
            
            learning_cursor.execute("""
                INSERT OR REPLACE INTO performance_tracking
                (date, field_name, radar_type, attempts, successes, avg_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (today, field_name, radar_type, attempts, successes, avg_conf))
        
        learning_conn.commit()
        conn.close()
        learning_conn.close()
        
        print("✓ Performance metrics updated")
    
    def load_knowledge_base(self) -> Dict:
        """Load the accumulated knowledge base."""
        if os.path.exists('radar_knowledge_base.pkl'):
            with open('radar_knowledge_base.pkl', 'rb') as f:
                return pickle.load(f)
        return {
            'version': 1.0,
            'created': datetime.now().isoformat(),
            'patterns': {},
            'synonyms': {},
            'improvements': []
        }
    
    def save_knowledge_base(self):
        """Save the updated knowledge base."""
        self.knowledge_base['last_updated'] = datetime.now().isoformat()
        
        with open('radar_knowledge_base.pkl', 'wb') as f:
            pickle.dump(self.knowledge_base, f)
        
        # Also save human-readable version
        with open('radar_knowledge_base.json', 'w') as f:
            json.dump(self.knowledge_base, f, indent=2, default=str)

class AdaptiveExtractionEngine:
    """Enhanced extraction engine that uses learned knowledge."""
    
    def __init__(self, learning_system: IntelligentLearningSystem):
        self.learning = learning_system
        self.performance_cache = {}
    
    def get_enhanced_prompt(self, radar_type: str, weak_fields: List[str] = None) -> str:
        """Generate an enhanced prompt using learned knowledge."""
        conn = sqlite3.connect(self.learning.learning_db)
        cursor = conn.cursor()
        
        # Get synonyms for all fields
        cursor.execute("SELECT field_name, synonym FROM field_synonyms")
        synonyms = defaultdict(list)
        for field, synonym in cursor.fetchall():
            synonyms[field].append(synonym)
        
        # Get successful patterns for this radar type
        cursor.execute("""
            SELECT field_name, successful_pattern, confidence_boost
            FROM learning_patterns
            WHERE radar_type = ?
            AND confidence_boost > 0
        """, (radar_type,))
        
        patterns = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
        
        # Build enhanced prompt
        prompt = f"""Analyze this {radar_type} marine radar display image.

Based on previous successful extractions, pay special attention to these patterns:
"""
        
        # Add learned patterns
        for field, pattern in patterns.items():
            if 'example' in pattern:
                prompt += f"\n- {field}: Usually appears as '{pattern['example']}'"
        
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
                prompt += f"\n- {field}: Look for {field.upper()}"
            
            # Add special instructions for weak fields
            if weak_fields and field in weak_fields:
                if field in ['gain', 'sea_clutter', 'rain_clutter']:
                    prompt += " (may be shown as a bar graph or percentage)"
                elif field in ['tune']:
                    prompt += " (may be a small indicator or AUTO)"
                elif field in ['vrm1', 'vrm2']:
                    prompt += " (Variable Range Markers - may be OFF)"
        
        prompt += "\n\nReturn ONLY a JSON object with these field names as keys."
        
        conn.close()
        return prompt
    
    def apply_confidence_boost(self, field_name: str, radar_type: str, 
                             base_confidence: float) -> float:
        """Apply learned confidence boost to extraction results."""
        conn = sqlite3.connect(self.learning.learning_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT confidence_boost 
            FROM learning_patterns
            WHERE field_name = ? AND radar_type = ?
        """, (field_name, radar_type))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            boost = result[0]
            return min(base_confidence + boost, 1.0)
        
        return base_confidence

def run_learning_cycle():
    """Run a complete learning cycle."""
    print("\n🚀 Starting Intelligent Learning Cycle\n")
    
    # Initialize learning system
    learning_system = IntelligentLearningSystem()
    
    # Run analysis and learning
    learning_system.analyze_and_learn()
    
    # Show performance improvements
    conn = sqlite3.connect(learning_system.learning_db)
    cursor = conn.cursor()
    
    print("\n📊 Performance Trend (Last 7 Days):")
    cursor.execute("""
        SELECT 
            date,
            SUM(successes) * 1.0 / SUM(attempts) as success_rate
        FROM performance_tracking
        WHERE date >= date('now', '-7 days')
        GROUP BY date
        ORDER BY date
    """)
    
    for date, rate in cursor.fetchall():
        print(f"  {date}: {rate*100:.1f}% success rate")
    
    conn.close()
    
    print("\n✅ Learning cycle complete!")
    print("\nTo use enhanced extraction:")
    print("1. The system will now use learned patterns")
    print("2. Weak fields will get special attention")
    print("3. Confidence scores will be boosted based on patterns")

# Integration with main extraction engine
def enhance_extraction_engine():
    """Enhance the existing extraction engine with learning capabilities."""
    
    code_to_add = '''
# Add to HybridRadarExtractor.__init__:
self.learning_system = IntelligentLearningSystem()
self.adaptive_engine = AdaptiveExtractionEngine(self.learning_system)

# Replace create_extraction_prompt with:
def create_extraction_prompt(self, radar_type: RadarType, target_fields: List[str] = None) -> str:
    # Get fields with low success rates
    weak_fields = self.get_weak_fields()
    
    # Use enhanced prompt from learning system
    return self.adaptive_engine.get_enhanced_prompt(
        radar_type.value, 
        weak_fields
    )

# Add this method:
def get_weak_fields(self) -> List[str]:
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
'''
    
    print("\n📝 Code to add to radar_extraction_engine.py:")
    print(code_to_add)

if __name__ == "__main__":
    # Run learning cycle
    run_learning_cycle()
    
    # Show how to integrate
    enhance_extraction_engine()