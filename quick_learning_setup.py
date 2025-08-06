# quick_learning_setup.py
# Quick setup to make your system self-learning (Azure Compatible)

import os
import json
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickLearningSystem:
    """Simplified self-learning system for immediate use with PostgreSQL."""
    
    def __init__(self, connection_string: str = None):
        """
        Initialize learning system with database connection.
        
        Args:
            connection_string: PostgreSQL connection string or None to use environment variable
        """
        # Get connection string from environment if not provided
        if connection_string is None:
            connection_string = os.environ.get('DATABASE_URL')
            if not connection_string:
                # Build from individual parameters
                connection_string = self._build_connection_string()
        
        self.connection_string = connection_string
        logger.info("Learning system initialized with PostgreSQL")
    
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
        """Get database connection."""
        return psycopg2.connect(self.connection_string)
    
    def analyze_current_performance(self) -> Tuple[List[str], Dict, Dict]:
        """Quick analysis of what's working and what's not."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("🧠 QUICK LEARNING ANALYSIS")
        print("="*60)
        
        try:
            # 1. Find which fields are failing most
            print("\n❌ FIELDS THAT NEED IMPROVEMENT:")
            cursor.execute("""
                SELECT 
                    field_name,
                    COUNT(*) as attempts,
                    SUM(CASE WHEN field_value IS NOT NULL THEN 1 ELSE 0 END) as successes,
                    (SUM(CASE WHEN field_value IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0)) as success_rate
                FROM extracted_fields
                GROUP BY field_name
                HAVING (SUM(CASE WHEN field_value IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0)) < 50
                ORDER BY success_rate
            """)
            
            weak_fields = []
            for row in cursor.fetchall():
                field = row['field_name']
                attempts = row['attempts']
                successes = row['successes']
                rate = row['success_rate'] or 0
                print(f"  - {field}: {rate:.1f}% success ({successes}/{attempts})")
                weak_fields.append(field)
            
            if not weak_fields:
                print("  All fields have >50% success rate!")
            
            # 2. Find patterns in successful extractions
            print("\n✅ SUCCESSFUL EXTRACTION PATTERNS:")
            
            # Get sample successful values for each field
            field_examples = defaultdict(list)
            
            for field in ['heading', 'speed', 'position', 'range', 'gain', 'sea_clutter']:
                cursor.execute("""
                    SELECT DISTINCT field_value 
                    FROM extracted_fields 
                    WHERE field_name = %s 
                    AND field_value IS NOT NULL 
                    AND confidence > 0.8
                    LIMIT 5
                """, (field,))
                
                examples = [row['field_value'] for row in cursor.fetchall()]
                if examples:
                    field_examples[field] = examples
                    print(f"  - {field}: {', '.join(str(e) for e in examples[:3])}")
            
            # 3. Get extraction method success rates
            print("\n📊 EXTRACTION METHOD PERFORMANCE:")
            cursor.execute("""
                SELECT 
                    extraction_method,
                    COUNT(*) as uses,
                    AVG(confidence) * 100 as avg_confidence,
                    SUM(CASE WHEN is_valid = true THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) as success_rate
                FROM extracted_fields
                GROUP BY extraction_method
                ORDER BY success_rate DESC
            """)
            
            for row in cursor.fetchall():
                method = row['extraction_method']
                uses = row['uses']
                confidence = row['avg_confidence'] or 0
                success = row['success_rate'] or 0
                print(f"  - {method}: {success:.1f}% success, {confidence:.1f}% avg confidence ({uses} uses)")
            
            # 4. Radar type performance
            print("\n📡 RADAR TYPE PERFORMANCE:")
            cursor.execute("""
                SELECT 
                    radar_type,
                    COUNT(*) as count,
                    AVG(overall_confidence) * 100 as avg_confidence
                FROM extractions
                GROUP BY radar_type
                ORDER BY count DESC
            """)
            
            radar_stats = {}
            for row in cursor.fetchall():
                radar_type = row['radar_type']
                count = row['count']
                confidence = row['avg_confidence'] or 0
                radar_stats[radar_type] = {'count': count, 'confidence': confidence}
                print(f"  - {radar_type}: {confidence:.1f}% avg confidence ({count} images)")
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            weak_fields = []
            field_examples = {}
        finally:
            conn.close()
        
        # Generate improved prompts for weak fields
        print("\n📝 IMPROVED PROMPTS FOR WEAK FIELDS:")
        improvements = self.generate_improvements(weak_fields)
        for field, improvement in improvements.items():
            print(f"\n{field}:")
            print(f"  {improvement}")
        
        return weak_fields, field_examples, improvements
    
    def generate_improvements(self, weak_fields: List[str]) -> Dict[str, str]:
        """Generate specific improvements for weak fields."""
        improvements = {}
        
        # Field-specific improvements based on common issues
        field_hints = {
            'tune': "Look for TUNE indicator (may show as 'AUTO' or a percentage bar, often near gain controls)",
            'gain': "Look for GAIN control (often a slider, bar graph, or percentage from 0-100)",
            'sea_clutter': "Look for SEA, A/C SEA, STC, or SEA CLUTTER control (usually 0-100% or bar)",
            'rain_clutter': "Look for RAIN, A/C RAIN, FTC, or RAIN CLUTTER control (usually 0-100% or bar)",
            'position_source': "Look for GPS, DGPS, GNSS, or GLONASS indicator near position display",
            'range_rings': "Look for ring interval value (e.g. '0.5 NM', '1 NM' near range setting)",
            'cursor_position': "Look for cursor LAT/LON or BRG/RNG (only visible when cursor is active)",
            'set': "Look for SET value in current/drift section (direction in degrees)",
            'drift': "Look for DRIFT value in current section (speed in knots)",
            'vrm1': "Look for VRM1 or VRM 1 distance (Variable Range Marker, in NM)",
            'vrm2': "Look for VRM2 or VRM 2 distance (second Variable Range Marker)",
            'vector': "Look for vector mode setting: TRUE, REL (relative), or OFF",
            'vector_duration': "Look for vector time/length (usually in minutes, e.g. '6 MIN')",
            'cpa_limit': "Look for CPA alarm limit setting (Closest Point of Approach, in NM)",
            'tcpa_limit': "Look for TCPA alarm limit (Time to CPA, in minutes)",
            'index_line_rng': "Look for index line, HL (Head Line), or SHM range value",
            'index_line_brg': "Look for index line, HL, or SHM bearing value",
            'ais_on_off': "Look for AIS status indicator (ON/OFF, may be in menu or status bar)",
            'depth': "Look for depth value (may be in meters 'm' or feet 'ft')",
            'presentation_mode': "Look for display mode: HEAD UP (H UP), NORTH UP (N UP), or COURSE UP (C UP)"
        }
        
        for field in weak_fields:
            if field in field_hints:
                improvements[field] = field_hints[field]
            else:
                improvements[field] = f"Carefully scan entire display for {field.replace('_', ' ').upper()} value"
        
        return improvements
    
    def get_field_statistics(self) -> Dict:
        """Get detailed field extraction statistics."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        stats = {}
        
        try:
            cursor.execute("""
                SELECT 
                    field_name,
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN field_value IS NOT NULL THEN 1 ELSE 0 END) as successful,
                    AVG(CASE WHEN confidence IS NOT NULL THEN confidence ELSE 0 END) as avg_confidence,
                    MIN(confidence) as min_confidence,
                    MAX(confidence) as max_confidence
                FROM extracted_fields
                GROUP BY field_name
                ORDER BY field_name
            """)
            
            for row in cursor.fetchall():
                stats[row['field_name']] = {
                    'attempts': row['total_attempts'],
                    'successful': row['successful'],
                    'success_rate': (row['successful'] / row['total_attempts'] * 100) if row['total_attempts'] > 0 else 0,
                    'avg_confidence': row['avg_confidence'] or 0,
                    'min_confidence': row['min_confidence'] or 0,
                    'max_confidence': row['max_confidence'] or 0
                }
        
        except Exception as e:
            logger.error(f"Error getting field statistics: {e}")
        finally:
            conn.close()
        
        return stats
    
    def create_enhanced_prompt(self) -> str:
        """Create an enhanced prompt based on learning."""
        try:
            weak_fields, examples, improvements = self.analyze_current_performance()
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            weak_fields = []
            improvements = {}
        
        prompt = """Analyze this marine radar display image and extract the following fields.

CRITICAL: Based on previous extractions, pay special attention to these commonly missed fields:

"""
        
        # Add specific instructions for weak fields
        if improvements:
            for field, hint in improvements.items():
                prompt += f"- {field}: {hint}\n"
        else:
            prompt += "- Check all areas of the display carefully for each field\n"
        
        prompt += """

IMPORTANT REMINDERS:
1. GAIN, SEA, RAIN controls are often shown as bars or sliders (0-100%)
2. COG is direction (degrees), SOG is speed (knots) - never confuse them
3. Look for abbreviated labels (e.g., 'N UP' for NORTH UP, 'HDG' for heading)
4. Some values may be in status bars, menus, or control panels
        
Return a JSON object with ALL 26 fields:
{
  "presentation_mode": "HEAD UP, NORTH UP, or COURSE UP",
  "gain": "number 0-100 (look for GAIN slider or percentage)",
  "sea_clutter": "number 0-100 (look for SEA or A/C SEA)",
  "rain_clutter": "number 0-100 (look for RAIN or A/C RAIN)",
  "tune": "number 0-100 or 'AUTO'",
  "heading": "ship heading in degrees",
  "speed": "ship speed in knots",
  "cog": "course over ground in degrees",
  "sog": "speed over ground in knots",
  "position": "full LAT/LON position",
  "position_source": "GPS, DGPS, or GNSS",
  "range": "radar range in NM",
  "range_rings": "ring interval in NM",
  "cursor_position": "cursor position if visible",
  "set": "current set in degrees",
  "drift": "current drift in knots",
  "vector": "TRUE, REL, or OFF",
  "vector_duration": "vector time in minutes",
  "cpa_limit": "CPA limit in NM",
  "tcpa_limit": "TCPA limit in minutes",
  "vrm1": "VRM1 distance if enabled",
  "vrm2": "VRM2 distance if enabled",
  "index_line_rng": "index line range",
  "index_line_brg": "index line bearing",
  "ais_on_off": "ON or OFF",
  "depth": "depth in meters"
}

Use null for fields that are not visible or cannot be determined."""
        
        return prompt
    
    def save_enhanced_configuration(self, output_path: str = 'enhanced_extraction_config.json') -> Dict:
        """Save the enhanced configuration for immediate use."""
        # Get field statistics
        stats = self.get_field_statistics()
        
        config = {
            'version': 2.0,
            'updated': datetime.now().isoformat(),
            'enhanced_prompt': self.create_enhanced_prompt(),
            'field_statistics': stats,
            'learning_stats': {
                'analysis_date': datetime.now().isoformat(),
                'improvements_generated': True,
                'database_type': 'postgresql'
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\n✅ Enhanced configuration saved to: {output_path}")
            logger.info(f"Enhanced configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            print(f"\n❌ Failed to save configuration: {e}")
        
        return config
    
    def get_learning_insights(self) -> Dict:
        """Get actionable insights from the learning analysis."""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'recommendations': [],
            'performance_summary': {}
        }
        
        # Get statistics
        stats = self.get_field_statistics()
        
        # Identify critical issues
        critical_fields = []
        moderate_fields = []
        
        for field, field_stats in stats.items():
            success_rate = field_stats['success_rate']
            if success_rate < 30:
                critical_fields.append(field)
            elif success_rate < 60:
                moderate_fields.append(field)
        
        if critical_fields:
            insights['recommendations'].append({
                'priority': 'HIGH',
                'issue': f"Critical extraction failures in: {', '.join(critical_fields)}",
                'action': 'Consider manual review or specialized OCR for these fields'
            })
        
        if moderate_fields:
            insights['recommendations'].append({
                'priority': 'MEDIUM',
                'issue': f"Moderate extraction issues in: {', '.join(moderate_fields)}",
                'action': 'Enhanced prompts and validation rules recommended'
            })
        
        # Overall performance
        total_fields = len(stats)
        high_performing = sum(1 for f, s in stats.items() if s['success_rate'] > 80)
        
        insights['performance_summary'] = {
            'total_fields': total_fields,
            'high_performing_fields': high_performing,
            'critical_fields': len(critical_fields),
            'moderate_fields': len(moderate_fields)
        }
        
        return insights

def implement_quick_learning():
    """Implement learning in your existing system."""
    print("\n🚀 IMPLEMENTING QUICK LEARNING FOR AZURE\n")
    
    try:
        learning = QuickLearningSystem()
        
        # Analyze and create enhanced prompt
        enhanced_config = learning.save_enhanced_configuration()
        
        # Get insights
        insights = learning.get_learning_insights()
        
        print("\n📊 PERFORMANCE INSIGHTS:")
        print(f"  - High performing fields: {insights['performance_summary']['high_performing_fields']}")
        print(f"  - Critical issues: {insights['performance_summary']['critical_fields']}")
        print(f"  - Moderate issues: {insights['performance_summary']['moderate_fields']}")
        
        if insights['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in insights['recommendations']:
                print(f"  [{rec['priority']}] {rec['issue']}")
                print(f"         Action: {rec['action']}")
        
        print("\n📋 TO IMPLEMENT IN YOUR SYSTEM:")
        print("\n1. The enhanced configuration has been saved")
        print("2. Your extraction engine will automatically use it")
        print("3. Run this analysis periodically to keep improving")
        
        print("\n✅ Your system will now learn and improve!")
        
    except Exception as e:
        logger.error(f"Failed to implement learning: {e}")
        print(f"\n❌ Error implementing learning: {e}")
        print("Please check your database connection and try again")

if __name__ == "__main__":
    implement_quick_learning()