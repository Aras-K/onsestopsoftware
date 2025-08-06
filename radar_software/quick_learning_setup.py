# quick_learning_setup.py
# Quick setup to make your system self-learning

import sqlite3
import json
from datetime import datetime
from collections import defaultdict

class QuickLearningSystem:
    """Simplified self-learning system for immediate use."""
    
    def __init__(self):
        self.db_path = "radar_extraction_system.db"
        
    def analyze_current_performance(self):
        """Quick analysis of what's working and what's not."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("🧠 QUICK LEARNING ANALYSIS")
        print("="*60)
        
        # 1. Find which fields are failing most
        print("\n❌ FIELDS THAT NEED IMPROVEMENT:")
        cursor.execute("""
            SELECT 
                field_name,
                COUNT(*) as attempts,
                SUM(CASE WHEN field_value IS NOT NULL THEN 1 ELSE 0 END) as successes,
                (SUM(CASE WHEN field_value IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as success_rate
            FROM extracted_fields
            GROUP BY field_name
            HAVING success_rate < 50
            ORDER BY success_rate
        """)
        
        weak_fields = []
        for field, attempts, successes, rate in cursor.fetchall():
            print(f"  - {field}: {rate:.1f}% success ({successes}/{attempts})")
            weak_fields.append(field)
        
        # 2. Find patterns in successful extractions
        print("\n✅ SUCCESSFUL EXTRACTION PATTERNS:")
        
        # Get sample successful values for each field
        field_examples = defaultdict(list)
        
        for field in ['heading', 'speed', 'position', 'range']:
            cursor.execute("""
                SELECT DISTINCT field_value 
                FROM extracted_fields 
                WHERE field_name = ? 
                AND field_value IS NOT NULL 
                AND confidence > 0.8
                LIMIT 5
            """, (field,))
            
            examples = [row[0] for row in cursor.fetchall()]
            if examples:
                field_examples[field] = examples
                print(f"  - {field}: {', '.join(examples[:3])}")
        
        conn.close()
        
        # 3. Generate improved prompts for weak fields
        print("\n📝 IMPROVED PROMPTS FOR WEAK FIELDS:")
        
        improvements = self.generate_improvements(weak_fields)
        for field, improvement in improvements.items():
            print(f"\n{field}:")
            print(f"  {improvement}")
        
        return weak_fields, field_examples, improvements
    
    def generate_improvements(self, weak_fields):
        """Generate specific improvements for weak fields."""
        improvements = {}
        
        # Field-specific improvements based on common issues
        field_hints = {
            'tune': "Look for TUNE indicator (may show as 'AUTO' or a percentage bar)",
            'gain': "Look for GAIN control (often a slider or percentage from 0-100)",
            'sea_clutter': "Look for SEA, A/C SEA, or STC control (usually 0-100%)",
            'rain_clutter': "Look for RAIN, A/C RAIN, or FTC control (usually 0-100%)",
            'position_source': "Look for GPS, DGPS, GNSS indicator near position display",
            'range_rings': "Look for ring interval value (e.g. '1 NM' near range setting)",
            'cursor_position': "Look for cursor LAT/LON or BRG/RNG (only when cursor visible)",
            'set': "Look for SET value (current direction in degrees)",
            'drift': "Look for DRIFT value (current speed in knots)",
            'vrm1': "Look for VRM1 distance (Variable Range Marker 1)",
            'vrm2': "Look for VRM2 distance (Variable Range Marker 2)",
            'vector': "Look for vector mode: TRUE, REL, or OFF",
            'cpa_limit': "Look for CPA alarm limit setting (in NM)",
            'tcpa_limit': "Look for TCPA alarm limit (in minutes)",
            'index_line_rng': "Look for index line range value",
            'index_line_brg': "Look for index line bearing value",
            'ais_on_off': "Look for AIS status indicator (ON/OFF)",
            'depth': "Look for depth value (may be in meters or feet)"
        }
        
        for field in weak_fields:
            if field in field_hints:
                improvements[field] = field_hints[field]
            else:
                improvements[field] = f"Ensure to look for {field.upper()} in all areas of the display"
        
        return improvements
    
    def create_enhanced_prompt(self):
        """Create an enhanced prompt based on learning."""
        weak_fields, examples, improvements = self.analyze_current_performance()
        
        prompt = """Analyze this marine radar display image and extract the following fields.

CRITICAL: Based on previous extractions, pay special attention to these commonly missed fields:

"""
        
        # Add specific instructions for weak fields
        for field, hint in improvements.items():
            prompt += f"- {field}: {hint}\n"
        
        prompt += """
        
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

Use null for fields that are not visible."""
        
        return prompt
    
    def save_enhanced_configuration(self):
        """Save the enhanced configuration for immediate use."""
        config = {
            'version': 2.0,
            'updated': datetime.now().isoformat(),
            'enhanced_prompt': self.create_enhanced_prompt(),
            'learning_stats': {
                'analysis_date': datetime.now().isoformat(),
                'improvements_generated': True
            }
        }
        
        with open('enhanced_extraction_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n✅ Enhanced configuration saved to: enhanced_extraction_config.json")
        
        return config

def implement_quick_learning():
    """Implement learning in your existing system."""
    print("\n🚀 IMPLEMENTING QUICK LEARNING\n")
    
    learning = QuickLearningSystem()
    
    # Analyze and create enhanced prompt
    enhanced_config = learning.save_enhanced_configuration()
    
    print("\n📋 TO IMPLEMENT IN YOUR SYSTEM:")
    print("\n1. Update radar_extraction_engine.py:")
    print("   - Load enhanced_extraction_config.json")
    print("   - Use the enhanced prompt for Gemini")
    
    print("\n2. Add this to your AIVisionExtractor:")
    
    code = '''
def create_extraction_prompt(self, radar_type, target_fields=None):
    # Load enhanced prompt if available
    if os.path.exists('enhanced_extraction_config.json'):
        with open('enhanced_extraction_config.json', 'r') as f:
            config = json.load(f)
        return config['enhanced_prompt']
    
    # Fallback to original prompt
    return self.original_prompt(radar_type, target_fields)
'''
    
    print(code)
    
    print("\n3. Run learning analysis periodically:")
    print("   python quick_learning_setup.py")
    
    print("\n✅ Your system will now learn and improve!")

if __name__ == "__main__":
    implement_quick_learning()