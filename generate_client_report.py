# generate_client_report.py
# Comprehensive report generator for One Stop Portal client

import sqlite3
import pandas as pd
import os
from datetime import datetime
from tabulate import tabulate
import json
import csv

class ClientReportGenerator:
    """Generate comprehensive reports for client presentation."""
    
    def __init__(self, db_path: str = "radar_extraction_system.db"):
        self.db_path = db_path
        self.report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_all_reports(self):
        """Generate all report formats for the client."""
        print("="*70)
        print("ONE STOP PORTAL - COMPREHENSIVE DATA EXTRACTION REPORT")
        print("="*70)
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. Summary Report
        self.generate_summary_report()
        
        # 2. Detailed Extraction Table
        self.generate_detailed_extraction_table()
        
        # 3. Field Success Analysis
        self.generate_field_analysis()
        
        # 4. Milestone Achievement Report
        self.generate_milestone_report()
        
        # 5. Export to Multiple Formats
        self.export_to_csv()
        self.export_to_html()
        self.export_to_excel()
        
        print("\n" + "="*70)
        print("ALL REPORTS GENERATED SUCCESSFULLY!")
        print("="*70)
    
    def generate_summary_report(self):
        """Generate high-level summary statistics."""
        conn = sqlite3.connect(self.db_path)
        
        # Overall statistics
        summary = pd.read_sql_query("""
            SELECT 
                COUNT(DISTINCT extraction_id) as total_images,
                AVG(overall_confidence) as avg_confidence,
                SUM(CASE WHEN extraction_status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN extraction_status = 'partial' THEN 1 ELSE 0 END) as partial,
                SUM(CASE WHEN extraction_status = 'failed' THEN 1 ELSE 0 END) as failed,
                COUNT(DISTINCT radar_type) as radar_types
            FROM extractions
        """, conn)
        
        print("\n--- EXTRACTION SUMMARY ---")
        print(f"Total Images Processed: {summary.iloc[0]['total_images']}")
        print(f"Average Confidence: {summary.iloc[0]['avg_confidence']:.2%}")
        print(f"Successful: {summary.iloc[0]['successful']}")
        print(f"Partial: {summary.iloc[0]['partial']}")
        print(f"Failed: {summary.iloc[0]['failed']}")
        print(f"Radar Types: {summary.iloc[0]['radar_types']}")
        
        conn.close()
    
    def generate_detailed_extraction_table(self):
        """Generate detailed table of all extractions."""
        conn = sqlite3.connect(self.db_path)
        
        # Get all extractions with field counts
        query = """
            SELECT 
                e.extraction_id,
                e.filename,
                e.radar_type,
                e.overall_confidence,
                e.extraction_status,
                COUNT(DISTINCT ef.field_name) as fields_extracted,
                GROUP_CONCAT(DISTINCT 
                    CASE WHEN ef.field_value IS NOT NULL 
                    THEN ef.field_name 
                    ELSE NULL END
                ) as extracted_fields
            FROM extractions e
            LEFT JOIN extracted_fields ef ON e.extraction_id = ef.extraction_id
            GROUP BY e.extraction_id
            ORDER BY e.extraction_id
        """
        
        df = pd.read_sql_query(query, conn)
        
        print("\n--- DETAILED EXTRACTION RESULTS ---")
        
        # Create summary table
        table_data = []
        for _, row in df.iterrows():
            # Check if meets Milestone 1 (21+ fields)
            milestone_status = "YES" if row['fields_extracted'] >= 21 else "NO"
            
            table_data.append([
                row['extraction_id'],
                row['filename'][:30] + "..." if len(row['filename']) > 30 else row['filename'],
                row['radar_type'],
                f"{row['overall_confidence']:.1%}",
                f"{row['fields_extracted']}/26",
                milestone_status
            ])
        
        print(tabulate(
            table_data[:20],  # Show first 20
            headers=["ID", "Filename", "Radar Type", "Confidence", "Fields", "M1 (21+)"],
            tablefmt="grid"
        ))
        
        if len(table_data) > 20:
            print(f"\n... and {len(table_data) - 20} more entries")
        
        # Save full table to file
        with open(f"extraction_details_{self.report_timestamp}.txt", 'w') as f:
            f.write(tabulate(
                table_data,
                headers=["ID", "Filename", "Radar Type", "Confidence", "Fields", "M1 (21+)"],
                tablefmt="grid"
            ))
        
        conn.close()
    
    def generate_field_analysis(self):
        """Analyze success rate for each of the 26 fields."""
        conn = sqlite3.connect(self.db_path)
        
        # Get field extraction statistics
        query = """
            SELECT 
                field_name,
                COUNT(*) as attempts,
                SUM(CASE WHEN field_value IS NOT NULL THEN 1 ELSE 0 END) as successes,
                AVG(confidence) as avg_confidence
            FROM extracted_fields
            GROUP BY field_name
            ORDER BY successes DESC
        """
        
        df = pd.read_sql_query(query, conn)
        
        print("\n--- FIELD EXTRACTION ANALYSIS (26 Required Fields) ---")
        
        # Define all 26 required fields
        required_fields = [
            "presentation_mode", "gain", "sea_clutter", "rain_clutter", "tune",
            "heading", "speed", "cog", "sog", "position", "position_source",
            "range", "range_rings", "cursor_position", "set", "drift",
            "vector", "vector_duration", "cpa_limit", "tcpa_limit",
            "vrm1", "vrm2", "index_line_rng", "index_line_brg", "ais_on_off", "depth"
        ]
        
        table_data = []
        for i, field in enumerate(required_fields, 1):
            if field in df['field_name'].values:
                row = df[df['field_name'] == field].iloc[0]
                success_rate = row['successes'] / row['attempts'] * 100 if row['attempts'] > 0 else 0
                status = "GOOD" if success_rate >= 80 else "NEEDS WORK" if success_rate >= 50 else "POOR"
                
                table_data.append([
                    i,
                    field,
                    f"{success_rate:.1f}%",
                    f"{row['successes']}/{row['attempts']}",
                    f"{row['avg_confidence']:.2f}" if pd.notna(row['avg_confidence']) else "N/A",
                    status
                ])
            else:
                table_data.append([i, field, "0.0%", "0/0", "N/A", "NOT EXTRACTED"])
        
        print(tabulate(
            table_data,
            headers=["#", "Field Name", "Success Rate", "Count", "Avg Conf", "Status"],
            tablefmt="grid"
        ))
        
        conn.close()
    
    def generate_milestone_report(self):
        """Generate milestone achievement report."""
        conn = sqlite3.connect(self.db_path)
        
        # Count images meeting Milestone 1 (21+ fields)
        query = """
            SELECT 
                COUNT(DISTINCT e.extraction_id) as total_images,
                SUM(CASE 
                    WHEN field_count >= 21 THEN 1 
                    ELSE 0 
                END) as milestone_achieved
            FROM (
                SELECT 
                    e.extraction_id,
                    COUNT(DISTINCT ef.field_name) as field_count
                FROM extractions e
                LEFT JOIN extracted_fields ef ON e.extraction_id = ef.extraction_id
                WHERE ef.field_value IS NOT NULL
                GROUP BY e.extraction_id
            ) as counts
            JOIN extractions e ON counts.extraction_id = e.extraction_id
        """
        
        result = pd.read_sql_query(query, conn).iloc[0]
        
        print("\n--- MILESTONE ACHIEVEMENT STATUS ---")
        print(f"Milestone 1: Extract 21+ data points from radar images")
        print(f"Status: {'ACHIEVED' if result['milestone_achieved'] > 0 else 'NOT ACHIEVED'}")
        print(f"Images with 21+ fields: {result['milestone_achieved']}/{result['total_images']}")
        print(f"Success Rate: {result['milestone_achieved']/result['total_images']*100:.1f}%")
        
        if result['milestone_achieved'] > 0:
            print("\n[MILESTONE 1 ACHIEVED - $2,000 PAYMENT DUE]")
        
        conn.close()
    
    def export_to_csv(self):
        """Export all data to CSV format."""
        conn = sqlite3.connect(self.db_path)
        
        # Main extraction data
        query = """
            SELECT 
                e.extraction_id,
                e.filename,
                e.radar_type,
                e.overall_confidence,
                e.extraction_status,
                e.extraction_timestamp,
                ef.field_name,
                ef.field_value,
                ef.confidence as field_confidence
            FROM extractions e
            LEFT JOIN extracted_fields ef ON e.extraction_id = ef.extraction_id
            ORDER BY e.extraction_id, ef.field_name
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Pivot to wide format (one row per image)
        pivot_df = df.pivot_table(
            index=['extraction_id', 'filename', 'radar_type', 'overall_confidence', 'extraction_status'],
            columns='field_name',
            values='field_value',
            aggfunc='first'
        ).reset_index()
        
        # Save to CSV
        csv_filename = f"radar_extractions_{self.report_timestamp}.csv"
        pivot_df.to_csv(csv_filename, index=False)
        print(f"\nExported to CSV: {csv_filename}")
        
        conn.close()
    
    def export_to_html(self):
        """Export data to HTML format with styling."""
        conn = sqlite3.connect(self.db_path)
        
        # Get summary data
        summary_query = """
            SELECT 
                e.extraction_id,
                e.filename,
                e.radar_type,
                e.overall_confidence,
                COUNT(DISTINCT ef.field_name) as fields_extracted
            FROM extractions e
            LEFT JOIN extracted_fields ef ON e.extraction_id = ef.extraction_id
            WHERE ef.field_value IS NOT NULL
            GROUP BY e.extraction_id
        """
        
        df = pd.read_sql_query(summary_query, conn)
        
        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>One Stop Portal - Radar Extraction Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .success {{ color: green; font-weight: bold; }}
        .partial {{ color: orange; font-weight: bold; }}
        .failed {{ color: red; font-weight: bold; }}
        .milestone {{ background-color: #2ecc71; color: white; padding: 10px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>ONE STOP PORTAL - Radar Data Extraction Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Summary Statistics</h2>
    <ul>
        <li>Total Images Processed: {len(df)}</li>
        <li>Average Confidence: {df['overall_confidence'].mean():.2%}</li>
        <li>Images with 21+ fields: {len(df[df['fields_extracted'] >= 21])}</li>
    </ul>
    
    <div class="milestone">
        <h3>Milestone 1 Status: {'ACHIEVED' if len(df[df['fields_extracted'] >= 21]) > 0 else 'NOT ACHIEVED'}</h3>
        <p>Requirement: Extract 21+ data points from radar images</p>
        <p>Achievement: {len(df[df['fields_extracted'] >= 21])}/{len(df)} images meet requirement</p>
    </div>
    
    <h2>Extraction Details</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Filename</th>
            <th>Radar Type</th>
            <th>Confidence</th>
            <th>Fields Extracted</th>
            <th>Milestone 1</th>
        </tr>
"""
        
        for _, row in df.iterrows():
            milestone_met = "YES" if row['fields_extracted'] >= 21 else "NO"
            milestone_class = "success" if row['fields_extracted'] >= 21 else "failed"
            
            html_content += f"""
        <tr>
            <td>{row['extraction_id']}</td>
            <td>{row['filename']}</td>
            <td>{row['radar_type']}</td>
            <td>{row['overall_confidence']:.1%}</td>
            <td>{row['fields_extracted']}/26</td>
            <td class="{milestone_class}">{milestone_met}</td>
        </tr>
"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        # Save HTML file
        html_filename = f"radar_report_{self.report_timestamp}.html"
        with open(html_filename, 'w') as f:
            f.write(html_content)
        
        print(f"Exported to HTML: {html_filename}")
        conn.close()
    
    def export_to_excel(self):
        """Export to Excel format with multiple sheets."""
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
            
            conn = sqlite3.connect(self.db_path)
            
            # Create Excel writer
            excel_filename = f"radar_extraction_report_{self.report_timestamp}.xlsx"
            
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # Sheet 1: Summary
                summary_df = pd.read_sql_query("""
                    SELECT 
                        COUNT(DISTINCT extraction_id) as 'Total Images',
                        AVG(overall_confidence) as 'Average Confidence',
                        COUNT(DISTINCT radar_type) as 'Radar Types'
                    FROM extractions
                """, conn)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 2: Detailed Extractions
                detail_query = """
                    SELECT 
                        extraction_id as 'ID',
                        filename as 'Filename',
                        radar_type as 'Radar Type',
                        overall_confidence as 'Confidence',
                        extraction_status as 'Status'
                    FROM extractions
                """
                detail_df = pd.read_sql_query(detail_query, conn)
                detail_df.to_excel(writer, sheet_name='Extractions', index=False)
                
                # Sheet 3: Field Analysis
                field_query = """
                    SELECT 
                        field_name as 'Field',
                        COUNT(*) as 'Attempts',
                        SUM(CASE WHEN field_value IS NOT NULL THEN 1 ELSE 0 END) as 'Successes'
                    FROM extracted_fields
                    GROUP BY field_name
                """
                field_df = pd.read_sql_query(field_query, conn)
                field_df['Success Rate'] = (field_df['Successes'] / field_df['Attempts'] * 100).round(1)
                field_df.to_excel(writer, sheet_name='Field Analysis', index=False)
            
            print(f"Exported to Excel: {excel_filename}")
            conn.close()
            
        except ImportError:
            print("Note: Install openpyxl for Excel export: pip install openpyxl")

def main():
    """Generate all client reports."""
    # Check if database exists
    db_path = "radar_extraction_system.db"
    
    if not os.path.exists(db_path):
        print("Error: Database not found. Please run extraction first.")
        return
    
    # Generate reports
    generator = ClientReportGenerator(db_path)
    generator.generate_all_reports()
    
    print("\nAll reports have been generated in the current directory.")
    print("Files created:")
    print("- extraction_details_[timestamp].txt")
    print("- radar_extractions_[timestamp].csv")
    print("- radar_report_[timestamp].html")
    print("- radar_extraction_report_[timestamp].xlsx (if openpyxl installed)")

if __name__ == "__main__":
    main()