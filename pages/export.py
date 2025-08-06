# pages/4_Export.py - Professional Export & Reporting Interface (Azure Compatible)

import streamlit as st
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import json
import io
import zipfile
import logging
from typing import Dict, Any, List
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from web_helpers import get_web_helper
    from radar_extraction_architecture import RADAR_FIELDS
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Export & Reports - Radar Extraction",
    page_icon="📥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Professional color scheme */
    :root {
        --primary-color: #1e40af;
        --secondary-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --dark-bg: #1e293b;
        --light-bg: #f8fafc;
    }
    
    /* Main container */
    .main {
        padding: 1rem 2rem;
        background-color: var(--light-bg);
    }
    
    /* Professional header */
    .export-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .export-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    .export-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-size: 1.1rem;
    }
    
    /* Export option cards */
    .export-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid var(--secondary-color);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 1rem;
    }
    
    .export-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    /* Filter section */
    .filter-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Preview section */
    .preview-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Statistics box */
    .stats-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid var(--primary-color);
        margin-bottom: 1rem;
    }
    
    /* Export button styling */
    .export-button {
        background: var(--success-color);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .export-button:hover {
        background: #059669;
        transform: translateY(-1px);
    }
    
    /* Format selector */
    .format-selector {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid var(--secondary-color);
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid var(--secondary-color);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Success message */
    .success-message {
        background: var(--success-color);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'web_helper' not in st.session_state:
    try:
        st.session_state.web_helper = get_web_helper({
            'anthropic': os.environ.get('ANTHROPIC_API_KEY'),
            'openai': os.environ.get('OPENAI_API_KEY'),
            'google': os.environ.get('GOOGLE_API_KEY')
        })
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.stop()

if 'export_settings' not in st.session_state:
    st.session_state.export_settings = {
        'format': 'csv',
        'include_metadata': True,
        'include_confidence': True,
        'only_successful': False
    }

# Helper functions
def get_export_statistics(start_date: datetime, end_date: datetime) -> Dict:
    """Get statistics for the export period."""
    try:
        stats = st.session_state.web_helper.db_manager.get_statistics(start_date, end_date)
        return stats.get('overall', {})
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return {}

def prepare_export_data(start_date: datetime, end_date: datetime, filters: Dict) -> pd.DataFrame:
    """Prepare data for export based on filters."""
    try:
        with st.session_state.web_helper.db_manager.get_connection() as conn:
            # Build query based on filters
            query = """
                SELECT 
                    e.extraction_id,
                    e.filename,
                    e.radar_type,
                    e.extraction_status,
                    e.overall_confidence,
                    e.processing_time,
                    e.extraction_timestamp,
                    e.requires_review,
                    ef.field_name,
                    ef.field_value,
                    ef.confidence as field_confidence,
                    ef.extraction_method,
                    ef.is_valid
                FROM extractions e
                LEFT JOIN extracted_fields ef ON e.extraction_id = ef.extraction_id
                WHERE e.extraction_timestamp BETWEEN %s AND %s
            """
            
            params = [start_date, end_date]
            
            # Apply filters
            if filters.get('only_successful'):
                query += " AND e.extraction_status = 'success'"
            
            if filters.get('min_confidence'):
                query += " AND e.overall_confidence >= %s"
                params.append(filters['min_confidence'])
            
            if filters.get('radar_types'):
                placeholders = ','.join(['%s'] * len(filters['radar_types']))
                query += f" AND e.radar_type IN ({placeholders})"
                params.extend(filters['radar_types'])
            
            query += " ORDER BY e.extraction_timestamp DESC, ef.field_name"
            
            df = pd.read_sql_query(query, conn, params=params)
            return df
            
    except Exception as e:
        logger.error(f"Error preparing export data: {e}")
        return pd.DataFrame()

def create_summary_report(df: pd.DataFrame) -> str:
    """Create a text summary report."""
    if df.empty:
        return "No data available for the selected period."
    
    # Get unique extractions
    unique_extractions = df['extraction_id'].nunique()
    
    # Calculate statistics
    avg_confidence = df.groupby('extraction_id')['overall_confidence'].first().mean()
    success_rate = (df[df['extraction_status'] == 'success']['extraction_id'].nunique() / unique_extractions * 100) if unique_extractions > 0 else 0
    
    # Field statistics
    field_stats = df.groupby('field_name').agg({
        'field_value': 'count',
        'field_confidence': 'mean'
    }).round(3)
    
    report = f"""
RADAR DATA EXTRACTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

SUMMARY STATISTICS
------------------
Total Extractions: {unique_extractions}
Average Confidence: {avg_confidence:.2%}
Success Rate: {success_rate:.1f}%
Period: {df['extraction_timestamp'].min()} to {df['extraction_timestamp'].max()}

RADAR TYPES PROCESSED
---------------------
{df.groupby('radar_type')['extraction_id'].nunique().to_string()}

TOP PERFORMING FIELDS
--------------------
{field_stats.nlargest(10, 'field_confidence').to_string()}

EXTRACTION METHODS USED
----------------------
{df.groupby('extraction_method')['extraction_id'].count().to_string()}

================================================================================
    """
    return report

def export_to_excel(df: pd.DataFrame) -> bytes:
    """Export data to Excel with multiple sheets."""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main data sheet
        if not df.empty:
            # Pivot data for better readability
            pivot_df = df.pivot_table(
                index=['extraction_id', 'filename', 'radar_type', 'extraction_timestamp'],
                columns='field_name',
                values='field_value',
                aggfunc='first'
            ).reset_index()
            
            pivot_df.to_excel(writer, sheet_name='Extraction Data', index=False)
            
            # Summary statistics sheet
            summary_data = {
                'Metric': ['Total Extractions', 'Average Confidence', 'Success Rate', 'Fields Extracted'],
                'Value': [
                    df['extraction_id'].nunique(),
                    f"{df.groupby('extraction_id')['overall_confidence'].first().mean():.2%}",
                    f"{(df[df['extraction_status'] == 'success']['extraction_id'].nunique() / df['extraction_id'].nunique() * 100):.1f}%",
                    df['field_name'].nunique()
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Field performance sheet
            field_perf = df.groupby('field_name').agg({
                'field_value': 'count',
                'field_confidence': 'mean',
                'is_valid': lambda x: (x == True).sum() / len(x) * 100
            }).round(2)
            field_perf.columns = ['Extraction Count', 'Avg Confidence', 'Validation Rate (%)']
            field_perf.to_excel(writer, sheet_name='Field Performance')
    
    output.seek(0)
    return output.read()

def create_zip_export(df: pd.DataFrame, include_images: bool = False) -> bytes:
    """Create a ZIP file with multiple export formats."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add CSV
        csv_data = df.to_csv(index=False)
        zip_file.writestr(f'extraction_data_{datetime.now():%Y%m%d}.csv', csv_data)
        
        # Add JSON
        json_data = df.to_json(orient='records', indent=2)
        zip_file.writestr(f'extraction_data_{datetime.now():%Y%m%d}.json', json_data)
        
        # Add summary report
        report = create_summary_report(df)
        zip_file.writestr(f'summary_report_{datetime.now():%Y%m%d}.txt', report)
        
        # Add metadata
        metadata = {
            'export_date': datetime.now().isoformat(),
            'total_records': len(df),
            'unique_extractions': df['extraction_id'].nunique() if not df.empty else 0,
            'date_range': {
                'start': str(df['extraction_timestamp'].min()) if not df.empty else None,
                'end': str(df['extraction_timestamp'].max()) if not df.empty else None
            }
        }
        zip_file.writestr('metadata.json', json.dumps(metadata, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer.read()

# Professional Header
st.markdown("""
<div class="export-header">
    <h1>📥 Export & Reporting Center</h1>
    <p>Generate professional reports and export extraction data</p>
</div>
""", unsafe_allow_html=True)

# Quick Export Options
st.markdown("## Quick Export")

quick_cols = st.columns(4)

with quick_cols[0]:
    if st.button("📊 Today's Report", use_container_width=True):
        today = datetime.now().date()
        df = prepare_export_data(
            datetime.combine(today, datetime.min.time()),
            datetime.combine(today, datetime.max.time()),
            {}
        )
        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Today's Data",
                csv,
                f"radar_export_today_{today}.csv",
                "text/csv"
            )

with quick_cols[1]:
    if st.button("📈 Last 7 Days", use_container_width=True):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        df = prepare_export_data(start_date, end_date, {})
        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Week's Data",
                csv,
                f"radar_export_week_{datetime.now():%Y%m%d}.csv",
                "text/csv"
            )

with quick_cols[2]:
    if st.button("📅 Last Month", use_container_width=True):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        df = prepare_export_data(start_date, end_date, {})
        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Month's Data",
                csv,
                f"radar_export_month_{datetime.now():%Y%m%d}.csv",
                "text/csv"
            )

with quick_cols[3]:
    if st.button("📦 Full Export", use_container_width=True):
        df = prepare_export_data(
            datetime(2000, 1, 1),
            datetime.now(),
            {}
        )
        if not df.empty:
            zip_data = create_zip_export(df)
            st.download_button(
                "⬇️ Download Complete Archive",
                zip_data,
                f"radar_export_complete_{datetime.now():%Y%m%d}.zip",
                "application/zip"
            )

# Custom Export Section
st.markdown("---")
st.markdown("## Custom Export")

# Filters
with st.container():
    st.markdown("###  Export Filters")
    
    filter_cols = st.columns(3)
    
    with filter_cols[0]:
        st.markdown("#### Date Range")
        date_option = st.selectbox(
            "Select period",
            ["Last 7 Days", "Last 30 Days", "Last Quarter", "Year to Date", "Custom Range"],
            key="date_filter"
        )
        
        if date_option == "Custom Range":
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
            end_date = st.date_input("End Date", datetime.now())
        else:
            end_date = datetime.now()
            if date_option == "Last 7 Days":
                start_date = end_date - timedelta(days=7)
            elif date_option == "Last 30 Days":
                start_date = end_date - timedelta(days=30)
            elif date_option == "Last Quarter":
                start_date = end_date - timedelta(days=90)
            else:  # Year to Date
                start_date = datetime(end_date.year, 1, 1)
    
    with filter_cols[1]:
        st.markdown("#### Data Filters")
        
        only_successful = st.checkbox("Only Successful Extractions", value=False)
        
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0,
            max_value=100,
            value=0,
            step=10,
            format="%d%%"
        ) / 100
        
        include_review = st.checkbox("Include Items Pending Review", value=True)
    
    with filter_cols[2]:
        st.markdown("#### Radar Types")
        
        # Get available radar types
        try:
            with st.session_state.web_helper.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT radar_type FROM extractions")
                available_types = [row[0] for row in cursor.fetchall()]
        except:
            available_types = []
        
        if available_types:
            selected_types = st.multiselect(
                "Select radar types",
                available_types,
                default=available_types
            )
        else:
            selected_types = []
            st.info("No radar types available")

# Export Format Selection
st.markdown("### 📄 Export Format")

format_cols = st.columns(4)

with format_cols[0]:
    export_format = st.radio(
        "Select format",
        ["CSV", "Excel", "JSON", "ZIP Archive"],
        key="format_select"
    )

with format_cols[1]:
    st.markdown("#### Include Options")
    include_metadata = st.checkbox("Include Metadata", value=True)
    include_confidence = st.checkbox("Include Confidence Scores", value=True)
    include_methods = st.checkbox("Include Extraction Methods", value=False)

with format_cols[2]:
    st.markdown("#### Field Selection")
    field_option = st.radio(
        "Fields to export",
        ["All Fields", "Critical Fields Only", "Custom Selection"],
        key="field_select"
    )
    
    if field_option == "Custom Selection":
        selected_fields = st.multiselect(
            "Select fields",
            list(RADAR_FIELDS.keys()),
            default=['heading', 'speed', 'position']
        )

with format_cols[3]:
    st.markdown("#### Export Details")
    export_name = st.text_input(
        "Export filename",
        value=f"radar_export_{datetime.now():%Y%m%d}",
        key="export_name"
    )
    
    add_timestamp = st.checkbox("Add timestamp to filename", value=True)

# Preview Section
st.markdown("---")
st.markdown("### 👁️ Data Preview")

# Prepare filters
filters = {
    'only_successful': only_successful,
    'min_confidence': min_confidence if min_confidence > 0 else None,
    'radar_types': selected_types if selected_types else None
}

# Get preview data
with st.spinner("Loading preview..."):
    preview_df = prepare_export_data(
        datetime.combine(start_date, datetime.min.time()) if isinstance(start_date, datetime) else start_date,
        datetime.combine(end_date, datetime.max.time()) if isinstance(end_date, datetime) else end_date,
        filters
    )

if not preview_df.empty:
    # Show statistics
    col1, col2, col3, col4 = st.columns(4)
    
    unique_extractions = preview_df['extraction_id'].nunique()
    total_fields = preview_df['field_name'].nunique()
    avg_confidence = preview_df.groupby('extraction_id')['overall_confidence'].first().mean()
    
    with col1:
        st.metric("Extractions", f"{unique_extractions:,}")
    with col2:
        st.metric("Total Records", f"{len(preview_df):,}")
    with col3:
        st.metric("Fields", total_fields)
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Show preview
    with st.expander("📊 Preview Data (First 100 rows)", expanded=True):
        # Pivot for better display
        if 'field_name' in preview_df.columns and 'field_value' in preview_df.columns:
            preview_pivot = preview_df.pivot_table(
                index=['extraction_id', 'filename', 'radar_type'],
                columns='field_name',
                values='field_value',
                aggfunc='first'
            ).reset_index()
            
            st.dataframe(
                preview_pivot.head(100),
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(preview_df.head(100), use_container_width=True, height=400)
    
    # Export button
    st.markdown("---")
    st.markdown("### 💾 Generate Export")
    
    export_col1, export_col2, export_col3 = st.columns([1, 2, 1])
    
    with export_col2:
        if st.button("🚀 Generate Export File", type="primary", use_container_width=True):
            with st.spinner(f"Generating {export_format} export..."):
                try:
                    # Generate filename
                    if add_timestamp:
                        filename = f"{export_name}_{datetime.now():%Y%m%d_%H%M%S}"
                    else:
                        filename = export_name
                    
                    # Generate export based on format
                    if export_format == "CSV":
                        export_data = preview_df.to_csv(index=False)
                        mime_type = "text/csv"
                        file_extension = "csv"
                    
                    elif export_format == "Excel":
                        export_data = export_to_excel(preview_df)
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        file_extension = "xlsx"
                    
                    elif export_format == "JSON":
                        export_data = preview_df.to_json(orient='records', indent=2)
                        mime_type = "application/json"
                        file_extension = "json"
                    
                    else:  # ZIP Archive
                        export_data = create_zip_export(preview_df)
                        mime_type = "application/zip"
                        file_extension = "zip"
                    
                    # Success message
                    st.markdown("""
                    <div class="success-message">
                        ✅ Export generated successfully!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download button
                    st.download_button(
                        label=f"⬇️ Download {export_format} File ({len(export_data) / 1024:.1f} KB)",
                        data=export_data,
                        file_name=f"{filename}.{file_extension}",
                        mime=mime_type,
                        use_container_width=True
                    )
                    
                    # Log export
                    logger.info(f"Export generated: {filename}.{file_extension}")
                    
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
                    logger.error(f"Export error: {e}")
    
    # Additional export options
    with st.expander("📊 Additional Export Options"):
        st.markdown("#### 📧 Email Export")
        email_cols = st.columns([2, 1])
        with email_cols[0]:
            recipient_email = st.text_input(
                "Recipient email",
                placeholder="user@example.com"
            )
        with email_cols[1]:
            if st.button("Send via Email", disabled=True):
                st.info("Email functionality coming soon")
        
        st.markdown("#### 📅 Scheduled Exports")
        schedule_cols = st.columns([2, 1])
        with schedule_cols[0]:
            schedule_option = st.selectbox(
                "Schedule frequency",
                ["Daily", "Weekly", "Monthly"],
                disabled=True
            )
        with schedule_cols[1]:
            if st.button("Setup Schedule", disabled=True):
                st.info("Scheduling feature coming soon")
        
        st.markdown("#### 🔗 API Export")
        st.code(f"""
# API Endpoint (Coming Soon)
GET /api/v1/export?start_date={start_date}&end_date={end_date}&format={export_format.lower()}
Authorization: Bearer YOUR_API_KEY
        """, language="bash")

else:
    st.warning("No data available for the selected filters")

# Sidebar with help and tips
with st.sidebar:
    st.markdown("### 📚 Export Guide")
    
    st.markdown("""
    #### Format Descriptions
    
    
    """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use filters to reduce file size
    - Excel format includes summaries
    - ZIP archives include all formats
    - Check preview before exporting
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Export Statistics")
    if not preview_df.empty:
        st.metric("Ready to Export", f"{unique_extractions} items")
        st.metric("Date Range", f"{(end_date - start_date).days} days")
        st.metric("File Size (est.)", f"{len(preview_df) * 0.1:.1f} KB")
    
    st.markdown("---")
    
    # Environment info
    st.markdown("### 🌍 Environment")
    st.markdown(f"**Mode**: {os.environ.get('ENVIRONMENT', 'Production')}")
    st.markdown(f"**Database**: PostgreSQL")
    if os.environ.get('AZURE_REGION'):
        st.markdown(f"**Region**: {os.environ.get('AZURE_REGION')}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

with col2:
    st.markdown(f"**System Version**: 2.0 Professional")

with col3:
    st.markdown(f"**Export Module**: v1.0")