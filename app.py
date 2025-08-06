# app.py - Client Dashboard for Radar Data Extraction System (Azure Compatible)
import streamlit as st
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing components and web helpers
try:
    from web_helpers import get_web_helper
    from radar_extraction_architecture import RADAR_FIELDS, RadarType
    from radar_database_management import DatabaseManager, ResultManager, ReviewStatus
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Radar Data Extraction System",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    
    /* Status indicators */
    .status-good { color: #10b981; }
    .status-warning { color: #f59e0b; }
    .status-error { color: #ef4444; }
    
    /* Professional button styling */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_keys' not in st.session_state:
    # Get API keys from environment variables (Azure App Service Configuration)
    st.session_state.api_keys = {
        'anthropic': os.environ.get('ANTHROPIC_API_KEY'),
        'openai': os.environ.get('OPENAI_API_KEY'),
        'google': os.environ.get('GOOGLE_API_KEY')
    }

if 'web_helper' not in st.session_state:
    try:
        st.session_state.web_helper = get_web_helper(st.session_state.api_keys)
        logger.info("Web helper initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize web helper: {e}")
        st.info("Please check your database connection and API keys in Azure Configuration")
        st.stop()

if 'current_user' not in st.session_state:
    st.session_state.current_user = os.environ.get('DEFAULT_USER', 'User')

# Helper function to safely get stats
def safe_get_stats(web_helper, start_date=None, end_date=None):
    """Safely get statistics with error handling."""
    try:
        return web_helper.db_manager.get_statistics(start_date, end_date)
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return None

# Sidebar
with st.sidebar:
    # Logo/Title
    st.markdown("## 🚢 Radar Extraction")
    st.markdown("---")
    
    # Navigation (Note: switch_page might not work in Azure, using different approach)
    
    # System Health
    st.markdown("### System Health")
    col1, col2 = st.columns(2)
    
    # Get current stats
    stats = safe_get_stats(
        st.session_state.web_helper,
        start_date=datetime.now() - timedelta(hours=1)
    )
    
    # API Status (check if keys are configured)
    api_status = "🟢" if any(st.session_state.api_keys.values()) else "🔴"
    with col1:
        st.markdown(f"**API**: {api_status}")
    
    # Database Status
    db_status = "🟢" if stats else "🔴"
    with col2:
        st.markdown(f"**Database**: {db_status}")
    
    # Processing Speed
    avg_time = 0.0
    if stats and stats.get('overall'):
        total_images = stats['overall'].get('total_images', 0) or 0
        if total_images > 0:
            avg_time = stats['overall'].get('avg_processing_time', 0.0) or 0.0
    
    speed_status = "🟢" if avg_time < 5 else "🟡" if avg_time < 10 else "🔴"
    st.markdown(f"**Avg Speed**: {speed_status} {avg_time:.1f}s")
    
    st.markdown("---")
    
    # Azure Deployment Info
    st.markdown("### Deployment Info")
    st.markdown(f"**Environment**: {os.environ.get('ENVIRONMENT', 'Development')}")
    st.markdown(f"**Region**: {os.environ.get('AZURE_REGION', 'Not Set')}")

# Check if we need to display a different page
page = st.session_state.get('page', 'dashboard')

if page == 'upload':
    exec(open('pages/1_Upload.py').read())
elif page == 'review':
    exec(open('pages/2_Review.py').read())
elif page == 'analytics':
    exec(open('pages/3_Analytics.py').read())
elif page == 'export':
    exec(open('pages/4_Export.py').read())
else:
    # Main Dashboard Content
    # Professional header
    st.markdown("""
    <div class="header-container">
        <h1 style='margin:0; padding:0;'>Radar Data Extraction System</h1>
        <p style='margin:0; padding:0; opacity:0.9;'>Automated Maritime Radar Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Indicators
    st.markdown("### 📈 Key Performance Indicators")

    # Get various time-based statistics with error handling
    today_stats = safe_get_stats(
        st.session_state.web_helper,
        start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        end_date=datetime.now()
    )

    week_stats = safe_get_stats(
        st.session_state.web_helper,
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now()
    )

    all_time_stats = safe_get_stats(st.session_state.web_helper)

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        today_count = 0
        week_count = 0
        
        if today_stats and today_stats.get('overall'):
            today_count = today_stats['overall'].get('total_images', 0) or 0
        
        if week_stats and week_stats.get('overall'):
            week_count = week_stats['overall'].get('total_images', 0) or 0
        
        st.metric(
            label="Images Processed Today",
            value=f"{today_count:,}",
            delta=f"{week_count:,} this week"
        )

    with col2:
        accuracy = 0.0
        if all_time_stats and all_time_stats.get('overall'):
            total_images = all_time_stats['overall'].get('total_images', 0) or 0
            successful = all_time_stats['overall'].get('successful', 0) or 0
            if total_images > 0:
                accuracy = (successful / total_images * 100)
        
        st.metric(
            label="Processing Accuracy",
            value=f"{accuracy:.1f}%",
            delta="Overall success rate"
        )

    with col3:
        pending = 0
        if all_time_stats and all_time_stats.get('overall'):
            pending = all_time_stats['overall'].get('pending_review', 0) or 0
        
        st.metric(
            label="Awaiting Review",
            value=pending,
            delta="Requires attention" if pending > 10 else "Up to date",
            delta_color="inverse"
        )

    with col4:
        today_avg_time = 0.0
        if today_stats and today_stats.get('overall'):
            total_images = today_stats['overall'].get('total_images', 0) or 0
            if total_images > 0:
                today_avg_time = today_stats['overall'].get('avg_processing_time', 0.0) or 0.0
        
        st.metric(
            label="Avg Processing Time",
            value=f"{today_avg_time:.1f}s",
            delta="Per image today"
        )

    st.markdown("---")

    # Processing Overview Section
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📊 Processing Volume (Last 30 Days)")
        
        # Get data for last 30 days - simplified for better performance
        analytics_data = st.session_state.web_helper.get_analytics_data(days=30)
        
        if analytics_data and analytics_data.get('daily_trend'):
            trend_df = pd.DataFrame(analytics_data['daily_trend'])
            
            if not trend_df.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=trend_df['date'],
                    y=trend_df['total'],
                    mode='lines',
                    name='Total Processed',
                    line=dict(color='#3b82f6', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.1)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=trend_df['date'],
                    y=trend_df['successful'],
                    mode='lines',
                    name='Successful',
                    line=dict(color='#10b981', width=2)
                ))
                
                fig.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True,
                    xaxis_title="Date",
                    yaxis_title="Number of Images",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No processing data available for the last 30 days")
        else:
            st.info("Processing volume data is being calculated...")

    with col2:
        st.markdown("### 🎯 Field Extraction Quality")
        
        # Get field statistics
        if all_time_stats and all_time_stats.get('field_statistics'):
            # Get top performing fields
            field_stats = sorted(
                all_time_stats['field_statistics'], 
                key=lambda x: x.get('avg_confidence', 0) or 0, 
                reverse=True
            )[:10]
            
            if field_stats:
                fields = [f['field_name'].replace('_', ' ').title() for f in field_stats]
                confidences = [(f.get('avg_confidence', 0) or 0) * 100 for f in field_stats]
                
                # Create horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=confidences,
                    y=fields,
                    orientation='h',
                    marker=dict(
                        color=confidences,
                        colorscale='Blues',
                        showscale=False
                    ),
                    text=[f"{c:.0f}%" for c in confidences],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    height=350,
                    margin=dict(l=0, r=50, t=0, b=0),
                    xaxis_title="Average Confidence",
                    xaxis=dict(range=[0, 105]),
                    yaxis=dict(autorange="reversed")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No field extraction data available yet")
        else:
            st.info("Field quality data is being calculated...")

    # Quick Actions Section
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📤 Upload Images", use_container_width=True, type="primary"):
            st.session_state.page = "upload"
            st.rerun()
        st.caption("Process new radar images")

    with col2:
        pending_count = 0
        if all_time_stats and all_time_stats.get('overall'):
            pending_count = all_time_stats['overall'].get('pending_review', 0) or 0
        
        button_label = f"👁️ Review ({pending_count})" if pending_count > 0 else "👁️ Review"
        if st.button(button_label, use_container_width=True, type="primary" if pending_count > 0 else "secondary"):
            st.session_state.page = "review"
            st.rerun()
        st.caption("Review flagged extractions")

    with col3:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.page = "analytics"
            st.rerun()
        st.caption("Detailed performance metrics")

    with col4:
        if st.button("📥 Export Data", use_container_width=True):
            st.session_state.page = "export"
            st.rerun()
        st.caption("Download extraction results")

    # Recent Activity
    st.markdown("---")
    st.markdown("### 🕒 Recent Activity")

    try:
        # Get recent extractions
        recent_extractions = st.session_state.web_helper.get_recent_extractions(limit=5)

        if recent_extractions:
            activity_data = []
            for ext in recent_extractions:
                # Handle timestamp - could be string or datetime
                if isinstance(ext.get('extraction_timestamp'), str):
                    # Parse string timestamp
                    try:
                        timestamp = datetime.fromisoformat(ext['extraction_timestamp'].replace('Z', '+00:00'))
                    except:
                        try:
                            timestamp = datetime.strptime(ext['extraction_timestamp'], "%Y-%m-%d %H:%M:%S.%f")
                        except:
                            timestamp = datetime.now()
                else:
                    timestamp = ext.get('extraction_timestamp', datetime.now())
                
                # Ensure timestamp is timezone-naive for comparison
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)
                
                time_ago = datetime.now() - timestamp
                
                if time_ago.days > 0:
                    time_str = f"{time_ago.days}d ago"
                elif time_ago.seconds > 3600:
                    time_str = f"{time_ago.seconds // 3600}h ago"
                else:
                    time_str = f"{time_ago.seconds // 60}m ago"
                
                confidence = ext.get('overall_confidence', 0) or 0
                status_icon = "✅" if confidence > 0.8 else "⚠️" if confidence > 0.5 else "❌"
                
                filename = ext.get('filename', 'Unknown')
                display_name = filename[:30] + "..." if len(filename) > 30 else filename
                
                activity_data.append({
                    "Status": status_icon,
                    "File": display_name,
                    "Type": ext.get('radar_type', 'Unknown'),
                    "Confidence": f"{confidence:.0%}",
                    "Time": time_str
                })
            
            if activity_data:
                activity_df = pd.DataFrame(activity_data)
                st.dataframe(activity_df, use_container_width=True, hide_index=True, height=250)
            else:
                st.info("No recent activity. Upload images to get started!")
        else:
            st.info("No recent activity. Upload images to get started!")
    except Exception as e:
        logger.error(f"Error displaying recent activity: {e}")
        st.info("Recent activity is being loaded...")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**System Version**: 2.0 (Azure)")

    with col2:
        try:
            storage_stats = st.session_state.web_helper.get_storage_stats()
            if storage_stats:
                st.markdown(f"**Storage Used**: {storage_stats.get('total_size_mb', 0):.1f} MB")
            else:
                st.markdown("**Storage**: Calculating...")
        except:
            st.markdown("**Storage**: N/A")

    with col3:
        st.markdown(f"**Last Updated**: {datetime.now().strftime('%H:%M:%S')}")

    # Auto-refresh option
    if st.checkbox("Auto-refresh (30s)", key="auto_refresh"):
        st.markdown("*Dashboard will refresh every 30 seconds*")
        st.rerun()  # Will cause refresh after delay

# Run the app
if __name__ == "__main__":
    # Azure App Service will set the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    # Note: Streamlit runs its own server, this is just for reference