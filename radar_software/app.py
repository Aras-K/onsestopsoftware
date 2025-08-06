# app.py - Client Dashboard for Radar Data Extraction System
import streamlit as st
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add your existing modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing components and web helpers
from web_helpers import get_web_helper
from radar_extraction_architecture import RADAR_FIELDS, RadarType
from radar_database_management import DatabaseManager, ResultManager, ReviewStatus

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
    st.session_state.api_keys = {
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'openai': os.getenv('OPENAI_API_KEY'),
        'google': os.getenv('GOOGLE_API_KEY')
    }

if 'web_helper' not in st.session_state:
    st.session_state.web_helper = get_web_helper(st.session_state.api_keys)

if 'current_user' not in st.session_state:
    st.session_state.current_user = "User"

# Sidebar
with st.sidebar:
    # Logo/Title
    st.markdown("## 🚢 Radar Extraction")
    st.markdown("---")
    
    # Navigation
    st.markdown("### Navigation")
    pages = {
        "🏠 Dashboard": "app.py",
        "📤 Upload & Process": "pages/1_Upload.py", 
        "👁️ Review Queue": "pages/2_Review.py",
        "📊 Analytics": "pages/3_Analytics.py",
        "📥 Export Data": "pages/4_Export.py"
    }
    
    for label, page in pages.items():
        if st.button(label, use_container_width=True, key=f"nav_{label}"):
            st.switch_page(page)
    
    st.markdown("---")
    
    # System Health
    st.markdown("### System Health")
    col1, col2 = st.columns(2)
    
    # Get current stats
    stats = st.session_state.web_helper.db_manager.get_statistics(
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

# Main content
# Professional header
st.markdown("""
<div class="header-container">
    <h1 style='margin:0; padding:0;'>Radar Data Extraction System</h1>
    <p style='margin:0; padding:0; opacity:0.9;'>Automated Maritime Radar Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# Key Performance Indicators
st.markdown("### 📈 Key Performance Indicators")

# Get various time-based statistics
today_stats = st.session_state.web_helper.db_manager.get_statistics(
    start_date=datetime.now().replace(hour=0, minute=0, second=0),
    end_date=datetime.now()
)

week_stats = st.session_state.web_helper.db_manager.get_statistics(
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
)

all_time_stats = st.session_state.web_helper.db_manager.get_statistics()

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
    
    # Get data for last 30 days
    dates = []
    processed_counts = []
    success_counts = []
    review_counts = []
    
    for i in range(30, -1, -1):
        date = datetime.now() - timedelta(days=i)
        stats = st.session_state.web_helper.db_manager.get_statistics(
            start_date=date.replace(hour=0, minute=0, second=0),
            end_date=date.replace(hour=23, minute=59, second=59)
        )
        
        if i % 3 == 0:  # Show every 3rd day for cleaner chart
            dates.append(date.strftime("%b %d"))
            if stats and stats.get('overall'):
                processed_counts.append(stats['overall'].get('total_images', 0) or 0)
                success_counts.append(stats['overall'].get('successful', 0) or 0)
                review_counts.append(stats['overall'].get('pending_review', 0) or 0)
            else:
                processed_counts.append(0)
                success_counts.append(0) 
                review_counts.append(0)
    
    # Create area chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=processed_counts,
        mode='lines',
        name='Total Processed',
        line=dict(color='#3b82f6', width=3),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=success_counts,
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

# Quick Actions Section
st.markdown("---")
st.markdown("### 🚀 Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📤 Upload Images", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Upload.py")
    st.caption("Process new radar images")

with col2:
    pending_count = 0
    if all_time_stats and all_time_stats.get('overall'):
        pending_count = all_time_stats['overall'].get('pending_review', 0) or 0
    
    button_label = f"👁️ Review ({pending_count})" if pending_count > 0 else "👁️ Review"
    if st.button(button_label, use_container_width=True, type="primary" if pending_count > 0 else "secondary"):
        st.switch_page("pages/2_Review.py")
    st.caption("Review flagged extractions")

with col3:
    if st.button("📊 View Analytics", use_container_width=True):
        st.switch_page("pages/3_Analytics.py")
    st.caption("Detailed performance metrics")

with col4:
    if st.button("📥 Export Data", use_container_width=True):
        st.switch_page("pages/4_Export.py")
    st.caption("Download extraction results")

# Recent Activity
st.markdown("---")
st.markdown("### 🕒 Recent Activity")

# Get recent extractions
recent_extractions = st.session_state.web_helper.db_manager.get_recent_extractions(limit=5)

if recent_extractions:
    activity_data = []
    for ext in recent_extractions:
        # Handle timestamp - could be string or datetime
        if isinstance(ext['extraction_timestamp'], str):
            # Parse string timestamp
            try:
                timestamp = datetime.fromisoformat(ext['extraction_timestamp'].replace('Z', '+00:00'))
            except:
                timestamp = datetime.strptime(ext['extraction_timestamp'], "%Y-%m-%d %H:%M:%S.%f")
        else:
            timestamp = ext['extraction_timestamp']
        
        time_ago = datetime.now() - timestamp
        
        if time_ago.days > 0:
            time_str = f"{time_ago.days}d ago"
        elif time_ago.seconds > 3600:
            time_str = f"{time_ago.seconds // 3600}h ago"
        else:
            time_str = f"{time_ago.seconds // 60}m ago"
        
        confidence = ext.get('overall_confidence', 0) or 0
        status_icon = "✅" if confidence > 0.8 else "⚠️" if confidence > 0.5 else "❌"
        
        activity_data.append({
            "Status": status_icon,
            "File": ext['filename'][:30] + "..." if len(ext['filename']) > 30 else ext['filename'],
            "Type": ext.get('radar_type', 'Unknown'),
            "Confidence": f"{confidence:.0%}",
            "Time": time_str
        })
    
    activity_df = pd.DataFrame(activity_data)
    st.dataframe(activity_df, use_container_width=True, hide_index=True, height=250)
else:
    st.info("No recent activity. Upload images to get started!")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**System Version**: 2.0")

with col2:
    storage_stats = st.session_state.web_helper.get_storage_stats()
    if storage_stats:
        st.markdown(f"**Storage Used**: {storage_stats.get('total_size_mb', 0):.1f} MB")

with col3:
    st.markdown(f"**Last Updated**: {datetime.now().strftime('%H:%M:%S')}")

# Auto-refresh option
if st.checkbox("Auto-refresh (30s)", key="auto_refresh"):
    st.markdown("*Dashboard will refresh every 30 seconds*")
    st.markdown(
        """
        <script>
            setTimeout(function() {
                window.location.reload();
            }, 30000);
        </script>
        """,
        unsafe_allow_html=True
    )