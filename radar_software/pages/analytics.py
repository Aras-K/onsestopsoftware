# pages/3_Analytics.py - Analytics Dashboard

import streamlit as st
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_helpers import get_web_helper
from radar_extraction_architecture import RADAR_FIELDS

# Page config
st.set_page_config(
    page_title="Analytics - Radar Extraction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main {
        padding: 1rem 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    h1 {
        color: #1e293b;
        font-weight: 600;
    }
    
    h2 {
        color: #334155;
        font-size: 1.5rem;
        font-weight: 500;
        margin-top: 2rem;
    }
    
    h3 {
        color: #475569;
        font-size: 1.25rem;
        font-weight: 500;
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .success-metric {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'web_helper' not in st.session_state:
    st.session_state.web_helper = get_web_helper({
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'openai': os.getenv('OPENAI_API_KEY'),
        'google': os.getenv('GOOGLE_API_KEY')
    })

# Header
st.title("📊 Analytics Dashboard")
st.markdown("Comprehensive insights into your radar data extraction performance")

# Date range selector
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

with col1:
    date_range = st.selectbox(
        "Time Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time", "Custom"],
        index=1
    )

# Calculate date range
if date_range == "Last 7 Days":
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
elif date_range == "Last 30 Days":
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
elif date_range == "Last 90 Days":
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
elif date_range == "All Time":
    start_date = datetime(2000, 1, 1)  # Effectively all time
    end_date = datetime.now()
else:  # Custom
    with col2:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col3:
        end_date = st.date_input("End Date", datetime.now())

with col4:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# Get analytics data
with st.spinner("Loading analytics data..."):
    analytics_data = st.session_state.web_helper.get_analytics_data(
        days=(end_date - start_date).days if isinstance(start_date, datetime) else 30
    )

# Section 1: Key Metrics Overview
st.markdown("## 🎯 Key Performance Indicators")

metric_cols = st.columns(5)

# Overall statistics
overall_stats = analytics_data.get('overall_stats', {})

with metric_cols[0]:
    total_processed = overall_stats.get('total_images', 0)
    st.metric(
        label="Total Processed",
        value=f"{total_processed:,}",
        delta=None
    )

with metric_cols[1]:
    success_rate = (overall_stats.get('successful', 0) / total_processed * 100) if total_processed > 0 else 0
    st.metric(
        label="Success Rate",
        value=f"{success_rate:.1f}%",
        delta=f"{success_rate - 85:.1f}%" if success_rate > 0 else None,
        delta_color="normal" if success_rate > 85 else "inverse"
    )

with metric_cols[2]:
    avg_confidence = overall_stats.get('avg_confidence', 0) * 100
    st.metric(
        label="Avg Confidence",
        value=f"{avg_confidence:.1f}%",
        delta=None
    )

with metric_cols[3]:
    avg_processing_time = overall_stats.get('avg_processing_time', 0)
    st.metric(
        label="Avg Time/Image",
        value=f"{avg_processing_time:.1f}s",
        delta=None
    )

with metric_cols[4]:
    pending_review = overall_stats.get('pending_review', 0)
    st.metric(
        label="Pending Review",
        value=pending_review,
        delta=None,
        delta_color="inverse" if pending_review > 0 else "off"
    )

# Section 2: Processing Trends
st.markdown("## 📈 Processing Trends")

# Daily processing chart
daily_data = analytics_data.get('daily_trend', [])
if daily_data:
    df_daily = pd.DataFrame(daily_data)
    
    fig_trend = go.Figure()
    
    # Add traces
    fig_trend.add_trace(go.Scatter(
        x=df_daily['date'],
        y=df_daily['total'],
        mode='lines+markers',
        name='Total Processed',
        line=dict(color='#3b82f6', width=3),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=df_daily['date'],
        y=df_daily['successful'],
        mode='lines+markers',
        name='Successful',
        line=dict(color='#10b981', width=3)
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=df_daily['date'],
        y=df_daily['failed'],
        mode='lines+markers',
        name='Failed',
        line=dict(color='#ef4444', width=2, dash='dot')
    ))
    
    fig_trend.update_layout(
        title="Daily Processing Volume",
        xaxis_title="Date",
        yaxis_title="Number of Images",
        hovermode='x unified',
        height=400,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No processing data available for the selected period")

# Section 3: Field Performance Analysis
st.markdown("## 🎯 Field Extraction Performance")

col1, col2 = st.columns([3, 2])

with col1:
    # Field success rates
    field_perf = analytics_data.get('field_performance', [])
    if field_perf:
        # Sort by success rate
        field_perf_sorted = sorted(field_perf, key=lambda x: x['success_rate'], reverse=True)
        
        # Create horizontal bar chart
        fig_fields = go.Figure()
        
        # Prepare data
        fields = [f['field'] for f in field_perf_sorted]
        success_rates = [f['success_rate'] for f in field_perf_sorted]
        counts = [f['count'] for f in field_perf_sorted]
        
        # Color based on performance
        colors = ['#10b981' if sr >= 80 else '#f59e0b' if sr >= 60 else '#ef4444' for sr in success_rates]
        
        fig_fields.add_trace(go.Bar(
            y=fields,
            x=success_rates,
            orientation='h',
            marker_color=colors,
            text=[f'{sr:.1f}%' for sr in success_rates],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Success Rate: %{x:.1f}%<br>Extracted: %{customdata}<extra></extra>',
            customdata=counts
        ))
        
        fig_fields.update_layout(
            title="Field Extraction Success Rates",
            xaxis_title="Success Rate (%)",
            yaxis_title="Field",
            height=600,
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig_fields, use_container_width=True)

with col2:
    # Radar type distribution
    radar_dist = analytics_data.get('radar_type_distribution', [])
    if radar_dist:
        df_radar = pd.DataFrame(radar_dist)
        
        fig_radar = go.Figure(data=[go.Pie(
            labels=df_radar['radar_type'],
            values=df_radar['count'],
            hole=.3,
            marker_colors=['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
        )])
        
        fig_radar.update_layout(
            title="Radar Type Distribution",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Method effectiveness
    st.markdown("### 🔧 Extraction Method Effectiveness")
    
    method_data = analytics_data.get('method_effectiveness', [])
    if method_data:
        for method in method_data:
            success_rate = (method['success_count'] / method['usage_count'] * 100) if method['usage_count'] > 0 else 0
            
            st.markdown(f"**{method['extraction_method']}**")
            
            # Progress bar for success rate
            progress_html = f"""
            <div style='background-color: #e5e7eb; border-radius: 10px; height: 20px; margin: 5px 0;'>
                <div style='background-color: {'#10b981' if success_rate > 80 else '#f59e0b'}; 
                           width: {success_rate}%; height: 100%; border-radius: 10px; 
                           text-align: center; color: white; font-size: 12px; line-height: 20px;'>
                    {success_rate:.1f}%
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
            
            st.caption(f"Used {method['usage_count']} times • Avg confidence: {method['avg_confidence']*100:.1f}%")
            st.markdown("---")

# Section 4: Performance Insights
st.markdown("## 💡 Performance Insights")

insight_cols = st.columns(3)

with insight_cols[0]:
    st.markdown("### 🎯 Top Performing Fields")
    top_fields = sorted(field_perf, key=lambda x: x['success_rate'], reverse=True)[:5]
    for field in top_fields:
        st.markdown(f"✅ **{field['field']}**: {field['success_rate']:.1f}%")

with insight_cols[1]:
    st.markdown("### ⚠️ Fields Needing Improvement")
    bottom_fields = sorted(field_perf, key=lambda x: x['success_rate'])[:5]
    for field in bottom_fields:
        st.markdown(f"🔴 **{field['field']}**: {field['success_rate']:.1f}%")

with insight_cols[2]:
    st.markdown("### 📊 Quick Stats")
    
    # Calculate additional insights
    if field_perf:
        fields_above_80 = sum(1 for f in field_perf if f['success_rate'] >= 80)
        fields_below_60 = sum(1 for f in field_perf if f['success_rate'] < 60)
        
        st.markdown(f"**High Performance Fields**: {fields_above_80}/{len(field_perf)}")
        st.markdown(f"**Low Performance Fields**: {fields_below_60}/{len(field_perf)}")
        
        # Milestone check
        if fields_above_80 >= 21:
            st.success("✅ Milestone 1 Achieved!")
        else:
            st.warning(f"📈 {21 - fields_above_80} more fields needed for Milestone 1")

# Section 5: Recent Activity
st.markdown("## 📋 Recent Activity")

# Get recent extractions
with st.session_state.web_helper.db_manager.get_connection() as conn:
    recent_df = pd.read_sql_query("""
        SELECT 
            filename,
            radar_type,
            overall_confidence,
            extraction_timestamp,
            requires_review,
            extraction_status
        FROM extractions
        ORDER BY extraction_timestamp DESC
        LIMIT 10
    """, conn)

if not recent_df.empty:
    # Format the dataframe
    recent_df['confidence'] = recent_df['overall_confidence'].apply(lambda x: f"{x:.1%}")
    recent_df['timestamp'] = pd.to_datetime(recent_df['extraction_timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    recent_df['status'] = recent_df['extraction_status'].apply(
        lambda x: '✅ Success' if x == 'success' else '⚠️ Partial' if x == 'partial' else '❌ Failed'
    )
    recent_df['review'] = recent_df['requires_review'].apply(lambda x: '🔍 Yes' if x else '✓ No')
    
    # Display table
    st.dataframe(
        recent_df[['timestamp', 'filename', 'radar_type', 'confidence', 'status', 'review']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'timestamp': 'Time',
            'filename': 'File',
            'radar_type': 'Radar Type',
            'confidence': 'Confidence',
            'status': 'Status',
            'review': 'Needs Review'
        }
    )

# Export section
st.markdown("---")
st.markdown("## 📥 Export Analytics Data")

export_cols = st.columns([2, 1, 1, 2])

with export_cols[0]:
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])

with export_cols[1]:
    include_details = st.checkbox("Include Details", value=True)

with export_cols[3]:
    if st.button("📥 Export Data", type="primary", use_container_width=True):
        # Generate export
        export_result = st.session_state.web_helper.export_data_for_web(
            format=export_format.lower(),
            days=(end_date - start_date).days if isinstance(start_date, datetime) else 30
        )
        
        if export_result['success']:
            st.download_button(
                label="⬇️ Download Export",
                data=export_result['content'],
                file_name=export_result['filename'],
                mime=export_result['mime_type']
            )
        else:
            st.error(f"Export failed: {export_result.get('error', 'Unknown error')}")

# Sidebar with additional info
with st.sidebar:
    st.markdown("## 📊 Analytics Guide")
    
    st.markdown("""
    ### Understanding Metrics
    
    **Success Rate**: Percentage of images processed with confidence > 80%
    
    **Field Performance**: How often each field is successfully extracted
    
    **Processing Time**: Average time to extract data from one image
    
    ### Performance Targets
    - Success Rate: > 85%
    - Avg Confidence: > 80%
    - Processing Time: < 5s
    
    ### Tips
    - Monitor low-performing fields
    - Review failed extractions
    - Export data for detailed analysis
    """)
    
    # System health indicator
    st.markdown("---")
    st.markdown("### 🔥 System Health")
    
    if success_rate > 85:
        st.success("System performing well")
    elif success_rate > 70:
        st.warning("Performance could be improved")
    else:
        st.error("System needs attention")