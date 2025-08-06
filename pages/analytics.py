# pages/3_Analytics.py - Professional Analytics Dashboard for Client

import streamlit as st
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging

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
    page_title="Analytics - Radar Extraction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    
    /* Main container styling */
    .main {
        padding: 1rem 2rem;
        background-color: var(--light-bg);
    }
    
    /* Professional header */
    .analytics-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .analytics-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    .analytics-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-size: 1.1rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid var(--secondary-color);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Success indicator */
    .success-indicator {
        background: linear-gradient(135deg, var(--success-color) 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
    }
    
    /* Section headers */
    h2 {
        color: var(--dark-bg);
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--secondary-color);
    }
    
    /* Streamlit metric override */
    div[data-testid="metric-container"] {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid var(--secondary-color);
    }
    
    /* Professional table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Executive summary box */
    .executive-summary {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    /* ROI indicator */
    .roi-indicator {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
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

# Professional Header
st.markdown("""
<div class="analytics-header">
    <h1>📊 Executive Analytics Dashboard</h1>
    <p>Real-time Performance Metrics & Business Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Date Range Selection with Quick Filters
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

with col1:
    quick_select = st.selectbox(
        "📅 Time Period",
        ["Today", "Last 7 Days", "Last 30 Days", "Last Quarter", "Year to Date"],
        index=2
    )

# Calculate date range based on selection
today = datetime.now()
if quick_select == "Today":
    start_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = today
elif quick_select == "Last 7 Days":
    start_date = today - timedelta(days=7)
    end_date = today
elif quick_select == "Last 30 Days":
    start_date = today - timedelta(days=30)
    end_date = today
elif quick_select == "Last Quarter":
    start_date = today - timedelta(days=90)
    end_date = today
else:  # Year to Date
    start_date = datetime(today.year, 1, 1)
    end_date = today

with col5:
    if st.button("🔄 Refresh Dashboard", type="primary", use_container_width=True):
        st.rerun()

# Get analytics data with error handling
try:
    with st.spinner("Loading analytics data..."):
        analytics_data = st.session_state.web_helper.get_analytics_data(
            days=(end_date - start_date).days
        )
        overall_stats = analytics_data.get('overall_stats', {})
except Exception as e:
    logger.error(f"Error loading analytics: {e}")
    st.error("Failed to load analytics data. Please check the connection.")
    st.stop()

# SECTION 1: EXECUTIVE SUMMARY
st.markdown("## 📈 Executive Summary")

# Calculate key business metrics
total_processed = overall_stats.get('total_images', 0) or 0
successful = overall_stats.get('successful', 0) or 0
failed = overall_stats.get('failed', 0) or 0
avg_processing_time = overall_stats.get('avg_processing_time', 0) or 0
success_rate = (successful / total_processed * 100) if total_processed > 0 else 0

# Time saved calculation (assuming 5 minutes manual extraction per image)
time_saved_hours = (total_processed * 5) / 60  # 5 minutes per image converted to hours
cost_savings = time_saved_hours * 50  # Assuming $50/hour labor cost

# Executive KPIs
exec_cols = st.columns(4)

with exec_cols[0]:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #64748b; margin: 0;">Total Processed</h4>
        <h2 style="color: #1e293b; margin: 0.5rem 0;">{:,}</h2>
        <p style="color: #10b981; margin: 0;">📈 Images analyzed</p>
    </div>
    """.format(total_processed), unsafe_allow_html=True)

with exec_cols[1]:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #64748b; margin: 0;">Automation Rate</h4>
        <h2 style="color: #1e293b; margin: 0.5rem 0;">{:.1f}%</h2>
        <p style="color: {}; margin: 0;">{}</p>
    </div>
    """.format(
        success_rate,
        "#10b981" if success_rate >= 85 else "#f59e0b",
        "✅ Excellent" if success_rate >= 85 else "⚠️ Needs Review"
    ), unsafe_allow_html=True)

with exec_cols[2]:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #64748b; margin: 0;">Time Saved</h4>
        <h2 style="color: #1e293b; margin: 0.5rem 0;">{:.0f} hrs</h2>
        <p style="color: #3b82f6; margin: 0;">⏰ Productivity gain</p>
    </div>
    """.format(time_saved_hours), unsafe_allow_html=True)

with exec_cols[3]:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #64748b; margin: 0;">Cost Savings</h4>
        <h2 style="color: #1e293b; margin: 0.5rem 0;">${:,.0f}</h2>
        <p style="color: #10b981; margin: 0;">💰 ROI delivered</p>
    </div>
    """.format(cost_savings), unsafe_allow_html=True)

# ROI Analysis Box
st.markdown("""
<div class="roi-indicator">
    📊 Return on Investment: {:.0f}% efficiency gain | ${:,.0f} saved | {:.0f} hours automated
</div>
""".format(success_rate, cost_savings, time_saved_hours), unsafe_allow_html=True)

# SECTION 2: PERFORMANCE TRENDS
st.markdown("## 📊 Performance Trends & Insights")

col1, col2 = st.columns([2, 1])

with col1:
    # Create comprehensive trend chart
    daily_data = analytics_data.get('daily_trend', [])
    if daily_data:
        df_daily = pd.DataFrame(daily_data)
        
        # Create subplot figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Daily Processing Volume", "Success Rate Trend"),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=df_daily['date'],
                y=df_daily['total'],
                name='Total Processed',
                marker_color='#3b82f6',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=df_daily['date'],
                y=df_daily['successful'],
                name='Successful',
                marker_color='#10b981',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Success rate trend
        df_daily['success_rate'] = (df_daily['successful'] / df_daily['total'] * 100).fillna(0)
        
        fig.add_trace(
            go.Scatter(
                x=df_daily['date'],
                y=df_daily['success_rate'],
                mode='lines+markers',
                name='Success Rate %',
                line=dict(color='#f59e0b', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        # Add 85% target line
        fig.add_hline(y=85, line_dash="dash", line_color="green", 
                     annotation_text="Target: 85%", row=2, col=1)
        
        fig.update_layout(
            height=500,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Images", row=1, col=1)
        fig.update_yaxes(title_text="Success Rate (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trend data available for the selected period")

with col2:
    # Performance Summary Card
    st.markdown("""
    <div class="executive-summary">
        <h4 style="color: #1e293b; margin-bottom: 1rem;">📊 Performance Summary</h4>
    """, unsafe_allow_html=True)
    
    # Calculate performance indicators
    if total_processed > 0:
        st.metric("Daily Average", f"{total_processed / max((end_date - start_date).days, 1):.0f} images")
        st.metric("Peak Performance", f"{max([d.get('total', 0) for d in daily_data], default=0)} images/day")
        st.metric("Accuracy Rate", f"{success_rate:.1f}%", 
                 delta=f"{success_rate - 85:.1f}% vs target",
                 delta_color="normal" if success_rate >= 85 else "inverse")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Quick insights
    st.markdown("### 💡 Quick Insights")
    
    if success_rate >= 90:
        st.success("🎯 Exceeding performance targets!")
    elif success_rate >= 85:
        st.success("✅ Meeting performance targets")
    elif success_rate >= 75:
        st.warning("⚠️ Below target, optimization needed")
    else:
        st.error("🔴 Significant improvement required")

# SECTION 3: FIELD EXTRACTION ANALYSIS
st.markdown("## 🎯 Field Extraction Intelligence")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Field Performance", "Radar Type Analysis", "Method Effectiveness"])

with tab1:
    field_perf = analytics_data.get('field_performance', [])
    if field_perf:
        # Categorize fields
        critical_fields = ['heading', 'speed', 'position', 'cog', 'sog']
        
        # Create performance matrix
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Sort and prepare data
            field_perf_sorted = sorted(field_perf, key=lambda x: x['success_rate'], reverse=True)
            
            # Create grouped bar chart
            fig_matrix = go.Figure()
            
            # Separate critical and non-critical fields
            critical_data = [f for f in field_perf_sorted if f['field'] in critical_fields]
            other_data = [f for f in field_perf_sorted if f['field'] not in critical_fields]
            
            # Add critical fields
            if critical_data:
                fig_matrix.add_trace(go.Bar(
                    name='Critical Fields',
                    y=[f['field'] for f in critical_data],
                    x=[f['success_rate'] for f in critical_data],
                    orientation='h',
                    marker_color='#ef4444',
                    text=[f"{f['success_rate']:.1f}%" for f in critical_data],
                    textposition='outside'
                ))
            
            # Add other fields
            if other_data:
                fig_matrix.add_trace(go.Bar(
                    name='Standard Fields',
                    y=[f['field'] for f in other_data[:10]],  # Top 10
                    x=[f['success_rate'] for f in other_data[:10]],
                    orientation='h',
                    marker_color='#3b82f6',
                    text=[f"{f['success_rate']:.1f}%" for f in other_data[:10]],
                    textposition='outside'
                ))
            
            fig_matrix.update_layout(
                title="Field Extraction Success Rates",
                xaxis_title="Success Rate (%)",
                height=500,
                showlegend=True,
                xaxis=dict(range=[0, 105]),
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_matrix, use_container_width=True)
        
        with col2:
            # Field categories performance
            st.markdown("### 📊 Category Performance")
            
            # Categorize and calculate averages
            navigation_fields = ['heading', 'speed', 'position', 'cog', 'sog']
            settings_fields = ['gain', 'sea_clutter', 'rain_clutter', 'tune']
            display_fields = ['range', 'presentation_mode', 'vector']
            
            categories = {
                'Navigation': navigation_fields,
                'Radar Settings': settings_fields,
                'Display': display_fields
            }
            
            for cat_name, fields in categories.items():
                cat_data = [f for f in field_perf if f['field'] in fields]
                if cat_data:
                    avg_rate = np.mean([f['success_rate'] for f in cat_data])
                    
                    color = "#10b981" if avg_rate >= 80 else "#f59e0b" if avg_rate >= 60 else "#ef4444"
                    
                    st.markdown(f"""
                    <div style="background: {color}20; padding: 1rem; border-radius: 8px; 
                               border-left: 4px solid {color}; margin-bottom: 1rem;">
                        <h4 style="margin: 0; color: #1e293b;">{cat_name}</h4>
                        <h2 style="margin: 0.5rem 0; color: {color};">{avg_rate:.1f}%</h2>
                        <p style="margin: 0; color: #64748b;">{len(cat_data)} fields tracked</p>
                    </div>
                    """, unsafe_allow_html=True)

with tab2:
    # Radar type distribution and performance
    radar_dist = analytics_data.get('radar_type_distribution', [])
    if radar_dist:
        col1, col2 = st.columns(2)
        
        with col1:
            df_radar = pd.DataFrame(radar_dist)
            
            # Pie chart for distribution
            fig_pie = go.Figure(data=[go.Pie(
                labels=df_radar['radar_type'],
                values=df_radar['count'],
                hole=.4,
                marker_colors=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
            )])
            
            fig_pie.update_layout(
                title="Radar Type Distribution",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Performance by radar type
            st.markdown("### 📊 Performance by Type")
            
            for _, row in df_radar.iterrows():
                radar_type = row['radar_type']
                count = row['count']
                avg_conf = row.get('avg_confidence', 0) * 100
                
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #1e293b;">{radar_type}</h4>
                    <div style="background: #e2e8f0; border-radius: 8px; height: 30px; margin: 0.5rem 0;">
                        <div style="background: #3b82f6; width: {avg_conf}%; height: 100%; 
                                   border-radius: 8px; display: flex; align-items: center; 
                                   justify-content: center; color: white; font-weight: 600;">
                            {avg_conf:.1f}%
                        </div>
                    </div>
                    <p style="color: #64748b; margin: 0;">{count} images processed</p>
                </div>
                """, unsafe_allow_html=True)

with tab3:
    # Method effectiveness analysis
    method_data = analytics_data.get('method_effectiveness', [])
    if method_data:
        df_methods = pd.DataFrame(method_data)
        
        # Calculate success rates
        df_methods['success_rate'] = (df_methods['success_count'] / df_methods['usage_count'] * 100).fillna(0)
        
        # Create comparative chart
        fig_methods = go.Figure()
        
        fig_methods.add_trace(go.Bar(
            x=df_methods['extraction_method'],
            y=df_methods['success_rate'],
            name='Success Rate',
            marker_color='#3b82f6',
            yaxis='y',
            text=[f"{rate:.1f}%" for rate in df_methods['success_rate']],
            textposition='outside'
        ))
        
        fig_methods.add_trace(go.Scatter(
            x=df_methods['extraction_method'],
            y=df_methods['avg_confidence'] * 100,
            name='Avg Confidence',
            mode='lines+markers',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=10),
            yaxis='y2'
        ))
        
        fig_methods.update_layout(
            title="Extraction Method Performance",
            xaxis_title="Method",
            yaxis=dict(title="Success Rate (%)", side="left"),
            yaxis2=dict(title="Confidence (%)", overlaying="y", side="right"),
            height=400,
            hovermode='x unified',
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_methods, use_container_width=True)

# SECTION 4: BUSINESS IMPACT & RECOMMENDATIONS
st.markdown("## 💼 Business Impact & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="executive-summary">
        <h3 style="color: #1e293b;">📈 Business Value Delivered</h3>
    """, unsafe_allow_html=True)
    
    # Calculate business metrics
    manual_time_per_image = 5  # minutes
    automation_time = avg_processing_time / 60  # convert to minutes
    time_reduction = ((manual_time_per_image - automation_time) / manual_time_per_image * 100) if manual_time_per_image > 0 else 0
    
    st.metric("Process Acceleration", f"{time_reduction:.0f}% faster")
    st.metric("Data Accuracy", f"{success_rate:.1f}%")
    st.metric("Monthly Volume Capacity", f"{total_processed * 30 / max((end_date - start_date).days, 1):,.0f} images")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="executive-summary">
        <h3 style="color: #1e293b;">🎯 Recommendations</h3>
    """, unsafe_allow_html=True)
    
    # Generate smart recommendations
    recommendations = []
    
    if success_rate < 85:
        recommendations.append("🔧 Optimize extraction algorithms for better accuracy")
    
    if avg_processing_time > 5:
        recommendations.append("⚡ Consider performance optimization to reduce processing time")
    
    field_perf = analytics_data.get('field_performance', [])
    if field_perf:
        low_performing = [f for f in field_perf if f['success_rate'] < 60]
        if len(low_performing) > 5:
            recommendations.append(f"📊 Focus on improving {len(low_performing)} underperforming fields")
    
    if not recommendations:
        recommendations.append("✅ System performing optimally")
        recommendations.append("📈 Consider scaling up operations")
    
    for rec in recommendations:
        st.markdown(f"• {rec}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# SECTION 5: EXPORT OPTIONS
st.markdown("---")
st.markdown("## 📥 Export & Reporting")

export_cols = st.columns([2, 2, 1, 2])

with export_cols[0]:
    report_type = st.selectbox(
        "Report Type",
        ["Executive Summary", "Detailed Analytics", "Performance Report", "Custom Export"]
    )

with export_cols[1]:
    export_format = st.selectbox(
        "Format",
        ["PDF Report", "Excel Workbook", "CSV Data", "JSON"]
    )

with export_cols[3]:
    if st.button("📥 Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            # Generate appropriate export based on selection
            if export_format in ["CSV Data", "JSON"]:
                export_result = st.session_state.web_helper.export_data_for_web(
                    format=export_format.split()[0].lower(),
                    days=(end_date - start_date).days
                )
                
                if export_result.get('success'):
                    st.download_button(
                        label="⬇️ Download Report",
                        data=export_result['content'],
                        file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now():%Y%m%d}.{export_format.split()[0].lower()}",
                        mime=export_result['mime_type']
                    )
                else:
                    st.error(f"Export failed: {export_result.get('error', 'Unknown error')}")
            else:
                st.info(f"{export_format} export coming soon!")

# Footer with system status
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

with col2:
    st.markdown(f"**Data Period**: {(end_date - start_date).days} days")

with col3:
    st.markdown(f"**System Version**: 2.0 Professional")

with col4:
    st.markdown(f"**Environment**: {os.environ.get('ENVIRONMENT', 'Production')}")