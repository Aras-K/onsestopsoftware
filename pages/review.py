# pages/2_Review.py - Review Page with Image Display and Original Filenames

import streamlit as st
import os
import sys
from datetime import datetime
import pandas as pd
import json
import base64
import logging
import time  # Added missing import

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
    page_title="Review - Radar Extraction",
    page_icon="👁️",
    layout="wide"
)

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

if 'current_review_index' not in st.session_state:
    st.session_state.current_review_index = 0

if 'review_items' not in st.session_state:
    st.session_state.review_items = []

if 'status_filter' not in st.session_state:
    st.session_state.status_filter = 'pending'

# Header
st.title("👁️ Review Extraction Results")
st.markdown("Review and correct extracted radar data with original images")

# Controls
col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

with col1:
    status_filter = st.selectbox(
        "Filter by Status",
        ["pending", "success", "failed", "all"],
        index=["pending", "success", "failed", "all"].index(st.session_state.status_filter),
        help="Filter extractions by status"
    )
    
    if status_filter != st.session_state.status_filter:
        st.session_state.status_filter = status_filter
        st.session_state.current_review_index = 0

with col2:
    limit = st.number_input("Items to Load", min_value=10, max_value=100, value=50, step=10)

with col3:
    if st.button("🔄 Refresh", type="primary", use_container_width=True):
        st.session_state.current_review_index = 0
        st.rerun()

with col4:
    # Get statistics
    stats = st.session_state.web_helper.get_extraction_statistics()
    pending_count = stats.get('pending', 0)
    st.metric("Pending", pending_count, delta=None if pending_count == 0 else "⚠️")

# Load review items
with st.spinner("Loading review items..."):
    st.session_state.review_items = st.session_state.web_helper.get_review_items_for_web(
        limit=limit,
        status_filter=status_filter
    )

if not st.session_state.review_items:
    st.info("No items to review with the current filter")
    st.stop()

# Review Navigation
st.markdown("---")
nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])

with nav_col1:
    if st.button("⬅️ Previous", disabled=st.session_state.current_review_index == 0):
        st.session_state.current_review_index -= 1
        st.rerun()

with nav_col2:
    # Item selector with original filename display
    item_options = []
    for i, item in enumerate(st.session_state.review_items):
        filename = item.get('original_filename', item.get('filename', 'Unknown'))
        timestamp = item.get('extraction_timestamp', item.get('timestamp', ''))
        
        # Format timestamp for display
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                time_str = timestamp[:16] if len(timestamp) > 16 else timestamp
        else:
            time_str = ""
        
        confidence = item.get('overall_confidence', 0) * 100
        status_icon = "✅" if item['status'] == 'success' else "⚠️" if item['status'] == 'pending' else "❌"
        
        option_text = f"{i+1}. {status_icon} {filename} ({confidence:.1f}%) - {time_str}"
        item_options.append(option_text)
    
    selected_option = st.selectbox(
        "Select Item to Review",
        options=range(len(item_options)),
        format_func=lambda x: item_options[x],
        index=st.session_state.current_review_index
    )
    
    if selected_option != st.session_state.current_review_index:
        st.session_state.current_review_index = selected_option
        st.rerun()

with nav_col3:
    if st.button("Next ➡️", disabled=st.session_state.current_review_index >= len(st.session_state.review_items) - 1):
        st.session_state.current_review_index += 1
        st.rerun()

# Current item
current_item = st.session_state.review_items[st.session_state.current_review_index]

# Display extraction info with timestamp
st.markdown("---")
info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:
    st.markdown("**Original Filename:**")
    st.markdown(f"📄 {current_item.get('original_filename', current_item.get('filename', 'Unknown'))}")

with info_col2:
    st.markdown("**Extraction Time:**")
    timestamp = current_item.get('extraction_timestamp', current_item.get('timestamp', ''))
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            st.markdown(f"🕐 {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            st.markdown(f"🕐 {timestamp}")
    else:
        st.markdown("🕐 N/A")

with info_col3:
    confidence = current_item.get('overall_confidence', 0)
    color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
    st.markdown("**Confidence:**")
    st.markdown(f"{color} {confidence:.1%}")

with info_col4:
    status = current_item.get('status', 'pending')
    status_display = {
        'success': '✅ Success',
        'pending': '⚠️ Pending',
        'failed': '❌ Failed',
        'rejected': '🚫 Rejected'
    }.get(status, status)
    st.markdown("**Status:**")
    st.markdown(status_display)

# Main content area - Image and Fields side by side
st.markdown("---")
img_col, fields_col = st.columns([1, 1])

# Image display
with img_col:
    st.markdown("### 🖼️ Original Image")
    
    # Try to display the image
    if current_item.get('image_base64'):
        # Display from base64 data
        try:
            image_html = f'''
            <div style="border: 2px solid #ddd; border-radius: 8px; padding: 10px; background: #f9f9f9;">
                <img src="data:image/png;base64,{current_item['image_base64']}" 
                     style="width: 100%; height: auto; border-radius: 4px;">
                <p style="text-align: center; margin-top: 10px; color: #666;">
                    {current_item.get('original_filename', 'Image')}
                </p>
            </div>
            '''
            st.markdown(image_html, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not display image: {e}")
            st.info("Image data is corrupted or unavailable")
    elif current_item.get('image_path'):
        # Display from path/URL
        try:
            st.image(current_item['image_path'], 
                    caption=current_item.get('original_filename', 'Radar Image'),
                    use_container_width=True)
        except:
            st.warning("Could not load image from storage")
            st.info(f"Path: {current_item['image_path']}")
    else:
        st.info("No image available for this extraction")
        st.markdown(f"**Filename:** {current_item.get('original_filename', 'Unknown')}")
        
        # Option to reprocess if image is missing
        if st.button("🔄 Reprocess Image"):
            st.info("Reprocessing functionality not yet implemented")

# Fields display and editing
with fields_col:
    st.markdown("### 📊 Extracted Fields")
    
    # Create tabs for different field categories
    tabs = st.tabs(["Navigation", "Radar Settings", "Display", "All Fields"])
    
    # Categorize fields
    navigation_fields = ['heading', 'speed', 'position', 'cog', 'sog']
    settings_fields = ['gain', 'sea_clutter', 'rain_clutter', 'tune']
    display_fields = ['range', 'presentation_mode', 'vector']
    
    # Store field corrections
    corrections = {}
    
    with tabs[0]:  # Navigation
        for field_name in navigation_fields:
            if field_name in current_item['fields']:
                field = current_item['fields'][field_name]
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    new_value = st.text_input(
                        f"{field_name.replace('_', ' ').title()}",
                        value=str(field['value']) if field['value'] else "",
                        key=f"nav_{field_name}_{current_item['extraction_id']}"
                    )
                    if new_value != str(field['value']):
                        corrections[field_name] = new_value
                
                with col2:
                    conf = field.get('confidence', 0)
                    color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                    st.markdown(f"{color} {conf:.1%}")
    
    with tabs[1]:  # Radar Settings
        for field_name in settings_fields:
            if field_name in current_item['fields']:
                field = current_item['fields'][field_name]
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    new_value = st.text_input(
                        f"{field_name.replace('_', ' ').title()}",
                        value=str(field['value']) if field['value'] else "",
                        key=f"settings_{field_name}_{current_item['extraction_id']}"
                    )
                    if new_value != str(field['value']):
                        corrections[field_name] = new_value
                
                with col2:
                    conf = field.get('confidence', 0)
                    color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                    st.markdown(f"{color} {conf:.1%}")
    
    with tabs[2]:  # Display
        for field_name in display_fields:
            if field_name in current_item['fields']:
                field = current_item['fields'][field_name]
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    new_value = st.text_input(
                        f"{field_name.replace('_', ' ').title()}",
                        value=str(field['value']) if field['value'] else "",
                        key=f"display_{field_name}_{current_item['extraction_id']}"
                    )
                    if new_value != str(field['value']):
                        corrections[field_name] = new_value
                
                with col2:
                    conf = field.get('confidence', 0)
                    color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                    st.markdown(f"{color} {conf:.1%}")
    
    with tabs[3]:  # All Fields
        # Display all fields in a scrollable area
        for field_name, field in current_item['fields'].items():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_value = st.text_input(
                    f"{field_name.replace('_', ' ').title()}",
                    value=str(field['value']) if field['value'] else "",
                    key=f"all_{field_name}_{current_item['extraction_id']}"
                )
                if new_value != str(field['value']):
                    corrections[field_name] = new_value
            
            with col2:
                conf = field.get('confidence', 0)
                color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                st.markdown(f"{color} {conf:.1%}")

# Review actions
st.markdown("---")
st.markdown("### ✍️ Review Actions")

# Review notes
review_notes = st.text_area(
    "Review Notes (optional)",
    value=current_item.get('review_notes', ''),
    height=100,
    placeholder="Add any notes about this extraction..."
)

# Action buttons
action_col1, action_col2, action_col3, action_col4 = st.columns(4)

with action_col1:
    if st.button("✅ Approve", type="primary", use_container_width=True):
        with st.spinner("Saving..."):
            success = st.session_state.web_helper.submit_review_from_web(
                extraction_id=current_item['extraction_id'],
                reviewer=os.environ.get('DEFAULT_USER', 'Reviewer'),
                action='approve',
                notes=review_notes
            )
            if success:
                st.success("✅ Approved successfully!")
                # Move to next item
                if st.session_state.current_review_index < len(st.session_state.review_items) - 1:
                    st.session_state.current_review_index += 1
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to save approval")

with action_col2:
    if corrections:
        if st.button("💾 Save Corrections", type="primary", use_container_width=True):
            with st.spinner("Saving corrections..."):
                success = st.session_state.web_helper.submit_review_from_web(
                    extraction_id=current_item['extraction_id'],
                    reviewer=os.environ.get('DEFAULT_USER', 'Reviewer'),
                    action='correct',
                    corrections=corrections,
                    notes=review_notes
                )
                if success:
                    st.success("✅ Corrections saved!")
                    # Move to next item
                    if st.session_state.current_review_index < len(st.session_state.review_items) - 1:
                        st.session_state.current_review_index += 1
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to save corrections")
    else:
        st.button("💾 Save Corrections", disabled=True, use_container_width=True, 
                 help="Make changes to fields first")

with action_col3:
    if st.button("❌ Reject", use_container_width=True):
        with st.spinner("Rejecting..."):
            success = st.session_state.web_helper.submit_review_from_web(
                extraction_id=current_item['extraction_id'],
                reviewer=os.environ.get('DEFAULT_USER', 'Reviewer'),
                action='reject',
                notes=review_notes if review_notes else "Extraction rejected during review"
            )
            if success:
                st.warning("❌ Rejected")
                # Move to next item
                if st.session_state.current_review_index < len(st.session_state.review_items) - 1:
                    st.session_state.current_review_index += 1
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to save rejection")

with action_col4:
    if current_item['status'] == 'success':
        if st.button("🔄 Reopen", use_container_width=True):
            reason = st.text_input("Reason for reopening", key="reopen_reason")
            if reason:
                with st.spinner("Reopening..."):
                    success = st.session_state.web_helper.reopen_for_review(
                        extraction_id=current_item['extraction_id'],
                        reason=reason,
                        reviewer=os.environ.get('DEFAULT_USER', 'Reviewer')
                    )
                    if success:
                        st.info("🔄 Reopened for review")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to reopen")
            else:
                st.warning("Please provide a reason for reopening")
    else:
        if st.button("⏭️ Skip", use_container_width=True):
            if st.session_state.current_review_index < len(st.session_state.review_items) - 1:
                st.session_state.current_review_index += 1
                st.rerun()
            else:
                st.info("No more items to review")

# Previous review history
if current_item.get('reviewed_by'):
    st.markdown("---")
    st.markdown("### 📝 Review History")
    
    history_cols = st.columns([2, 2, 3])
    with history_cols[0]:
        st.markdown(f"**Reviewed by:** {current_item['reviewed_by']}")
    with history_cols[1]:
        if current_item.get('review_timestamp'):
            st.markdown(f"**Review time:** {current_item['review_timestamp']}")
    with history_cols[2]:
        if current_item.get('review_notes'):
            st.markdown(f"**Previous notes:** {current_item['review_notes']}")

# Sidebar with statistics and filters
with st.sidebar:
    st.markdown("### 📊 Review Statistics")
    
    # Overall stats
    stats = st.session_state.web_helper.get_extraction_statistics()
    
    st.metric("Total Extractions", stats.get('total', 0))
    st.metric("Successful", stats.get('success', 0), 
             delta=f"{(stats.get('success', 0) / stats.get('total', 1) * 100):.1f}%")
    st.metric("Pending Review", stats.get('pending', 0))
    st.metric("Failed", stats.get('failed', 0))
    st.metric("Already Reviewed", stats.get('reviewed', 0))
    
    st.markdown("---")
    
    # Field confidence distribution for current item
    if current_item['fields']:
        st.markdown("### 🎯 Field Confidence")
        
        high_conf = sum(1 for f in current_item['fields'].values() if f.get('confidence', 0) > 0.8)
        med_conf = sum(1 for f in current_item['fields'].values() if 0.5 < f.get('confidence', 0) <= 0.8)
        low_conf = sum(1 for f in current_item['fields'].values() if f.get('confidence', 0) <= 0.5)
        
        st.markdown(f"🟢 High: {high_conf}")
        st.markdown(f"🟡 Medium: {med_conf}")
        st.markdown(f"🔴 Low: {low_conf}")
    
    st.markdown("---")
    
    # Extraction metadata
    st.markdown("### 📋 Extraction Details")
    st.markdown(f"**ID:** {current_item['extraction_id']}")
    st.markdown(f"**Type:** {current_item.get('radar_type', 'Unknown')}")
    st.markdown(f"**Processing Time:** {current_item.get('processing_time', 0):.1f}s")
    
    # Original filename
    st.markdown(f"**Original File:** {current_item.get('original_filename', 'N/A')}")
    
    # Timestamp
    if current_item.get('extraction_timestamp'):
        st.markdown(f"**Extracted:** {current_item['extraction_timestamp'][:19]}")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("📥 Export Current", use_container_width=True):
        # Export current item as JSON
        export_data = {
            'extraction_id': current_item['extraction_id'],
            'filename': current_item.get('original_filename', current_item.get('filename')),
            'timestamp': current_item.get('extraction_timestamp'),
            'fields': current_item['fields'],
            'confidence': current_item.get('overall_confidence'),
            'status': current_item.get('status')
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            "💾 Download JSON",
            data=json_str,
            file_name=f"extraction_{current_item['extraction_id']}.json",
            mime="application/json"
        )
    
    if st.button("📊 View Analytics", use_container_width=True):
        st.switch_page("pages/3_Analytics.py")
    
    if st.button("🏠 Dashboard", use_container_width=True):
        st.switch_page("app.py")