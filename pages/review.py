# pages/2_Review.py - Enhanced Review Interface with Success Items Support

import streamlit as st
import os
import sys
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional, List
import time
import base64
from io import BytesIO
from PIL import Image
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
    page_title="Review & Validation - Radar Extraction",
    page_icon="👁️",
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
        --info-color: #6366f1;
    }
    
    /* Main container */
    .main { 
        padding: 1rem 2rem; 
        background-color: #f8fafc;
    }
    
    /* Professional header */
    .review-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    /* Field container */
    .field-container {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 3px solid transparent;
        transition: all 0.2s;
    }
    
    .field-container:hover {
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left-color: var(--secondary-color);
    }
    
    /* Confidence badges */
    .confidence-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.875rem;
        font-weight: 600;
        text-align: center;
    }
    
    .high-conf { 
        background: #d4f4dd; 
        color: #1e7e34; 
    }
    
    .med-conf { 
        background: #fff3cd; 
        color: #856404; 
    }
    
    .low-conf { 
        background: #f8d7da; 
        color: #721c24; 
    }
    
    /* Review mode badge */
    .mode-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        margin-left: 1rem;
    }
    
    .mode-pending {
        background: var(--warning-color);
        color: white;
    }
    
    .mode-success {
        background: var(--success-color);
        color: white;
    }
    
    .mode-all {
        background: var(--info-color);
        color: white;
    }
    
    /* Image container */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Review panel */
    .review-panel {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        height: 100%;
    }
    
    /* Status indicator */
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .status-success {
        background: var(--success-color);
        color: white;
    }
    
    .status-warning {
        background: var(--warning-color);
        color: white;
    }
    
    .status-error {
        background: var(--danger-color);
        color: white;
    }
    
    .status-info {
        background: var(--info-color);
        color: white;
    }
    
    /* Field edit indicator */
    .field-edited {
        background-color: #fef3c7 !important;
        border-left-color: var(--warning-color) !important;
    }
    
    /* Action buttons */
    .action-button {
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s;
    }
    
    /* Progress indicator */
    .progress-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 24px;
        background: white;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] { 
        padding: 8px 16px; 
        font-weight: 500;
    }
    
    /* Notes area */
    .notes-container {
        background: #f1f5f9;
        padding: 1rem;
        border-radius: 8px;
        border: 2px dashed #cbd5e1;
    }
    
    /* Success item indicator */
    .success-item-banner {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 1rem;
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
        st.error(f"Failed to initialize web helper: {e}")
        st.stop()

# Session state initialization
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

if 'all_corrections' not in st.session_state:
    st.session_state.all_corrections = {}

if 'reviewer' not in st.session_state:
    st.session_state.reviewer = ""

if 'review_items' not in st.session_state:
    st.session_state.review_items = None

if 'items_reviewed' not in st.session_state:
    st.session_state.items_reviewed = 0

if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()

if 'review_mode' not in st.session_state:
    st.session_state.review_mode = 'pending'  # 'pending', 'success', 'all'

# Helper functions
def get_current_corrections():
    """Get corrections for current item"""
    if st.session_state.review_items and st.session_state.current_idx < len(st.session_state.review_items):
        extraction_id = st.session_state.review_items[st.session_state.current_idx]['extraction_id']
        if extraction_id not in st.session_state.all_corrections:
            st.session_state.all_corrections[extraction_id] = {}
        return st.session_state.all_corrections[extraction_id]
    return {}

def set_correction(field_name: str, value: str, original_value: str):
    """Set a correction for a field"""
    if st.session_state.review_items and st.session_state.current_idx < len(st.session_state.review_items):
        extraction_id = st.session_state.review_items[st.session_state.current_idx]['extraction_id']
        
        if extraction_id not in st.session_state.all_corrections:
            st.session_state.all_corrections[extraction_id] = {}
        
        if value != original_value:
            st.session_state.all_corrections[extraction_id][field_name] = value
        elif field_name in st.session_state.all_corrections[extraction_id]:
            del st.session_state.all_corrections[extraction_id][field_name]

def load_review_items(force_reload=False):
    """Load review items based on current mode."""
    if force_reload or st.session_state.review_items is None:
        with st.spinner(f"Loading {st.session_state.review_mode} items..."):
            try:
                # Load items based on review mode
                if st.session_state.review_mode == 'pending':
                    items = st.session_state.web_helper.get_review_items_for_web(
                        limit=50, 
                        status_filter='pending'
                    )
                elif st.session_state.review_mode == 'success':
                    items = st.session_state.web_helper.get_review_items_for_web(
                        limit=50, 
                        status_filter='success'
                    )
                else:  # 'all'
                    items = st.session_state.web_helper.get_review_items_for_web(
                        limit=100, 
                        status_filter=None
                    )
                
                if items:
                    st.session_state.review_items = items
                    st.session_state.current_idx = 0
                    return True
                else:
                    st.session_state.review_items = []
                    return False
            except Exception as e:
                logger.error(f"Error loading review items: {e}")
                st.error(f"Failed to load review items: {e}")
                return False
    return True

def get_image_display(extraction_id: int):
    """Get image for display with multiple fallback options."""
    try:
        # Try to get image from storage
        image_data = st.session_state.web_helper.get_image_for_display(extraction_id)
        if image_data:
            return image_data, "stored"
        
        # Try to get base64 encoded image
        image_b64 = st.session_state.web_helper.get_image_as_base64(extraction_id)
        if image_b64:
            return base64.b64decode(image_b64), "base64"
        
        return None, None
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None, None

def render_field_professional(field_name: str, field_data: Dict, extraction_id: int, idx: int, is_success_item: bool = False):
    """Render a field with professional styling and edit tracking."""
    corrections = get_current_corrections()
    original_value = str(field_data.get('value', '') or '')
    current_value = corrections.get(field_name, original_value)
    is_edited = field_name in corrections
    
    # Determine confidence level
    confidence = field_data.get('confidence', 0)
    conf_class = "high-conf" if confidence >= 0.8 else "med-conf" if confidence >= 0.6 else "low-conf"
    conf_icon = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
    
    # Field container with edit indicator
    container_class = "field-container field-edited" if is_edited else "field-container"
    
    col1, col2, col3, col4 = st.columns([3, 5, 2, 1])
    
    with col1:
        field_label = field_name.replace('_', ' ').title()
        if is_edited:
            st.markdown(f"{conf_icon} **{field_label}** ✏️")
        else:
            st.markdown(f"{conf_icon} **{field_label}**")
    
    with col2:
        field_key = f"field_{extraction_id}_{field_name}_{idx}"
        
        # Input field with proper state management
        # Disable editing for success items unless explicitly allowed
        disabled = is_success_item and not st.session_state.get('allow_success_edits', False)
        
        new_val = st.text_input(
            f"Value for {field_name}",
            value=current_value,
            key=field_key,
            label_visibility="collapsed",
            placeholder=f"Enter {field_label}",
            disabled=disabled,
            on_change=lambda: set_correction(
                field_name,
                st.session_state[field_key],
                original_value
            ) if not disabled else None
        )
        
        # Show original value if edited
        if is_edited and original_value:
            st.caption(f"📝 Original: {original_value}")
    
    with col3:
        st.markdown(f'<span class="confidence-badge {conf_class}">{confidence:.0%} confidence</span>', 
                   unsafe_allow_html=True)
    
    with col4:
        if is_edited and not is_success_item:
            if st.button("↩️", key=f"reset_{field_key}", help="Reset to original"):
                set_correction(field_name, original_value, original_value)
                st.rerun()

def submit_review(action: str, item: Dict, notes: str):
    """Submit the review with proper error handling."""
    corrections = get_current_corrections()
    
    # Check if this is a success item being re-reviewed
    is_success_item = item.get('status') == 'success'
    
    # Auto-detect if corrections were made
    if action == 'approve' and corrections:
        action = 'correct'
        st.info(f"📝 Submitting with {len(corrections)} corrections")
    
    try:
        # Different handling for success items
        if is_success_item and action == 'reject':
            # Move successful item back to pending
            success = st.session_state.web_helper.reopen_for_review(
                extraction_id=item['extraction_id'],
                reason=notes or "Re-opened for review",
                reviewer=st.session_state.reviewer
            )
            message = "Item moved back to pending review"
        else:
            success = st.session_state.web_helper.submit_review_from_web(
                extraction_id=item['extraction_id'],
                reviewer=st.session_state.reviewer,
                action=action,
                corrections=corrections if corrections else None,
                notes=notes or f"{action.title()} by {st.session_state.reviewer} at {datetime.now():%Y-%m-%d %H:%M}"
            )
            message = None
        
        if success:
            st.session_state.items_reviewed += 1
            
            # Clear corrections for this item
            if item['extraction_id'] in st.session_state.all_corrections:
                del st.session_state.all_corrections[item['extraction_id']]
            
            # Show success message
            if action == 'reject':
                st.error(message or "❌ Item rejected and marked for reprocessing")
            elif action == 'correct':
                st.success(f"✅ Item approved with {len(corrections)} corrections")
            else:
                st.success("✅ Item approved")
            
            # Progress to next item
            time.sleep(1)
            
            if st.session_state.current_idx < len(st.session_state.review_items) - 1:
                st.session_state.current_idx += 1
                st.rerun()
            else:
                st.balloons()
                st.success("🎉 Batch complete! Loading next batch...")
                st.session_state.review_items = None
                st.session_state.current_idx = 0
                time.sleep(1)
                st.rerun()
        else:
            st.error("❌ Failed to submit review. Please try again.")
            
    except Exception as e:
        logger.error(f"Error submitting review: {e}")
        st.error(f"Error: {str(e)}")

# Professional Header
st.markdown("""
<div class="review-header">
    <h1 style="margin: 0;">Review & Validation Center</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.95;">Verify and correct extracted radar data</p>
</div>
""", unsafe_allow_html=True)

# Reviewer Setup
if not st.session_state.reviewer:
    st.markdown("### 👤 Reviewer Authentication")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        reviewer_input = st.text_input(
            "Enter your name to begin:",
            placeholder="John Smith",
            help="Your name will be logged with all reviews"
        )
    with col2:
        if st.button("🚀 Start Reviewing", type="primary", use_container_width=True):
            if reviewer_input:
                st.session_state.reviewer = reviewer_input
                st.rerun()
            else:
                st.error("Please enter your name")
    
    st.info("💡 **Tip**: Your reviews help improve the extraction accuracy over time")
    st.stop()

# Review Mode Selection
st.markdown("### 📋 Review Mode")
col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

with col1:
    if st.button(
        "🔍 Pending Review", 
        type="primary" if st.session_state.review_mode == 'pending' else "secondary",
        use_container_width=True,
        help="Review items that need validation"
    ):
        if st.session_state.review_mode != 'pending':
            st.session_state.review_mode = 'pending'
            st.session_state.review_items = None
            st.session_state.current_idx = 0
            st.rerun()

with col2:
    if st.button(
        "✅ Successful Items", 
        type="primary" if st.session_state.review_mode == 'success' else "secondary",
        use_container_width=True,
        help="Review already approved items"
    ):
        if st.session_state.review_mode != 'success':
            st.session_state.review_mode = 'success'
            st.session_state.review_items = None
            st.session_state.current_idx = 0
            st.rerun()

with col3:
    if st.button(
        "📊 All Items", 
        type="primary" if st.session_state.review_mode == 'all' else "secondary",
        use_container_width=True,
        help="Review all items regardless of status"
    ):
        if st.session_state.review_mode != 'all':
            st.session_state.review_mode = 'all'
            st.session_state.review_items = None
            st.session_state.current_idx = 0
            st.rerun()

# Current mode indicator
mode_badges = {
    'pending': '<span class="mode-badge mode-pending">Reviewing Pending Items</span>',
    'success': '<span class="mode-badge mode-success">Reviewing Successful Items</span>',
    'all': '<span class="mode-badge mode-all">Reviewing All Items</span>'
}
st.markdown(mode_badges[st.session_state.review_mode], unsafe_allow_html=True)

# Session Information Bar
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Reviewer", st.session_state.reviewer)
with col2:
    st.metric("Items Reviewed", st.session_state.items_reviewed)
with col3:
    session_time = (datetime.now() - st.session_state.session_start).seconds // 60
    st.metric("Session Time", f"{session_time} min")
with col4:
    if st.button("🔄 Refresh Queue", use_container_width=True):
        st.session_state.review_items = None
        st.rerun()

# Load review items
if not load_review_items():
    mode_text = {
        'pending': "Excellent! No items need review at this time.",
        'success': "No successful items to review.",
        'all': "No items available."
    }
    
    st.markdown(f"""
    <div class="status-indicator status-{'success' if st.session_state.review_mode == 'pending' else 'info'}">
        {mode_text[st.session_state.review_mode]}
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Check for New Items", type="primary", use_container_width=True):
            st.session_state.review_items = None
            st.rerun()
    st.stop()

# Get current item
if st.session_state.review_items and st.session_state.current_idx < len(st.session_state.review_items):
    current_item = st.session_state.review_items[st.session_state.current_idx]
    total_items = len(st.session_state.review_items)
    is_success_item = current_item.get('status') == 'success'
else:
    st.error("No items available for review")
    st.stop()

# Success item banner
if is_success_item:
    st.markdown("""
    <div class="success-item-banner">
        ✅ This is a SUCCESSFUL extraction - Review for quality assurance
    </div>
    """, unsafe_allow_html=True)
    
    # Option to enable editing for success items
    st.session_state.allow_success_edits = st.checkbox(
        "Enable field editing for this successful item",
        value=False,
        help="Check to make corrections to this already approved item"
    )

# Progress indicator
progress = (st.session_state.current_idx + 1) / total_items
st.markdown(f"""
<div class="progress-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span><strong>Item {st.session_state.current_idx + 1} of {total_items}</strong></span>
        <span>ID: {current_item['extraction_id']} | Status: {current_item.get('status', 'pending').upper()}</span>
        <span>{current_item.get('radar_type', 'Unknown Type')}</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.progress(progress)

# Get current corrections
corrections = get_current_corrections()

# Main Review Interface
col_left, col_right = st.columns([5, 7])

# LEFT COLUMN - Image Display
with col_left:
    st.markdown("### 📸 Original Radar Image")
    
    with st.container():
        # Get image from storage/database
        image_data, source = get_image_display(current_item['extraction_id'])
        
        if image_data:
            try:
                # Display the stored image
                st.image(image_data, use_container_width=True, caption=f"Source: {source}")
                
                # Image info
                st.success(f"✅ Image loaded from {source}")
                
            except Exception as e:
                logger.error(f"Error displaying image: {e}")
                st.error("Failed to display image")
                
                # Fallback upload option
                st.warning("⚠️ Error displaying stored image")
                uploaded = st.file_uploader(
                    "Upload image for comparison",
                    type=['png', 'jpg', 'jpeg'],
                    key=f"fallback_upload_{current_item['extraction_id']}",
                    help="Upload the original radar image for reference"
                )
                if uploaded:
                    st.image(uploaded, use_container_width=True)
        else:
            # No stored image - provide upload option
            st.warning("⚠️ Original image not found in storage")
            
            uploaded = st.file_uploader(
                "Upload radar image for comparison",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key=f"manual_upload_{current_item['extraction_id']}_{st.session_state.current_idx}",
                help="Upload the original radar image to compare with extracted values"
            )
            
            if uploaded:
                st.image(uploaded, use_container_width=True, caption="Uploaded for reference")
                st.info("💡 This image is for reference only and won't be saved")
            else:
                # Placeholder
                st.info("📤 Upload the radar image to compare with extracted values")
                st.markdown("""
                <div style="background: #f1f5f9; padding: 4rem; text-align: center; 
                           border-radius: 12px; border: 2px dashed #cbd5e1;">
                    <h3 style="color: #64748b;">No Image Available</h3>
                    <p style="color: #94a3b8;">Upload an image or review fields based on data</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Image metadata
    with st.expander("📊 Extraction Metadata"):
        st.markdown(f"**Filename**: {current_item.get('filename', 'Unknown')}")
        st.markdown(f"**Radar Type**: {current_item.get('radar_type', 'Unknown')}")
        st.markdown(f"**Overall Confidence**: {current_item.get('overall_confidence', 0):.1%}")
        st.markdown(f"**Timestamp**: {current_item.get('timestamp', 'Unknown')}")
        st.markdown(f"**Status**: {current_item.get('status', 'pending').upper()}")
        if current_item.get('reviewed_by'):
            st.markdown(f"**Previously Reviewed By**: {current_item.get('reviewed_by')}")

# RIGHT COLUMN - Field Review
with col_right:
    st.markdown("### 📝 Review & Edit Fields")
    
    # Corrections indicator
    if corrections:
        st.markdown(f"""
        <div class="status-indicator status-warning">
            ✏️ You have edited {len(corrections)} field(s)
        </div>
        """, unsafe_allow_html=True)
    
    # Field tabs
    tabs = st.tabs(["🚨 Priority Review", "🧭 Navigation", "⚙️ Settings", "📊 Display", "📋 All Fields"])
    
    with tabs[0]:  # Priority Review
        st.markdown("#### Fields Requiring Attention")
        
        priority_fields = []
        for field_name, field_data in current_item['fields'].items():
            if field_data.get('confidence', 0) < 0.7 or field_data.get('value') is None:
                priority_fields.append((field_name, field_data))
        
        if priority_fields:
            for i, (field_name, field_data) in enumerate(priority_fields):
                render_field_professional(field_name, field_data, current_item['extraction_id'], 
                                        1000 + i, is_success_item)
        else:
            st.success("✅ All fields meet confidence threshold")
    
    with tabs[1]:  # Navigation
        st.markdown("#### Navigation Data")
        nav_fields = ['heading', 'speed', 'position', 'cog', 'sog', 'set', 'drift']
        for i, field in enumerate(nav_fields):
            if field in current_item['fields']:
                render_field_professional(field, current_item['fields'][field], 
                                        current_item['extraction_id'], 2000 + i, is_success_item)
    
    with tabs[2]:  # Settings
        st.markdown("#### Radar Settings")
        settings_fields = ['gain', 'sea_clutter', 'rain_clutter', 'tune']
        for i, field in enumerate(settings_fields):
            if field in current_item['fields']:
                render_field_professional(field, current_item['fields'][field], 
                                        current_item['extraction_id'], 3000 + i, is_success_item)
    
    with tabs[3]:  # Display
        st.markdown("#### Display Parameters")
        display_fields = ['presentation_mode', 'range', 'range_rings', 'vector', 'vector_duration']
        for i, field in enumerate(display_fields):
            if field in current_item['fields']:
                render_field_professional(field, current_item['fields'][field], 
                                        current_item['extraction_id'], 4000 + i, is_success_item)
    
    with tabs[4]:  # All Fields
        st.markdown("#### Complete Field List")
        
        # Sort fields by name for consistency
        sorted_fields = sorted(current_item['fields'].items())
        
        # Quick stats
        total_fields = len(sorted_fields)
        high_conf = sum(1 for _, f in sorted_fields if f.get('confidence', 0) >= 0.8)
        
        st.markdown(f"**Total**: {total_fields} fields | **High Confidence**: {high_conf}/{total_fields}")
        st.markdown("---")
        
        for i, (field_name, field_data) in enumerate(sorted_fields):
            render_field_professional(field_name, field_data, current_item['extraction_id'], 
                                    5000 + i, is_success_item)

# Review Notes Section
st.markdown("---")
st.markdown("### 📝 Review Notes & Comments")

review_notes = st.text_area(
    "Add any observations or comments about this extraction:",
    height=100,
    placeholder="e.g., Image quality issues, missing fields, incorrect values...",
    key=f"notes_{current_item['extraction_id']}_{st.session_state.current_idx}",
    help="These notes will be saved with your review"
)

# Action Buttons (different for success items)
st.markdown("### ✅ Review Actions")

if is_success_item:
    # Actions for successful items
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if corrections and st.session_state.allow_success_edits:
            if st.button("💾 Save Corrections", type="primary", use_container_width=True):
                submit_review('correct', current_item, review_notes)
        else:
            if st.button("✅ Confirm Quality", type="primary", use_container_width=True):
                st.success("Quality confirmed!")
                if st.session_state.current_idx < total_items - 1:
                    st.session_state.current_idx += 1
                    st.rerun()
    
    with col2:
        if st.button("🔄 Re-open for Review", type="secondary", use_container_width=True):
            if not review_notes:
                st.error("⚠️ Please provide a reason for re-opening in the notes")
            else:
                submit_review('reject', current_item, review_notes)
    
    with col3:
        if st.button("⏭️ Skip", use_container_width=True):
            if st.session_state.current_idx < total_items - 1:
                st.session_state.current_idx += 1
                st.rerun()
            else:
                st.info("Last item in batch")
else:
    # Standard actions for pending items
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        approve_label = f"✅ Approve{f' ({len(corrections)} edits)' if corrections else ''}"
        if st.button(approve_label, type="primary", use_container_width=True, key="approve_btn"):
            submit_review('approve', current_item, review_notes)
    
    with col2:
        if st.button("❌ Reject", type="secondary", use_container_width=True, key="reject_btn"):
            if not review_notes:
                st.error("⚠️ Please provide a reason for rejection in the notes")
            else:
                submit_review('reject', current_item, review_notes)
    
    with col3:
        if st.button("⏭️ Skip", use_container_width=True, key="skip_btn"):
            if st.session_state.current_idx < total_items - 1:
                st.session_state.current_idx += 1
                st.rerun()
            else:
                st.info("Last item in batch")

with col4:
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.session_state.current_idx > 0:
            if st.button("◀️ Previous", use_container_width=True, key="prev_btn"):
                st.session_state.current_idx -= 1
                st.rerun()
    with nav_col2:
        if st.session_state.current_idx < total_items - 1:
            if st.button("Next ▶️", use_container_width=True, key="next_btn"):
                st.session_state.current_idx += 1
                st.rerun()

# Queue Status
st.markdown("---")
remaining = total_items - st.session_state.current_idx - 1
if remaining > 0:
    st.info(f"📋 {remaining} more items in current batch | Mode: {st.session_state.review_mode.upper()}")
else:
    st.warning("📋 Last item in batch - new items will load after submission")

# Sidebar with help and stats
with st.sidebar:
    st.markdown("### 📊 Session Statistics")
    st.metric("Review Mode", st.session_state.review_mode.title())
    st.metric("Items in Queue", total_items)
    st.metric("Current Position", st.session_state.current_idx + 1)
    st.metric("Edits Made", len(corrections))
    
    # Quick filters
    st.markdown("---")
    st.markdown("### 🔍 Quick Filters")
    
    # Stats by status
    if st.session_state.review_items:
        pending_count = sum(1 for item in st.session_state.review_items if item.get('status') == 'pending')
        success_count = sum(1 for item in st.session_state.review_items if item.get('status') == 'success')
        
        st.markdown(f"**Pending**: {pending_count} items")
        st.markdown(f"**Success**: {success_count} items")
    
    # Keyboard shortcuts help
    st.markdown("---")
    st.markdown("### ⌨️ Tips")
    st.markdown("""
    - **Tab**: Navigate between fields
    - **Enter**: Submit field changes
    - Changes are highlighted in yellow
    - Click ↩️ to reset a field
    - Switch modes to review different item types
    - Success items can be re-opened if issues found
    """)
    
    # Debug info (collapsible)
    with st.expander("🔧 Debug Info"):
        st.markdown(f"**Mode**: {st.session_state.review_mode}")
        st.markdown(f"**Extraction ID**: {current_item['extraction_id']}")
        st.markdown(f"**Status**: {current_item.get('status', 'pending')}")
        st.markdown(f"**Corrections**: {corrections}")
        st.markdown(f"**Image Source**: {source if 'source' in locals() else 'N/A'}")
        
        if st.button("Force Reload", key="force_reload"):
            st.session_state.review_items = None
            st.session_state.all_corrections = {}
            st.rerun()