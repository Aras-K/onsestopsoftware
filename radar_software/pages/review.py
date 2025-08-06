# pages/2_Review.py - Review Interface with Image Storage and Field Persistence

import streamlit as st
import os
import sys
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional
import time
import base64
from io import BytesIO
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_helpers import get_web_helper
from radar_extraction_architecture import RADAR_FIELDS

# Page config
st.set_page_config(
    page_title="Review Queue - Radar Extraction",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; font-weight: 500; }
    .confidence-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .high-conf { background: #d4f4dd; color: #1e7e34; }
    .med-conf { background: #fff3cd; color: #856404; }
    .low-conf { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'web_helper' not in st.session_state:
    st.session_state.web_helper = get_web_helper({
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'openai': os.getenv('OPENAI_API_KEY'),
        'google': os.getenv('GOOGLE_API_KEY')
    })

if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# Initialize corrections dict for each extraction_id
if 'all_corrections' not in st.session_state:
    st.session_state.all_corrections = {}

if 'reviewer' not in st.session_state:
    st.session_state.reviewer = ""

if 'review_items' not in st.session_state:
    st.session_state.review_items = None

if 'items_reviewed' not in st.session_state:
    st.session_state.items_reviewed = 0

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
    corrections = get_current_corrections()
    if value != original_value:
        corrections[field_name] = value
    elif field_name in corrections:
        del corrections[field_name]

def load_review_items(force_reload=False):
    """Load review items from database."""
    if force_reload or st.session_state.review_items is None:
        with st.spinner("Loading review queue..."):
            items = st.session_state.web_helper.get_review_items_for_web(limit=100)
            if items:
                st.session_state.review_items = items
                st.session_state.current_idx = 0
                st.success(f"✅ Loaded {len(items)} items for review")
                return True
            else:
                st.session_state.review_items = []
                return False
    return True

def render_field(field_name: str, field_data: Dict, extraction_id: int, idx: int):
    """Render a single field for review with proper state management."""
    col1, col2, col3 = st.columns([3, 5, 2])
    
    with col1:
        icon = "🟢" if field_data['confidence'] >= 0.9 else "🟡" if field_data['confidence'] >= 0.7 else "🔴"
        field_label = field_name.replace('_', ' ').title()
        st.markdown(f"{icon} **{field_label}**")
    
    with col2:
        # Get current corrections for this item
        corrections = get_current_corrections()
        
        # Determine current value (correction or original)
        original_value = field_data['value'] or ""
        current_value = corrections.get(field_name, original_value)
        
        # Create unique key for this field
        field_key = f"field_{extraction_id}_{field_name}_{idx}"
        
        # Text input with on_change callback
        new_val = st.text_input(
            f"Value for {field_name}",
            value=current_value,
            key=field_key,
            label_visibility="collapsed",
            on_change=lambda: set_correction(
                field_name, 
                st.session_state[field_key],
                original_value
            )
        )
        
        # Show original value if changed
        if field_name in corrections and original_value:
            st.caption(f"Original: {original_value}")
    
    with col3:
        conf_class = "high-conf" if field_data['confidence'] >= 0.9 else "med-conf" if field_data['confidence'] >= 0.7 else "low-conf"
        st.markdown(f'<span class="confidence-badge {conf_class}">{field_data["confidence"]:.0%}</span>', unsafe_allow_html=True)

def get_image_from_db(extraction_id: int):
    """Try to get image from database or image storage."""
    try:
        # First try to get from web_helper
        image_data = st.session_state.web_helper.get_image_for_display(extraction_id)
        if image_data:
            return image_data
        
        # Alternative: check if there's a stored image path in the database
        # This would require modification to your database schema to store image paths
        # For now, return None if not found
        return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def submit_review(action: str, item: Dict, notes: str):
    """Submit the review and handle navigation."""
    corrections = get_current_corrections()
    
    if action == 'approve' and corrections:
        action = 'correct'
    
    try:
        success = st.session_state.web_helper.submit_review_from_web(
            extraction_id=item['extraction_id'],
            reviewer=st.session_state.reviewer,
            action=action,
            corrections=corrections if corrections else None,
            notes=notes or f"{action.title()}d by {st.session_state.reviewer}"
        )
        
        if success:
            st.session_state.items_reviewed += 1
            
            # Clear corrections for this item
            if item['extraction_id'] in st.session_state.all_corrections:
                del st.session_state.all_corrections[item['extraction_id']]
            
            if action == 'reject':
                st.warning("❌ Marked for reprocessing")
            else:
                st.success("✅ Review saved successfully!")
            
            # Move to next item
            time.sleep(0.5)
            
            if st.session_state.current_idx < len(st.session_state.review_items) - 1:
                st.session_state.current_idx += 1
                st.rerun()
            else:
                st.balloons()
                st.info("🎉 Current batch complete! Loading more items...")
                st.session_state.review_items = None
                st.session_state.current_idx = 0
                time.sleep(1)
                st.rerun()
        else:
            st.error("Failed to submit review. Please try again.")
            
    except Exception as e:
        st.error(f"Error submitting review: {str(e)}")

def next_item():
    """Move to next item without saving."""
    if st.session_state.current_idx < len(st.session_state.review_items) - 1:
        st.session_state.current_idx += 1
        st.rerun()
    else:
        st.info("This is the last item in the current batch")

# Title
st.title("👁️ Review Queue")

# Debug info in sidebar
with st.sidebar:
    st.markdown("### 🐛 Debug Information")
    st.markdown(f"**Items in queue**: {len(st.session_state.review_items) if st.session_state.review_items else 0}")
    st.markdown(f"**Current index**: {st.session_state.current_idx}")
    st.markdown(f"**Items reviewed**: {st.session_state.items_reviewed}")
    
    corrections = get_current_corrections()
    st.markdown(f"**Current corrections**: {len(corrections)}")
    
    if st.button("🔄 Force Reload Queue"):
        st.session_state.review_items = None
        st.session_state.current_idx = 0
        st.session_state.all_corrections = {}
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 Help")
    st.markdown("""
    - Items load in batches
    - After reviewing all items, new batch loads automatically
    - Changes are saved when you click Approve/Reject
    - Use Force Reload if stuck
    """)

# Get reviewer name
if not st.session_state.reviewer:
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        reviewer_input = st.text_input("Enter your name to begin reviewing:", placeholder="John Doe")
    with col2:
        if st.button("Start", type="primary", use_container_width=True):
            if reviewer_input:
                st.session_state.reviewer = reviewer_input
                st.rerun()
    if not reviewer_input:
        st.stop()

# Show reviewer info
st.markdown(f"**Reviewer**: {st.session_state.reviewer} | **Session Reviews**: {st.session_state.items_reviewed}")

# Load review items
if not load_review_items():
    st.success("🎉 Excellent! No items need review at this time.")
    
    if st.button("🔄 Check for New Items", type="primary"):
        st.session_state.review_items = None
        st.rerun()
    st.stop()

# Get current item
if st.session_state.review_items and st.session_state.current_idx < len(st.session_state.review_items):
    current_item = st.session_state.review_items[st.session_state.current_idx]
    total_items = len(st.session_state.review_items)
else:
    st.error("No items available for review")
    st.stop()

# Progress bar and info
progress = (st.session_state.current_idx + 1) / total_items
st.progress(progress)
st.markdown(f"**Reviewing item {st.session_state.current_idx + 1} of {total_items}** (Extraction ID: {current_item['extraction_id']})")

# Get current corrections
corrections = get_current_corrections()

# Main content
col1, col2 = st.columns([5, 7])

# Left column - Image
with col1:
    st.markdown("### 📸 Radar Image")
    
    # Try to get image from database
    image_data = get_image_from_db(current_item['extraction_id'])
    
    if image_data:
        st.image(image_data, use_container_width=True)
    else:
        # Fallback: allow manual upload for this review session
        st.warning("⚠️ Original image not found in database")
        uploaded = st.file_uploader(
            "Upload radar image for comparison",
            type=['png', 'jpg', 'jpeg'],
            key=f"img_upload_{current_item['extraction_id']}_{st.session_state.current_idx}",
            help="This image won't be saved - for reference only"
        )
        if uploaded:
            st.image(uploaded, use_container_width=True)
        else:
            st.info("Upload the radar image to compare with extracted values")
    
   

# Right column - Fields
with col2:
    st.markdown("### Review Fields")
    
    if corrections:
        st.info(f" You have edited {len(corrections)} field(s)")
    
    tabs = st.tabs(["🚨 Needs Review", "🧭 Navigation", "⚙️ Settings", "📋 All Fields"])
    
    with tabs[0]:
        review_needed = False
        for field_name, field_data in current_item['fields'].items():
            if field_data['confidence'] < 0.8 or field_data['value'] is None:
                review_needed = True
                render_field(field_name, field_data, current_item['extraction_id'], 0)
        
        if not review_needed:
            st.success("✅ No fields need immediate review")
    
    with tabs[1]:
        nav_fields = ['heading', 'speed', 'cog', 'sog', 'set', 'drift']
        for i, field in enumerate(nav_fields):
            if field in current_item['fields']:
                render_field(field, current_item['fields'][field], current_item['extraction_id'], 100 + i)
    
    with tabs[2]:
        settings_fields = ['presentation_mode', 'gain', 'sea_clutter', 'rain_clutter', 'tune', 'range', 'range_rings']
        for i, field in enumerate(settings_fields):
            if field in current_item['fields']:
                render_field(field, current_item['fields'][field], current_item['extraction_id'], 200 + i)
    
    with tabs[3]:
        for i, (field_name, field_data) in enumerate(sorted(current_item['fields'].items())):
            render_field(field_name, field_data, current_item['extraction_id'], 300 + i)

# Notes and actions
st.markdown("---")

# Notes
review_notes = st.text_area(
    "📝 Review Notes (optional):",
    height=100,
    placeholder="Add any comments about this review...",
    key=f"notes_{current_item['extraction_id']}_{st.session_state.current_idx}"
)

# Action buttons
st.markdown("### Actions")
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    if st.button("✅ Approve", type="primary", use_container_width=True, key="approve_btn"):
        submit_review('approve', current_item, review_notes)

with col2:
    if st.button("❌ Reject", use_container_width=True, key="reject_btn"):
        if not review_notes:
            st.error("Please provide a reason in the notes")
        else:
            submit_review('reject', current_item, review_notes)

with col3:
    if st.button("⏭️ Skip", use_container_width=True, key="skip_btn"):
        next_item()

with col4:
    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.session_state.current_idx > 0:
            if st.button("◀️", use_container_width=True, key="prev_btn"):
                st.session_state.current_idx -= 1
                st.rerun()
    with nav_cols[1]:
        if st.session_state.current_idx < total_items - 1:
            if st.button("▶️", use_container_width=True, key="next_btn"):
                next_item()

# Show remaining items info
st.markdown("---")
remaining = total_items - st.session_state.current_idx - 1
if remaining > 0:
    st.info(f" {remaining} more items in current batch")
else:
    st.warning(" Last item in current batch - new items will load after submission")