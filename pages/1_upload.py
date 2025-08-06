# pages/1_Upload.py - Upload and Process Page with Integrated Image Storage

import streamlit as st
import os
import sys
import asyncio
import tempfile
from datetime import datetime
import pandas as pd
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_helpers import get_web_helper

# Page config
st.set_page_config(
    page_title="Upload & Process - Radar Extraction",
    page_icon="📤",
    layout="wide"
)

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'openai': os.getenv('OPENAI_API_KEY'),
        'google': os.getenv('GOOGLE_API_KEY')
    }

if 'web_helper' not in st.session_state:
    st.session_state.web_helper = get_web_helper(st.session_state.api_keys)

# Header
st.title("📤 Upload & Process Radar Images")
st.markdown("Upload radar screenshots for automatic data extraction")

# Check API keys
if not any(st.session_state.api_keys.values()):
    st.error("⚠️ No API keys configured. Please set up at least one API key.")
    st.stop()

# File uploader
uploaded_files = st.file_uploader(
    "Drag and drop radar images here",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    accept_multiple_files=True,
    help="Support PNG, JPG, JPEG, BMP, TIFF formats"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} images ready for processing")
    
    # Process button
    if st.button("🚀 Start Processing", type="primary"):
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Results container
        results_data = []
        results_df = pd.DataFrame()
        results_table = st.empty()
        
        # Process each file
        for idx, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            try:
                # Process image (image storage is handled internally by web_helper)
                async def process_single():
                    return await st.session_state.web_helper.process_single_image_async(temp_path)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(process_single())
                
                # Determine status
                if result['success']:
                    if result['overall_confidence'] > 0.8:
                        status = "✅"
                        action = "Success"
                    elif result['overall_confidence'] > 0.5:
                        status = "⚠️"
                        action = "Review"
                    else:
                        status = "❌"
                        action = "Failed"
                    
                    results_data.append({
                        "Status": status,
                        "Image": uploaded_file.name,
                        "Confidence": f"{result['overall_confidence']:.1%}",
                        "Fields": f"{result['field_count']}/26",
                        "Time": f"{result['processing_time']:.1f}s",
                        "Action": action
                    })
                else:
                    results_data.append({
                        "Status": "❌",
                        "Image": uploaded_file.name,
                        "Confidence": "0%",
                        "Fields": "0/26",
                        "Time": "-",
                        "Action": "Error"
                    })
                
                # Update results table
                results_df = pd.DataFrame(results_data)
                results_table.dataframe(results_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                results_data.append({
                    "Status": "❌",
                    "Image": uploaded_file.name,
                    "Confidence": "0%",
                    "Fields": "0/26",
                    "Time": "-",
                    "Action": f"Error: {str(e)[:30]}..."
                })
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        # Summary
        st.markdown("---")
        st.markdown("### 📊 Summary")
        
        if results_df.empty:
            st.warning("No results to display")
        else:
            col1, col2, col3 = st.columns(3)
            
            total = len(results_df)
            success = len(results_df[results_df['Status'] == '✅'])
            review = len(results_df[results_df['Status'] == '⚠️'])
            failed = len(results_df[results_df['Status'] == '❌'])
            
            with col1:
                st.metric("Success", f"{success}/{total}")
            with col2:
                st.metric("Need Review", review)
            with col3:
                st.metric("Failed", failed)
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if review > 0:
                    if st.button("👁️ Go to Review", type="primary"):
                        st.switch_page("pages/2_Review.py")
            with col2:
                if st.button("📤 Upload More"):
                    st.rerun()
                    
else:
    # Instructions
    st.info("👆 Upload radar images to begin processing")
    
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
        1. **Drag and drop** multiple radar images or click to browse
        2. Supported formats: PNG, JPG, JPEG, BMP, TIFF
        3. Click **Start Processing** to begin
        4. Each image takes 3-5 seconds to process
        5. Review any flagged items after processing
        6. Images are automatically stored for review purposes
        """)

# Sidebar info
with st.sidebar:
    st.markdown("### 📊 Processing Info")
    st.markdown("""
    - ✅ **Success**: >80% confidence
    - ⚠️ **Review**: 50-80% confidence  
    - ❌ **Failed**: <50% confidence
    
    **Fields Extracted**: 26 total
    - Navigation data
    - Radar settings
    - Display parameters
    """)
    
    # Storage info
    st.markdown("### 💾 Storage Info")
    storage_stats = st.session_state.web_helper.get_storage_stats()
    if storage_stats:
        st.markdown(f"**Stored Images**: {storage_stats.get('valid_images', 0)}")
        st.markdown(f"**Total Size**: {storage_stats.get('total_size_mb', 0):.1f} MB")
        
        # Clean up old images button
        if st.button("🗑️ Clean Old Images (>7 days)"):
            cleaned = st.session_state.web_helper.cleanup_old_images(days=7)
            st.success(f"Cleaned {cleaned} old images")