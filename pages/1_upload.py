# pages/1_Upload.py - Upload and Process Page with Integrated Image Storage (Azure Compatible)

import streamlit as st
import os
import sys
import asyncio
import tempfile
from datetime import datetime
import pandas as pd
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from web_helpers import get_web_helper
except ImportError as e:
    st.error(f"Failed to import web_helpers: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Upload & Process - Radar Extraction",
    page_icon="📤",
    layout="wide"
)

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'anthropic': os.environ.get('ANTHROPIC_API_KEY'),
        'openai': os.environ.get('OPENAI_API_KEY'),
        'google': os.environ.get('GOOGLE_API_KEY')
    }

if 'web_helper' not in st.session_state:
    try:
        st.session_state.web_helper = get_web_helper(st.session_state.api_keys)
    except Exception as e:
        st.error(f"Failed to initialize web helper: {e}")
        st.info("Please check your database connection in Azure Configuration")
        st.stop()

# Header
st.title("📤 Upload & Process Radar Images")
st.markdown("Upload radar screenshots for automatic data extraction")

# Check API keys
if not any(st.session_state.api_keys.values()):
    st.error("⚠️ No API keys configured. Please set up at least one API key in Azure App Service Configuration.")
    with st.expander("How to configure API keys"):
        st.markdown("""
        In Azure App Service Configuration, add:
        - `OPENAI_API_KEY` - For GPT-4 Vision
        - `GOOGLE_API_KEY` - For Gemini Vision
        - `ANTHROPIC_API_KEY` - For Claude Vision (optional)
        """)
    st.stop()

# File uploader
uploaded_files = st.file_uploader(
    "Drag and drop radar images here",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    accept_multiple_files=True,
    help="Support PNG, JPG, JPEG, BMP, TIFF formats"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} image(s) ready for processing")
    
    # Display file info
    file_info = []
    total_size = 0
    for file in uploaded_files:
        size_mb = file.size / (1024 * 1024)
        total_size += size_mb
        file_info.append({
            "File": file.name,
            "Size": f"{size_mb:.2f} MB",
            "Type": file.type
        })
    
    with st.expander(f"📁 File Details (Total: {total_size:.2f} MB)"):
        st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)
    
    # Process button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)
    with col2:
        batch_size = st.selectbox("Batch Size", [1, 2, 4], index=1, help="Process multiple images in parallel")
    with col3:
        save_results = st.checkbox("Save Results", value=True, help="Save extraction results to database")
    
    if process_button:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Results container
        results_data = []
        results_df = pd.DataFrame()
        results_placeholder = st.empty()
        
        # Timing
        start_time = time.time()
        
        # Process each file
        for idx, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (idx) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Save uploaded file to temporary location
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name
                
                # Process image using synchronous wrapper
                result = st.session_state.web_helper.process_single_image(temp_path)
                
                # Determine status
                if result.get('success', False):
                    confidence = result.get('overall_confidence', 0)
                    if confidence > 0.8:
                        status = "✅"
                        action = "Success"
                        status_color = "success"
                    elif confidence > 0.5:
                        status = "⚠️"
                        action = "Review"
                        status_color = "warning"
                    else:
                        status = "❌"
                        action = "Failed"
                        status_color = "error"
                    
                    # Count extracted fields
                    fields_extracted = result.get('field_count', 0)
                    
                    results_data.append({
                        "Status": status,
                        "Image": uploaded_file.name[:30] + "..." if len(uploaded_file.name) > 30 else uploaded_file.name,
                        "Confidence": f"{confidence:.1%}",
                        "Fields": f"{fields_extracted}/26",
                        "Time": f"{result.get('processing_time', 0):.1f}s",
                        "Action": action,
                        "extraction_id": result.get('extraction_id')
                    })
                    
                    # Show inline success/warning
                    if confidence > 0.8:
                        st.success(f"✅ {uploaded_file.name}: Processed successfully ({confidence:.1%} confidence)")
                    elif confidence > 0.5:
                        st.warning(f"⚠️ {uploaded_file.name}: Needs review ({confidence:.1%} confidence)")
                    else:
                        st.error(f"❌ {uploaded_file.name}: Failed ({confidence:.1%} confidence)")
                        
                else:
                    error_msg = result.get('error', 'Unknown error')
                    results_data.append({
                        "Status": "❌",
                        "Image": uploaded_file.name[:30] + "..." if len(uploaded_file.name) > 30 else uploaded_file.name,
                        "Confidence": "0%",
                        "Fields": "0/26",
                        "Time": "-",
                        "Action": "Error",
                        "extraction_id": None
                    })
                    st.error(f"❌ {uploaded_file.name}: {error_msg}")
                
                # Update results table
                results_df = pd.DataFrame(results_data)
                with results_placeholder.container():
                    st.dataframe(
                        results_df[['Status', 'Image', 'Confidence', 'Fields', 'Time', 'Action']], 
                        use_container_width=True, 
                        hide_index=True
                    )
                
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {e}")
                st.error(f"Error processing {uploaded_file.name}: {str(e)[:100]}")
                results_data.append({
                    "Status": "❌",
                    "Image": uploaded_file.name[:30] + "..." if len(uploaded_file.name) > 30 else uploaded_file.name,
                    "Confidence": "0%",
                    "Fields": "0/26",
                    "Time": "-",
                    "Action": "Error",
                    "extraction_id": None
                })
                
            finally:
                # Clean up temporary file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.warning(f"Could not delete temp file: {e}")
        
        # Complete
        progress_bar.progress(1.0)
        elapsed_time = time.time() - start_time
        status_text.text(f"✅ Processing complete! ({elapsed_time:.1f}s total)")
        
        # Summary
        st.markdown("---")
        st.markdown("### 📊 Processing Summary")
        
        if results_df.empty:
            st.warning("No results to display")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            total = len(results_df)
            success = len(results_df[results_df['Status'] == '✅'])
            review = len(results_df[results_df['Status'] == '⚠️'])
            failed = len(results_df[results_df['Status'] == '❌'])
            
            with col1:
                st.metric("✅ Success", f"{success}/{total}", 
                         delta=f"{(success/total*100):.0f}%" if total > 0 else "0%")
            with col2:
                st.metric("⚠️ Need Review", review)
            with col3:
                st.metric("❌ Failed", failed)
            with col4:
                avg_time = elapsed_time / total if total > 0 else 0
                st.metric("⏱️ Avg Time", f"{avg_time:.1f}s")
            
            # Export results option
            st.markdown("### 💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"extraction_results_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if review > 0:
                    if st.button("👁️ Go to Review Queue", type="primary"):
                        st.session_state.page = "review"
                        st.rerun()
            
            # Option to process more
            if st.button("📤 Upload More Images"):
                st.rerun()
                    
else:
    # Instructions
    st.info("👆 Upload radar images to begin processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("ℹ️ Instructions", expanded=True):
            st.markdown("""
            ### How to use:
            1. **Drag and drop** radar images or click to browse
            2. Supported formats: PNG, JPG, JPEG, BMP, TIFF
            3. Click **Start Processing** to begin
            4. Each image takes 3-5 seconds to process
            5. Review any flagged items after processing
            
            ### Quality tips:
            - Use high-resolution images
            - Ensure text is clearly visible
            - Full screen captures work best
            """)
    
    with col2:
        with st.expander("📊 Expected Results", expanded=True):
            st.markdown("""
            ### Fields Extracted:
            - **Navigation**: Heading, Speed, Position
            - **Radar Settings**: Gain, Sea/Rain Clutter
            - **Display**: Range, Presentation Mode
            - **Targets**: AIS, Vector settings
            
            ### Success Rates:
            - ✅ **High confidence**: 80%+ accuracy
            - ⚠️ **Medium**: May need review
            - ❌ **Low**: Manual input needed
            """)

# Sidebar info
with st.sidebar:
    st.markdown("### 📊 Processing Info")
    st.markdown("""
    **Confidence Levels:**
    - ✅ **Success**: >80% confidence
    - ⚠️ **Review**: 50-80% confidence  
    - ❌ **Failed**: <50% confidence
    
    **Total Fields**: 26
    - Navigation data (6)
    - Radar settings (5)
    - Display parameters (8)
    - Target info (7)
    """)
    
    st.markdown("---")
    
    # Storage info
    st.markdown("### 💾 Storage Info")
    try:
        storage_stats = st.session_state.web_helper.get_storage_stats()
        if storage_stats:
            storage_type = storage_stats.get('storage_type', 'Unknown')
            st.markdown(f"**Type**: {storage_type}")
            st.markdown(f"**Images**: {storage_stats.get('valid_images', 0)}")
            st.markdown(f"**Size**: {storage_stats.get('total_size_mb', 0):.1f} MB")
            
            if storage_stats.get('storage_enabled', True):
                # Clean up old images button
                if st.button("🗑️ Clean Old Images"):
                    with st.spinner("Cleaning..."):
                        cleaned = st.session_state.web_helper.cleanup_old_images(days=7)
                        st.success(f"Cleaned {cleaned} old images")
                        time.sleep(1)
                        st.rerun()
            else:
                st.info("Image storage disabled")
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        st.info("Storage info unavailable")
    
    st.markdown("---")
    
    # API Status
    st.markdown("### 🔌 API Status")
    api_count = sum(1 for v in st.session_state.api_keys.values() if v)
    if api_count > 0:
        st.success(f"✅ {api_count} API(s) configured")
    else:
        st.error("❌ No APIs configured")
    
    # Environment info
    st.markdown("---")
    st.markdown("### 🌍 Environment")
    st.markdown(f"**Mode**: {os.environ.get('ENVIRONMENT', 'Development')}")
    if os.environ.get('AZURE_REGION'):
        st.markdown(f"**Region**: {os.environ.get('AZURE_REGION')}")