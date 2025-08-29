# pages/1_Upload.py - Upload and Process Page with Original Filename Preservation

import streamlit as st
import os
import sys
import asyncio
import tempfile
from datetime import datetime
import pandas as pd
import time
import logging
from radar_visualization import RadarVisualization
import cv2
from PIL import Image
from io import BytesIO

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

def _assess_risk_level(range_nm):
    """Assess collision risk based on range."""
    if range_nm < 1.0:
        return "CRITICAL"
    elif range_nm < 3.0:
        return "HIGH"
    elif range_nm < 6.0:
        return "MEDIUM"
    else:
        return "LOW"

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
st.markdown("Upload radar screenshots for automatic data extraction and target detection")

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
            "Type": file.type,
            "Upload Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    with st.expander(f"📁 File Details (Total: {total_size:.2f} MB)"):
        st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)
    
    # Process button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)
    with col2:
        batch_size = st.selectbox("Batch Size", [1, 2, 4], index=1)
    with col3:
        save_results = st.checkbox("Save Results", value=True)
    
    if process_button:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Results container
        results_data = []
        all_visualizations = []  # Store all visualizations here
        
        # Timing
        start_time = time.time()
        
        # Process each file
        for idx, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            original_filename = uploaded_file.name
            temp_path = None
            
            try:
                # Create temp file
                file_extension = os.path.splitext(original_filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name
                
                # Process image
                result = st.session_state.web_helper.process_single_image(
                    temp_path, 
                    original_filename=original_filename
                )
                
                if result.get('success', False):
                    confidence = result.get('overall_confidence', 0)
                    extraction_timestamp = result.get('extraction_timestamp', datetime.now().isoformat())
                    
                    # Determine status
                    if confidence > 0.8:
                        status = "✅"
                        action = "Success"
                    elif confidence > 0.5:
                        status = "⚠️"
                        action = "Review"
                    else:
                        status = "❌"
                        action = "Failed"
                    
                    # Get counts
                    fields_extracted = result.get('field_count', 0)
                    detected_targets = result.get('detected_targets', {})
                    target_count = detected_targets.get('total', 0)
                    
                    # Add to results
                    results_data.append({
                        "Status": status,
                        "Image": original_filename,
                        "Confidence": f"{confidence:.1%}",
                        "Fields": f"{fields_extracted}/26",
                        "Targets": str(target_count),
                        "Time": f"{result.get('processing_time', 0):.1f}s",
                        "Timestamp": extraction_timestamp,
                        "Action": action,
                        "extraction_id": result.get('extraction_id')
                    })
                    
                    # Show status
                    timestamp_str = datetime.fromisoformat(extraction_timestamp).strftime("%H:%M:%S")
                    if confidence > 0.8:
                        st.success(f"✅ [{timestamp_str}] {original_filename}: Processed successfully ({confidence:.1%} confidence)")
                    elif confidence > 0.5:
                        st.warning(f"⚠️ [{timestamp_str}] {original_filename}: Needs review ({confidence:.1%} confidence)")
                    else:
                        st.error(f"❌ [{timestamp_str}] {original_filename}: Failed ({confidence:.1%} confidence)")
                    
                    # Handle target detection visualization
                    if target_count > 0:
                        try:
                            # Create visualization
                            viz = RadarVisualization()
                            viz_image = viz.visualize_targets(temp_path, detected_targets)
                            
                            if viz_image is not None:
                                # Convert to PIL
                                viz_pil = Image.fromarray(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
                                
                                # Store for later display
                                all_visualizations.append({
                                    'filename': original_filename,
                                    'original_path': temp_path,
                                    'viz_image': viz_pil,
                                    'targets': detected_targets,
                                    'confidence': confidence
                                })
                        except Exception as e:
                            logger.error(f"Error creating visualization for {original_filename}: {e}")
                    else:
                        st.info(f"No targets detected in {original_filename}")
                        
                else:
                    # Handle error
                    error_msg = result.get('error', 'Unknown error')
                    results_data.append({
                        "Status": "❌",
                        "Image": original_filename,
                        "Confidence": "0%",
                        "Fields": "0/26",
                        "Targets": "0",
                        "Time": "-",
                        "Timestamp": datetime.now().isoformat(),
                        "Action": "Error",
                        "extraction_id": None
                    })
                    st.error(f"❌ {original_filename}: {error_msg}")
                    
            except Exception as e:
                logger.error(f"Error processing {original_filename}: {e}")
                st.error(f"Error processing {original_filename}: {str(e)[:100]}")
                results_data.append({
                    "Status": "❌",
                    "Image": original_filename,
                    "Confidence": "0%",
                    "Fields": "0/26",
                    "Targets": "0",
                    "Time": "-",
                    "Timestamp": datetime.now().isoformat(),
                    "Action": "Error",
                    "extraction_id": None
                })
                
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        # Don't delete yet if we have visualizations
                        if not all_visualizations or all_visualizations[-1]['original_path'] != temp_path:
                            os.unlink(temp_path)
                    except:
                        pass
        
        # Complete processing
        progress_bar.progress(1.0)
        elapsed_time = time.time() - start_time
        status_text.text(f"✅ Processing complete! ({elapsed_time:.1f}s total)")
        
        # Display results table
        if results_data:
            st.markdown("### 📊 Processing Summary")
            results_df = pd.DataFrame(results_data)
            display_columns = ['Status', 'Image', 'Confidence', 'Fields', 'Targets', 'Time', 'Action']
            st.dataframe(results_df[display_columns], use_container_width=True, hide_index=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            total = len(results_df)
            success = len(results_df[results_df['Status'] == '✅'])
            review = len(results_df[results_df['Status'] == '⚠️'])
            failed = len(results_df[results_df['Status'] == '❌'])
            
            with col1:
                st.metric("✅ Success", f"{success}/{total}")
            with col2:
                st.metric("⚠️ Need Review", review)
            with col3:
                st.metric("❌ Failed", failed)
            with col4:
                avg_time = elapsed_time / total if total > 0 else 0
                st.metric("⏱️ Avg Time", f"{avg_time:.1f}s")
        
        # Display all visualizations
        if all_visualizations:
            st.markdown("### 🎯 Target Detection Results")
            
            for viz_data in all_visualizations:
                with st.expander(f"📸 {viz_data['filename']} - {viz_data['targets']['total']} targets detected"):
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🚢 Vessels", viz_data['targets'].get('vessels', 0))
                    with col2:
                        st.metric("🏝️ Landmasses", viz_data['targets'].get('landmasses', 0))
                    with col3:
                        st.metric("⚠️ Obstacles", viz_data['targets'].get('obstacles', 0))
                    with col4:
                        st.metric("Confidence", f"{viz_data['confidence']:.1%}")
                    
                    # Show visualization
                    st.image(viz_data['viz_image'], caption="Detected Targets", use_column_width=True)
                    
                    # Target details
                    if viz_data['targets'].get('targets'):
                        target_data = []
                        for i, target in enumerate(viz_data['targets']['targets'][:10], 1):
                            target_data.append({
                                "ID": i,
                                "Type": target['type'].upper(),
                                "Range (NM)": f"{target['range_nm']:.1f}",
                                "Bearing (°)": f"{target['bearing_deg']:.0f}",
                                "Confidence": f"{target['confidence']:.0%}",
                                "Risk": _assess_risk_level(target['range_nm'])
                            })
                        df_targets = pd.DataFrame(target_data)
                        st.dataframe(df_targets, use_container_width=True, hide_index=True)
                    
                    # Download button
                    buf = BytesIO()
                    viz_data['viz_image'].save(buf, format='PNG')
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=buf.getvalue(),
                        file_name=f"{os.path.splitext(viz_data['filename'])[0]}_annotated.png",
                        mime="image/png"
                    )
                    
else:
    # Instructions remain the same
    st.info("👆 Upload radar images to begin processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("ℹ️ Instructions", expanded=True):
            st.markdown("""
            
            """)
    
    with col2:
        with st.expander("📊 Expected Results", expanded=True):
            st.markdown("""
            """)

# Sidebar info
with st.sidebar:
    st.markdown("## ⚙️ System Info")
    
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