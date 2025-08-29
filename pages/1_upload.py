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
    
    # Display file info with timestamps
    file_info = []
    total_size = 0
    for file in uploaded_files:
        size_mb = file.size / (1024 * 1024)
        total_size += size_mb
        file_info.append({
            "File": file.name,  # Original filename
            "Size": f"{size_mb:.2f} MB",
            "Type": file.type,
            "Upload Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    with st.expander(f"📁 File Details (Total: {total_size:.2f} MB)"):
        st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)
    
    # Process button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        process_button = st.button(" Start Processing", type="primary", use_container_width=True)
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
            
            # Get the original filename
            original_filename = uploaded_file.name
            
            # Save uploaded file to temporary location
            temp_path = None
            try:
                # Create temp file with same extension as original
                file_extension = os.path.splitext(original_filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name
                
                # Process image with original filename preserved
                result = st.session_state.web_helper.process_single_image(
                    temp_path, 
                    original_filename=original_filename  # Pass original filename
                )
                
                # Determine status
                if result.get('success', False):
                    confidence = result.get('overall_confidence', 0)
                    extraction_timestamp = result.get('extraction_timestamp', datetime.now().isoformat())
                    
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
                    target_count = 0
                    if result.get('success', False) and 'detected_targets' in result:
                        target_count = result.get('detected_targets', {}).get('total', 0)

                    results_data.append({
                        "Status": status,
                        "Image": original_filename,
                        "Confidence": f"{confidence:.1%}",
                        "Fields": f"{fields_extracted}/26",
                        "Targets": str(target_count),  # Add this line
                        "Time": f"{result.get('processing_time', 0):.1f}s",
                        "Timestamp": extraction_timestamp,
                        "Action": action,
                        "extraction_id": result.get('extraction_id')
                    })
                    
                    # Show inline success/warning with timestamp
                    timestamp_str = datetime.fromisoformat(extraction_timestamp).strftime("%H:%M:%S")
                    if confidence > 0.8:
                        st.success(f"✅ [{timestamp_str}] {original_filename}: Processed successfully ({confidence:.1%} confidence)")
                    elif confidence > 0.5:
                        st.warning(f"⚠️ [{timestamp_str}] {original_filename}: Needs review ({confidence:.1%} confidence)")
                    else:
                        st.error(f"❌ [{timestamp_str}] {original_filename}: Failed ({confidence:.1%} confidence)")
                    st.write(f"DEBUG - Detected targets for {original_filename}: {targets}")

                    if targets.get('total', 0) == 0:
                        st.info(f"No targets detected in {original_filename}. This could be due to:")
                        st.write("- Image quality or contrast")
                        st.write("- Detection thresholds need adjustment")
                        st.write("- Radar display type not recognized")
                    else:
                        st.success(f"✅ Detected {targets['total']} targets in {original_filename}")
                    
                    all_visualizations = []
                    # Display target detection results if available
                    if result.get('success', False) and 'detected_targets' in result:
                        targets = result.get('detected_targets', {})
                        if targets.get('total', 0) > 0:
                            all_visualizations.append({
                                'filename': original_filename,
                                'targets': targets,
                                'temp_path': temp_path,
                                'viz_image': viz_pil if 'viz_pil' in locals() else None
                            })
                                            
                     
                      
                            # Create professional visualization
                            st.markdown("### 🎯 Target Detection Analysis")
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🚢 Vessels", targets.get('vessels', 0))
                            with col2:
                                st.metric("🏝️ Landmasses", targets.get('landmasses', 0))
                            with col3:
                                st.metric("⚠️ Obstacles", targets.get('obstacles', 0))
                            with col4:
                                st.metric("Total Detected", targets.get('total', 0))
                            
                            # Side-by-side comparison
                            try:
                                viz = RadarVisualization()
                                viz_image = viz.visualize_targets(temp_path, targets)
                                
                                if viz_image is not None:
                                    # Load original for comparison
                                    original_img = Image.open(temp_path)
                                    
                                    # Convert visualization to PIL
                                    viz_pil = Image.fromarray(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
                                    
                                    # Display side by side
                                    col_orig, col_viz = st.columns(2)
                                    
                                    with col_orig:
                                        st.markdown("**Original Radar Image**")
                                        st.image(original_img, use_column_width=True)
                                    
                                    with col_viz:
                                        st.markdown("**Detected Targets**")
                                        st.image(viz_pil, use_column_width=True)
                                    
                                    # Detailed target list
                                    if targets.get('targets'):
                                        with st.expander("📊 Detailed Target Information"):
                                            # Create a table of targets
                                            target_data = []
                                            for i, target in enumerate(targets['targets'], 1):
                                                target_data.append({
                                                    "ID": i,
                                                    "Type": target['type'].upper(),
                                                    "Range (NM)": f"{target['range_nm']:.1f}",
                                                    "Bearing (°)": f"{target['bearing_deg']:.0f}",
                                                    "Confidence": f"{target['confidence']:.0%}",
                                                    "Moving": "Yes" if target.get('is_moving') else "No",
                                                    "Risk": _assess_risk_level(target['range_nm'])
                                                })
                                            
                                            df_targets = pd.DataFrame(target_data)
                                            st.dataframe(df_targets, use_container_width=True, hide_index=True)
                                            
                                            # Risk summary
                                            critical = sum(1 for t in target_data if t['Risk'] == 'CRITICAL')
                                            high = sum(1 for t in target_data if t['Risk'] == 'HIGH')
                                            
                                            if critical > 0:
                                                st.error(f"⚠️ {critical} target(s) require immediate attention (< 1 NM)")
                                            if high > 0:
                                                st.warning(f"📍 {high} target(s) require close monitoring (< 3 NM)")
                                            if all_visualizations:
                                                st.markdown("### 🎯 All Target Detection Results")
                                                for viz_data in all_visualizations:
                                                    with st.expander(f"Targets for {viz_data['filename']} ({viz_data['targets']['total']} detected)"):
                                                        # Display the visualization here
                                                        pass
                                    # Save annotated image
                                    annotated_filename = f"{os.path.splitext(original_filename)[0]}_annotated.png"
                                    result['annotated_image'] = viz_pil
                                    result['annotated_filename'] = annotated_filename
                                    if 'annotated_image' in result:
 
                                        from io import BytesIO
                                        buf = BytesIO()
                                        result['annotated_image'].save(buf, format='PNG')
                                        byte_data = buf.getvalue()
                                        
                                    st.download_button(
                                        label="📥 Download Annotated Image",
                                        data=byte_data,
                                        file_name=result['annotated_filename'],
                                        mime="image/png"
                                    ) 
                            except Exception as e:
                                logger.error(f"Visualization error: {e}")
                                st.error("Could not generate target visualization")
                            
                else:
                    error_msg = result.get('error', 'Unknown error')
                    current_timestamp = datetime.now().isoformat()
                    
                    results_data.append({
                        "Status": "❌",
                        "Image": original_filename,
                        "Confidence": "0%",
                        "Fields": "0/26",
                        "Targets": "0",  # Add this line
                        "Time": "-",
                        "Timestamp": current_timestamp,
                        "Action": "Error",
                        "extraction_id": None
                    })
                    st.error(f"❌ {original_filename}: {error_msg}")
                
                # Update results table
                results_df = pd.DataFrame(results_data)
                with results_placeholder.container():
                    # Display results with timestamp
                    # Update display_columns (around line 158)
                    display_columns = ['Status', 'Image', 'Confidence', 'Fields', 'Targets', 'Time', 'Action']
                    st.dataframe(
                        results_df[display_columns], 
                        use_container_width=True, 
                        hide_index=True
                    )
                
            except Exception as e:
                logger.error(f"Error processing {original_filename}: {e}")
                st.error(f"Error processing {original_filename}: {str(e)[:100]}")
                
                current_timestamp = datetime.now().isoformat()
                results_data.append({
                    "Status": "❌",
                    "Image": original_filename,
                    "Confidence": "0%",
                    "Fields": "0/26",
                    "Time": "-",
                    "Timestamp": current_timestamp,
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
        completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_text.text(f"✅ Processing complete at {completion_time}! ({elapsed_time:.1f}s total)")
        
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
            
            # Export results option with timestamps
            st.markdown("### 💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Include timestamps in export
                export_df = results_df.copy()
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"extraction_results_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if review > 0:
                    if st.button("👁️ Go to Review Queue", type="primary"):
                        st.switch_page("pages/2_Review.py")
            
            # Option to process more
            if st.button("📤 Upload More Images"):
                st.rerun()
                    
else:
    # Instructions remain the same
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
            - Original filenames are preserved
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
            
            ### Data Tracking:
            - Extraction timestamp recorded
            - Original filenames preserved
            - Images stored for review
            """)

# Sidebar info (rest remains the same)

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