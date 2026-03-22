import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import json
from datetime import datetime
from collections import defaultdict

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import PPEDetector
from src.compliance import PPEComplianceChecker
from src.image_utils import load_image, resize_image, draw_bounding_box, save_image


st.set_page_config(
    page_title="PPE Compliance System",
    page_icon="images/favicon.ico" if os.path.exists("images/favicon.ico") else None,
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    model_path = "models/best_model.pt"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return None
    try:
        return PPEDetector(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def initialize_session_state():
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'log_data' not in st.session_state:
        st.session_state.log_data = []


def save_detection_log(image_name, detections, compliance, risk_score):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'image_name': image_name,
        'detected_objects': len(sum(detections.values(), [])),
        'is_compliant': compliance['is_compliant'],
        'missing_ppe': compliance['missing_ppe'],
        'risk_score': risk_score['risk_score'],
        'severity': risk_score['severity']
    }
    st.session_state.log_data.append(log_entry)
    return log_entry


def visualize_detections(image, detections, compliance, risk_score):
    image_display = image.copy()
    
    person_count = len(detections.get('person', []))
    
    for class_name, boxes in detections.items():
        if class_name == 'person':
            color = (0, 255, 0) if compliance['is_compliant'] else (255, 0, 0)
        else:
            color = (0, 255, 0)
        
        for box in boxes:
            x1, y1, x2, y2 = box['box']
            confidence = box['confidence']
            image_display = draw_bounding_box(
                image_display, x1, y1, x2, y2,
                label=class_name,
                confidence=confidence,
                color=color
            )
    
    text_y = 30
    text_info = [
        f"Persons: {person_count}",
        f"Compliant: {str(compliance['is_compliant'])}",
        f"Risk Score: {risk_score['risk_score']}/100",
        f"Severity: {risk_score['severity']}"
    ]
    
    for text in text_info:
        cv2.putText(image_display, text, (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
    
    return image_display


def main():
    initialize_session_state()
    
    st.title("PPE Compliance Detection System")
    st.markdown("Real-time detection and compliance monitoring for construction sites")
    
    detector = load_model()
    if detector is None:
        st.stop()
    
    checker = PPEComplianceChecker()
    
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Detection", "Analytics", "Compliance Rules", "Logs"]
    )
    
    with tab1:
        st.header("Image Detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            area_type = st.selectbox(
                "Select Area Type",
                ["high_risk_area", "standard_area", "office_area"],
                help="Different PPE requirements based on area type"
            )
        
        with col2:
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
        
        upload_tab, camera_tab = st.tabs(["Upload Image", "Camera Capture"])
        
        with upload_tab:
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png"],
                help="Upload a construction site image"
            )
            
            if uploaded_file is not None:
                image = load_image(uploaded_file)
                
                if st.button("Run Detection", key="detect_uploaded"):
                    with st.spinner("Running detection..."):
                        image_resized = resize_image(image, max_size=640)
                        
                        temp_path = f"temp_{uploaded_file.name}"
                        save_image(image_resized, temp_path)
                        
                        detections = detector.detect(
                            temp_path,
                            conf_threshold=conf_threshold
                        )
                        
                        compliance = checker.check_compliance(
                            detections, area_type=area_type
                        )
                        risk_score = checker.calculate_risk_score(
                            detections, area_type=area_type,
                            confidence_threshold=conf_threshold
                        )
                        
                        log_entry = save_detection_log(
                            uploaded_file.name, detections, compliance, risk_score
                        )
                        
                        visualized = visualize_detections(
                            image_resized, detections, compliance, risk_score
                        )
                        
                        st.success("Detection completed!")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.image(visualized, caption="Detection Results", use_column_width=True)
                        
                        with col2:
                            st.subheader("Results")
                            
                            status_color = "green" if compliance['is_compliant'] else "red"
                            st.metric(
                                "Compliance Status",
                                "PASS" if compliance['is_compliant'] else "FAIL",
                                delta=None
                            )
                            
                            st.metric(
                                "Risk Score",
                                f"{risk_score['risk_score']}/100",
                                delta=None
                            )
                            
                            st.metric(
                                "Severity Level",
                                risk_score['severity'],
                                delta=None
                            )
                            
                            # Show required vs detected PPE
                            st.divider()
                            st.subheader("PPE Compliance Details")
                            col_req, col_det = st.columns(2)
                            
                            with col_req:
                                st.write(f"**Required ({area_type}):**")
                                for item in compliance['required_ppe']:
                                    st.write(f"  ✓ {item}")
                            
                            with col_det:
                                st.write("**Detected:**")
                                if compliance['detected_ppe']:
                                    for item in compliance['detected_ppe']:
                                        st.write(f"  ✓ {item}")
                                else:
                                    st.warning("  ✗ No PPE detected")
                            
                            # Show missing PPE
                            if compliance['missing_ppe']:
                                st.error(f"**Missing:** {', '.join(compliance['missing_ppe'])}")
                        
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        
        with camera_tab:
            st.info("Camera capture requires streaming support. Use upload tab for now.")
    
    with tab2:
        st.header("Analytics Dashboard")
        
        if st.session_state.log_data:
            log_df = pd.DataFrame(st.session_state.log_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_detections = len(log_df)
                st.metric("Total Detections", total_detections)
            
            with col2:
                compliant_count = (log_df['is_compliant'] == True).sum()
                compliant_pct = (compliant_count / len(log_df) * 100) if len(log_df) > 0 else 0
                st.metric("Compliant Rate", f"{compliant_pct:.1f}%")
            
            with col3:
                avg_risk = log_df['risk_score'].mean()
                st.metric("Average Risk Score", f"{avg_risk:.1f}")
            
            with col4:
                critical_count = (log_df['severity'] == 'CRITICAL').sum()
                st.metric("Critical Cases", critical_count)
            
            st.subheader("Risk Score Distribution")
            risk_hist = log_df['risk_score'].value_counts().sort_index()
            st.bar_chart(risk_hist)
            
            st.subheader("Compliance Status Over Time")
            log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
            log_df_sorted = log_df.sort_values('timestamp')
            
            compliance_timeline = log_df_sorted[['timestamp', 'is_compliant']].copy()
            st.line_chart(
                compliance_timeline.set_index('timestamp')['is_compliant'].astype(int)
            )
            
            st.subheader("Severity Breakdown")
            severity_counts = log_df['severity'].value_counts()
            st.bar_chart(severity_counts)
            
            with st.expander("Detailed Log"):
                st.dataframe(
                    log_df[['timestamp', 'image_name', 'is_compliant', 'risk_score', 'severity']],
                    use_container_width=True
                )
        else:
            st.info("No detections yet. Run detection on an image to see analytics.")
    
    with tab3:
        st.header("Compliance Rules Configuration")
        
        st.subheader("High Risk Area")
        st.write("Required PPE: helmet, vest, gloves, boots")
        st.write("Penalties: helmet (50), vest (30), boots (20), gloves (10)")
        
        st.subheader("Standard Area")
        st.write("Required PPE: helmet, vest")
        st.write("Penalties: helmet (50), vest (30)")
        
        st.subheader("Office Area")
        st.write("Required PPE: helmet")
        st.write("Penalties: helmet (50)")
        
        st.divider()
        
        st.subheader("Risk Score Levels")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**LOW**")
            st.write("0-24")
            st.write("Compliant")
        
        with col2:
            st.write("**MEDIUM**")
            st.write("25-49")
            st.write("Some violations")
        
        with col3:
            st.write("**HIGH**")
            st.write("50-79")
            st.write("Multiple violations")
        
        with col4:
            st.write("**CRITICAL**")
            st.write("80-100")
            st.write("Unsafe")
    
    with tab4:
        st.header("Detection Logs")
        
        if st.session_state.log_data:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Logs as CSV"):
                    log_df = pd.DataFrame(st.session_state.log_data)
                    csv = log_df.to_csv(index=False)
                    st.download_button(
                        label="Download",
                        data=csv,
                        file_name=f"ppe_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Clear All Logs"):
                    st.session_state.log_data = []
                    st.rerun()
            
            log_df = pd.DataFrame(st.session_state.log_data)
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info("No logs available yet.")


if __name__ == "__main__":
    main()
