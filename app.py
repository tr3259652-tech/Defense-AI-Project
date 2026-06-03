import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO

# 1. Page Configuration & Stealth Theme Setup
st.set_page_config(layout="wide", page_title="Project Aegis - Tactical Center")

# Native CSS Injection to overwrite default colors to an industrial dark mode
st.markdown("""
    <style>
    .stApp {
        background-color: #050811;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #02040a !important;
        border-right: 2px solid #00e5ff;
    }
    div[data-testid="stMetricValue"] {
        color: #00ff66 !important;
        font-family: 'Courier New', monospace;
    }
    h1, h2, h3 {
        color: #00e5ff !important;
        font-family: 'Segoe UI', sans-serif;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #00e5ff;
        background-color: #0b0f19;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Command Header
st.title("⚡ AEGIS: TACTICAL RECONNAISSANCE INTERFACE")
st.markdown("<p style='color: #64748b; font-size: 1.1rem; margin-top: -15px;'>Local Hardware Engine // Sensor Fusion & Neural Inference Benchmarking</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. Sidebar Panel
st.sidebar.markdown("<h2 style='font-size: 1.2rem; color: #00e5ff;'>🛰️ CONTROL MATRIX</h2>", unsafe_allow_html=True)
confidence_slider = st.sidebar.slider("AI Neural Gate (Sensitivity)", 0.05, 1.0, 0.15)
edge_threshold = st.sidebar.slider("Heuristic Structural Filters", 50, 300, (100, 200))

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-size: 1rem; color: #00ff66;'>📟 TELEMETRY STATUS</h3>", unsafe_allow_html=True)
st.sidebar.code("SYSTEM: ACTIVE\nENGINE: CPU_OPTIMIZED\nPORT: LOCAL_HOST")

# 4. Image Input Link
uploaded_file = st.file_uploader("📂 INITIALIZE SECURE TELEMETRY UPLOAD STREAM", type=["jpg", "jpeg", "png"])

raw_img = None
threat_folder = "data/train/threat"

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif os.path.exists(threat_folder) and len(os.listdir(threat_folder)) > 0:
    all_images = os.listdir(threat_folder)
    raw_img = cv2.imread(os.path.join(threat_folder, all_images[0]))

# 5. Core Analytical Panels
if raw_img is not None:
    col1, col2 = st.columns(2)
    
    # ----------------------------------------------------
    # PIPELINE A: MANUAL HEURISTIC EDGE MODEL
    # ----------------------------------------------------
    with col1:
        st.markdown("### 📡 PIPELINE ALPHA: HEURISTIC PROCESSING")
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, edge_threshold[0], edge_threshold[1])
        st.image(edges, caption="Computed High-Frequency Structural Grid", width=380, clamp=True)
        
        edge_density = (np.sum(edges > 0) / edges.size) * 100
        mean_brightness = np.mean(gray)
        
        m1, m2 = st.columns(2)
        with m1: st.metric("STRUCTURAL DENSITY", f"{edge_density:.2f} %")
        with m2: st.metric("LUMINANCE FACTOR", f"{mean_brightness:.1f} DB")

    # ----------------------------------------------------
    # PIPELINE B: AI NEURAL DEEP LEARNING MODEL
    # ----------------------------------------------------
    with col2:
        st.markdown("### 🤖 PIPELINE BETA: NEURAL AGENT INFERENCE")
        cv2.imwrite("temp_uploaded_recon.jpg", raw_img)
        model = YOLO("yolov8n.pt")
        results = model("temp_uploaded_recon.jpg", device='cpu', conf=confidence_slider, imgsz=80)
        annotated_frame = results[0].plot()
        
        # Line 92 - Fully fixed color channel conversion split safely
        rgb_annotated = cv2.cvtColor(
            annotated_frame, 
            cv2.COLOR_BGR2RGB
        )
        st.image(rgb_annotated, caption="Neural Target Recognition Overlay", width=380)
        
        detected_boxes = len(results[0].boxes)
        if detected_boxes > 0:
            st.markdown(f"<div style='background-color: #450a0a; border: 1px solid #ef4444; padding: 10px; border-radius: 4px; color: #f87171; font-weight: bold;'>🚨 ALARM: POSITIVE THREAT ATTRIBUTES REGISTERED ({detected_boxes} Target Assets)</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background-color: #172554; border: 1px solid #3b82f6; padding: 10px; border-radius: 4px; color: #60a5fa;'>ℹ️ READOUT: FEED STABLE // Signature Matches Baseline Parameters</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # ----------------------------------------------------
    # NATIVE METRICS AND SIMULATED SIGINT GRAPHS
    # ----------------------------------------------------
    st.markdown("### 📊 NATIVE SENSOR SIGINT & REAL-TIME ANALYTICS")
    
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("#### 📡 Local Signal Spectrogram Metrics")
        st.write("Dynamic variance distribution logs tracking structural anomalies:")
        
        base_noise = np.random.randn(20) * 5
        edge_influence = np.linspace(edge_threshold[0]/10, edge_threshold[1]/5, 20)
        simulated_telemetry_wave = np.abs(base_noise + edge_influence + (edge_density * 3))
        
        st.area_chart(simulated_telemetry_wave, height=180, use_container_width=True)

    with g_col2:
        st.markdown("#### 🚨 Tactical Processor Risk Distribution")
        st.write("Simulated computational load across tactical logic threads:")
        
        st.write("Structural Signature Threat Index:")
        st.progress(min(int(edge_density * 10), 100))
        
        st.write("Neural Engine Classification Load:")
        st.progress(int(confidence_slider * 100))
else:
    st.info("📡 Tactical system link offline. Ingest aerial reconnaissance files to initialize live processing matrix.")