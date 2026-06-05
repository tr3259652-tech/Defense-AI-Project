import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 1. Page Configuration & Title Styling
st.set_page_config(page_title="PROJECT AEGIS // COMMAND FRAME v4.2", layout="wide")

st.markdown("""
    <style>
        :root {
            --bg-dark: #080a10;
            --main-cyan: #00ffff;
            --main-amber: #ffcc00;
            --alert-red: #ff3333;
            --panel-bg: rgba(16, 22, 35, 0.9);
            --border-glow: 0 0 10px rgba(0, 255, 255, 0.5);
            --font-family: 'Courier New', Courier, monospace;
        }
        .stApp {
            background-color: var(--bg-dark);
            color: white;
            font-family: var(--font-family);
        }
        #MainMenu, header, footer {visibility: hidden;}
        h1 {
            color: var(--main-cyan);
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 3px;
            text-shadow: var(--border-glow);
            padding: 20px 0;
            margin: 0;
            border-bottom: 1px solid rgba(0, 255, 255, 0.3);
        }
        h1 span { color: white; }
        .sidebar-header {
            color: var(--main-amber);
            font-weight: bold;
            text-transform: uppercase;
            padding-left: 10px;
            border-bottom: 1px solid rgba(255, 204, 0, 0.3);
            margin-bottom: 10px;
        }
        .telemetry-frame {
            background-color: var(--panel-bg);
            border: 1px solid var(--main-cyan);
            box-shadow: var(--border-glow);
            padding: 15px;
            margin-bottom: 20px;
        }
        .telemetry-header {
            color: var(--main-cyan);
            text-transform: uppercase;
            font-size: 0.9rem;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(0, 255, 255, 0.2);
        }
        .alert-banner {
            background-color: var(--alert-red);
            color: white;
            text-align: center;
            font-weight: bold;
            text-transform: uppercase;
            padding: 12px;
            border: 1px solid black;
            box-shadow: 0 0 15px rgba(255, 51, 51, 0.7);
            margin: 15px 0;
            font-size: 1.1rem;
            letter-spacing: 1px;
        }
        .stSlider, .stFileUploader, .stButton button {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid var(--main-cyan) !important;
            color: var(--main-cyan) !important;
            border-radius: 0 !important;
        }
        .stButton button { text-transform: uppercase; width: 100%; }
        .success-text { color: #00ff00; font-weight: bold; }
        .network-panel, .sigint-panel {
            background-color: var(--panel-bg);
            border: 1px solid var(--main-cyan);
            padding: 15px;
            margin-bottom: 20px;
        }
        .panel-header {
            color: var(--main-cyan);
            text-transform: uppercase;
            font-size: 0.9rem;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(0, 255, 255, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>PROJECT <span>AEGIS</span> // COMMAND FRAME v4.2</h1>", unsafe_allow_html=True)


# 2. Load Model
@st.cache_resource
def load_military_model():
    return YOLO("military_model.pt")


model = load_military_model()

# 3. Sidebar / System Control Panel
with st.sidebar:
    st.markdown("<div class='sidebar-header'>SYSTEM CONTROL // AEGIS_CORE</div>", unsafe_allow_html=True)
    st.write("SYSTEM STATUS: <span class='success-text'>ACTIVE</span>", unsafe_allow_html=True)
    st.write("AI ENGINE: <span class='success-text'>YOLOv8n</span>", unsafe_allow_html=True)
    st.write("GPU UTIL: <span class='success-text'>0% (LOCAL)</span>", unsafe_allow_html=True)
    st.markdown("---")

    conf_threshold = st.slider("TRACKING SENSITIVITY", 0.05, 1.00, 0.20, step=0.05)
    st.markdown("---")
    if st.button("ACTIVATE COUNTERMEASURES"):
        st.error("CMD ACCEPTED: COUNTERMEASURES INITIATED.")

# 4. Main Content Layout
col_main, col_sigint = st.columns([2.5, 1])

with col_main:
    st.markdown("<div class='telemetry-frame'>", unsafe_allow_html=True)
    st.markdown("<div class='telemetry-header'>LIVE SATELLITE TELEMETRY // FEED</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Ingest Target Telemetry Imagery...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing target frames..."):
            results = model.predict(source=bgr_img, conf=conf_threshold)
            detected_count = len(results[0].boxes.data.tolist())
            annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

        # 🔥 FIX: Changed use_column_width to use_container_width to remove the warnings
        st.image(annotated_img, caption="Processed Network Tracking Out", use_container_width=True)

        if detected_count > 0:
            st.markdown("<div class='alert-banner'>⚠️ ALERT: THREAT DETECTED. MULTIPLE ARMOR SIGNATURES MATCHED.</div>",
                        unsafe_allow_html=True)
            st.markdown(f"Targets locked in frame: <span class='success-text'>{detected_count}</span>",
                        unsafe_allow_html=True)

            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                label_name = model.names[cls_id]
                confidence_score = float(box.conf[0].item()) * 100

                st.markdown(f"""
                    <div style='border: 1px solid var(--main-cyan); padding: 8px; background: rgba(0,255,255,0.05); margin-bottom: 5px; font-family: monospace;'>
                        Track: <span style='color: white; font-weight: bold;'>{label_name.upper()}</span> | 
                        Conf: <span style='color: var(--main-amber); font-weight: bold;'>{confidence_score:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='border: 1px solid #00ff00; padding: 10px; color: #00ff00; text-align: center;'>✅ STATUS: Feed stable. Zero threat signatures matched.</div>",
                unsafe_allow_html=True)
    else:
        placeholder_img = Image.new('RGB', (1000, 500), color=(10, 15, 25))
        st.image(placeholder_img, caption="Waiting for data feed...", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_sigint:
    st.markdown("<div class='network-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header'>NETWORK STATUS</div>", unsafe_allow_html=True)
    st.write("<span class='success-text'>SECURE</span>", unsafe_allow_html=True)
    st.write("Active Nodes: 1 (HOST)")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sigint-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header'>SIGINT SUMMARY</div>", unsafe_allow_html=True)
    if uploaded_file is not None and detected_count > 0:
        st.error("GROUND MOVEMENT DETECTED.")
        st.write(f"Inference Time: {results[0].speed['inference']:.1f}ms")
    else:
        st.write("Scanning...")
    st.markdown("</div>", unsafe_allow_html=True)