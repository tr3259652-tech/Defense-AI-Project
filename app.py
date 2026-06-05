import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO

# 1. Page Configuration
st.set_page_config(page_title="PROJECT AEGIS // COMMAND FRAME v4.2", layout="wide")


# Helper function to convert image to base64 for CSS backgrounds
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# --- Custom CSS for the full command frame aesthetic ---
# Replace 'your_logo_here.png' with your own asset path, or keep it commented out.
# logo_base64 = image_to_base64(Image.open('your_logo_here.png'))

st.markdown(f"""
    <style>
        /* Define command console color palette */
        :root {{
            --bg-dark: #080a10;
            --main-cyan: #00ffff;
            --main-amber: #ffcc00;
            --alert-red: #ff3333;
            --panel-bg: rgba(16, 22, 35, 0.9);
            --border-glow: 0 0 10px rgba(0, 255, 255, 0.5);
            --font-family: 'Courier New', Courier, monospace;
        }}

        /* Apply global styles */
        .stApp {{
            background-color: var(--bg-dark);
            color: white;
            font-family: var(--font-family);
        }}

        /* Hide default Streamlit elements to maximize screen space */
        #MainMenu, header, footer {{visibility: hidden;}}
        .uploadedFile {{display: none;}} /* Hide default upload file name */

        /* Main title styling */
        h1 {{
            color: var(--main-cyan);
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 3px;
            text-shadow: var(--border-glow);
            padding: 20px 0;
            margin: 0;
            border-bottom: 1px solid rgba(0, 255, 255, 0.3);
        }}
        h1 span {{
            color: white;
        }}

        /* Sidebar panel styling */
        .css-1634bda {{  /* Sidebar container */
            background-color: var(--panel-bg);
            border-right: 1px solid var(--main-cyan);
            padding-top: 20px;
        }}
        .sidebar-header {{
            color: var(--main-amber);
            font-weight: bold;
            text-transform: uppercase;
            padding-left: 10px;
            border-bottom: 1px solid rgba(255, 204, 0, 0.3);
            margin-bottom: 10px;
        }}

        /* Main content panel styling */
        .block-container {{
            padding: 20px;
        }}

        /* Central Telemetry Frame Styling */
        .telemetry-frame {{
            background-color: var(--panel-bg);
            border: 1px solid var(--main-cyan);
            box-shadow: var(--border-glow);
            padding: 10px;
            margin-bottom: 20px;
        }}
        .telemetry-header {{
            color: var(--main-cyan);
            text-transform: uppercase;
            font-size: 0.9rem;
            margin-bottom: 5px;
            border-bottom: 1px solid rgba(0, 255, 255, 0.2);
        }}

        /* Alert Banner styling */
        .alert-banner {{
            background-color: var(--alert-red);
            color: white;
            text-align: center;
            font-weight: bold;
            text-transform: uppercase;
            padding: 8px;
            border: 1px solid black;
            box-shadow: 0 0 15px rgba(255, 51, 51, 0.7);
            margin-bottom: 10px;
        }}

        /* Input widget styling */
        .stSlider, .stFileUploader, .stButton button {{
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid var(--main-cyan) !important;
            color: var(--main-cyan) !important;
            border-radius: 0 !important;
        }}
        .stFileUploader label {{
            color: var(--main-amber) !important;
        }}
        .stButton button {{
            text-transform: uppercase;
            width: 100%;
        }}
        .stButton button:hover {{
            background-color: rgba(0, 255, 255, 0.1) !important;
        }}

        /* Specific text styling */
        .success-text {{ color: #00ff00; }}
        .warning-text {{ color: var(--main-amber); font-weight: bold; }}
        .info-text {{ color: white; }}

        /* Network and SIGINT Panels */
        .network-panel, .sigint-panel {{
            background-color: var(--panel-bg);
            border: 1px solid var(--main-cyan);
            padding: 10px;
            margin-bottom: 20px;
        }}
        .panel-header {{
            color: var(--main-cyan);
            text-transform: uppercase;
            font-size: 0.9rem;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(0, 255, 255, 0.2);
        }}

    </style>
""", unsafe_allow_html=True)

# 2. Page Header
st.markdown("<h1>PROJECT <span>AEGIS</span> // COMMAND FRAME v4.2</h1>", unsafe_allow_html=True)


# 3. Load Model
@st.cache_resource
def load_military_model():
    # Use the same weights file
    return YOLO("military_model.pt")


try:
    model = load_military_model()
except FileNotFoundError:
    st.error("⚠️ CRITICAL ERROR: 'military_model.pt' not found in root directory. System offline.")
    st.stop()

# 4. Sidebar / System Control Panel
with st.sidebar:
    st.markdown("<div class='sidebar-header'>SYSTEM CONTROL // AEGIS_CORE</div>", unsafe_allow_html=True)
    st.write("SYSTEM STATUS: <span class='success-text'>ACTIVE</span>", unsafe_allow_html=True)
    st.write("AI ENGINE: <span class='success-text'>YOLOv8n</span>", unsafe_allow_html=True)
    st.write("GPU UTIL: <span class='success-text'>0% (LOCAL)</span>", unsafe_allow_html=True)
    st.markdown("---")

    # Sensitivity Control
    conf_threshold = st.slider("TRACKING SENSITIVITY", 0.05, 1.00, 0.20, step=0.05)

    st.markdown("---")
    if st.button("ACTIVATE COUNTERMEASURES"):
        st.error("CMD ACCEPTED: COUNTERMEASURES INITIATED.")
    if st.button("SYSTEM REBOOT"):
        st.experimental_rerun()

# 5. Main Content Layout
col_main, col_sigint = st.columns([2.5, 1])

with col_main:
    # --- Live Satellite Telemetry Panel ---
    st.markdown("<div class='telemetry-frame'>", unsafe_allow_html=True)
    st.markdown("<div class='telemetry-header'>LIVE SATELITE TELEMETRY // FEAD</div>", unsafe_allow_html=True)

    # Data Ingestion
    uploaded_file = st.file_uploader("Ingest Target Telemetry Imagery...", type=["jpg", "jpeg", "png"])

    # Placeholder/Raw feed view
    placeholder_img = Image.new('RGB', (1000, 600), color=(10, 15, 25))
    image_view = st.image(placeholder_img, caption="Waiting for data feed...", use_column_width=True)

    if uploaded_file is not None:
        # Reset view with uploaded image immediately
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        image_view.image(image, caption="Data Feed Aquired.", use_column_width=True)

        # Convert RGB to BGR for the YOLO engine
        bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Run active inference
        with st.spinner("Analyzing target frames..."):
            results = model.predict(source=bgr_img, conf=conf_threshold)

            # Robust count extraction
            detected_count = len(results[0].boxes.data.tolist())

            # Update view with annotated image
            annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            image_view.image(annotated_img, caption="Processed Network Tracking Out", use_column_width=True)

            # --- Dynamic Warning System ---
            if detected_count > 0:
                st.markdown(
                    "<div class='alert-banner'>⚠️ ALERT: THREAT DETECTED. MULTIPLE ARMOR SIGNATURES MATCHED.</div>",
                    unsafe_allow_html=True)
                st.write(f"Targets locked in frame: <span class='success-text'>{detected_count}</span>",
                         unsafe_allow_html=True)

                # Threat Details
                threat_cols = st.columns(min(detected_count, 3))  # Max 3 detail columns
                col_idx = 0
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    label_name = model.names[cls_id]
                    confidence_score = float(box.conf[0].item()) * 100

                    if col_idx < len(threat_cols):
                        with threat_cols[col_idx]:
                            st.markdown(f"""
                                <div style='border: 1px solid var(--main-cyan); padding: 5px; background: rgba(0,255,255,0.05)'>
                                    Track: <span style='color: white; font-weight: bold;'>{label_name.upper()}</span><br/>
                                    Conf: <span style='color: var(--main-amber);'>{confidence_score:.1f}%</span>
                                </div>
                            """, unsafe_allow_html=True)
                        col_idx += 1
            else:
                st.markdown(
                    "<div style='border: 1px solid #00ff00; padding: 10px; color: #00ff00; text-align: center;'>✅ STATUS: Feed stable. Zero threat signatures matched.</div>",
                    unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # End Telemetry Frame

with col_sigint:
    # --- Network Status Panel ---
    st.markdown("<div class='network-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header'>NETWORK STATUS</div>", unsafe_allow_html=True)
    st.write("<span class='success-text'>SECURE</span>", unsafe_allow_html=True)
    st.write("Active Nodes: 1 (HOST)")
    st.write("Latency: <span class='success-text'>N/A (Local)</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- SIGINT Panel ---
    st.markdown("<div class='sigint-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header'>SIGINT SUMMARY</div>", unsafe_allow_html=True)

    if uploaded_file is not None and detected_count > 0:
        st.error("GROUND MOVEMENT DETECTED.")
        st.write(f"Inference Time: {results[0].speed['inference']:.1f}ms")
    else:
        st.write("Scanning...")
    st.markdown("</div>", unsafe_allow_html=True)