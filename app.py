import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 1. Page Configuration & Custom small-caps Styling
st.set_page_config(page_title="project aegis", layout="wide")

st.markdown(
    """
    <h1 style='font-family: "Courier New", Courier, monospace; font-weight: 800; color: #00e5ff; letter-spacing: 2px; text-align: center;'>
        project <span style='font-variant: small-caps; font-size: 1.1em;'>aegis</span>
    </h1>
    """,
    unsafe_allow_html=True
)


# 2. Load the Weights Safely into Memory
@st.cache_resource
def load_military_model():
    return YOLO("military_model.pt")


model = load_military_model()

# 3. Sidebar UI Panel Design
st.sidebar.markdown(
    """
    <div style='border-bottom: 2px solid #1e293b; padding-bottom: 10px; margin-bottom: 20px;'>
        <h3 style='font-family: "Courier New", Courier, monospace; color: #94a3b8; margin: 0; font-size: 1.1rem;'>
            control_panel // <span style='font-variant: small-caps; color: #00e5ff;'>aegis_core</span>
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)

conf_threshold = st.sidebar.slider("Target Confidence Cutoff", 0.05, 1.00, 0.20, step=0.05)

# 4. Target Asset Data Ingestion
uploaded_file = st.file_uploader("Ingest Target Telemetry Imagery...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert RGB to BGR for the YOLO engine calculations
    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.subheader("📡 Core Live Feed Processing Matrix")

    # Run active inference engine
    with st.spinner("Analyzing target frames..."):
        results = model.predict(source=bgr_img, conf=conf_threshold)
        annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

        # 🔥 THE ROBUST FIX: Safely convert tensor values into a standard Python integer count
        detected_count = len(results[0].boxes.data.tolist())

    # Present the processed outputs side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Raw Feed Input", use_column_width=True)
    with col2:
        st.image(annotated_img, caption="Processed Network Tracking Out", use_column_width=True)

    # 5. Dynamic Warning System Output Panel
    st.markdown("---")
    if detected_count > 0:
        st.error(f"🚨 TARGET DETECTION ALERT: Locked onto {detected_count} signature hardware assets in frame.")

        # Loop through each bounding box data matrix to print details
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label_name = model.names[cls_id]
            confidence_score = float(box.conf[0].item()) * 100

            st.info(
                f"📍 Threat Asset Recognized: **{label_name}** | Core Confidence Rating: **{confidence_score:.1f}%**")
    else:
        st.warning("⚠️ STATUS: Feed stable. Zero threat signatures matched above current threshold parameters.")