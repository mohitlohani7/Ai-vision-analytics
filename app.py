import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import uuid
from video_processor import analyze_video, analyze_image, process_live_frame
from graphs import generate_class_graph
from report_generator import generate_pdf, get_ai_summary
from groq import Groq
from openai import OpenAI

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Vision Analytics", layout="wide", page_icon="👁️")

# -----------------------------------------------------------------------------
# 2. SESSION STATE INIT
# -----------------------------------------------------------------------------
if "analytics" not in st.session_state:
    st.session_state.analytics = None
if "duration" not in st.session_state:
    st.session_state.duration = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "output_path" not in st.session_state:
    st.session_state.output_path = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None
if "run_live" not in st.session_state:
    st.session_state.run_live = False

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")

    input_source = st.radio(
        "Input Source",
        ("📹 Upload Video", "🖼️ Upload Image", "🔴 Live Webcam Feed")
    )

    st.subheader("AI Config")
    ai_provider = st.selectbox("Provider", ("Groq", "OpenAI"))

    api_key = st.text_input("API Key", type="password")

    # ✅ REAL API VALIDATION
    if api_key and st.button("Verify API"):
        try:
            if ai_provider == "OpenAI":
                client = OpenAI(api_key=api_key)
                client.models.list()
            else:
                client = Groq(api_key=api_key)
                client.models.list()

            st.success("✅ API Key Valid")
        except:
            st.error("❌ Invalid API Key")

# -----------------------------------------------------------------------------
# 4. MAIN TITLE
# -----------------------------------------------------------------------------
st.title("👁️ AI Vision Analytics System")

os.makedirs("temp", exist_ok=True)

# -----------------------------------------------------------------------------
# 5. LIVE MODE (FIXED LOOP)
# -----------------------------------------------------------------------------
if input_source == "🔴 Live Webcam Feed":

    run_live = st.toggle("Start Camera", value=False)
    st.session_state.run_live = run_live

    frame_placeholder = st.empty()

    if run_live:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Camera not found")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed = process_live_frame(frame)
                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

                frame_placeholder.image(rgb, use_container_width=True)

                if not st.session_state.run_live:
                    break

            cap.release()

    else:
        st.info("Turn on camera")

# -----------------------------------------------------------------------------
# 6. FILE UPLOAD MODE
# -----------------------------------------------------------------------------
else:

    uploaded_file = None

    if input_source == "📹 Upload Video":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
        file_type = "video"

    elif input_source == "🖼️ Upload Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
        file_type = "image"

    if uploaded_file:

        # Reset state
        st.session_state.processed = False
        st.session_state.analytics = None

        file_id = str(uuid.uuid4())
        file_path = os.path.join("temp", f"{file_id}")

        if file_type == "video":
            file_path += ".mp4"
        else:
            file_path += ".jpg"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success("File uploaded")

        if st.button("Start Analysis"):

            with st.spinner("Processing..."):

                try:
                    if file_type == "video":
                        output_path = file_path.replace(".mp4", "_out.mp4")
                        analytics, duration = analyze_video(file_path, output_path, None)
                    else:
                        output_path = file_path.replace(".jpg", "_out.jpg")
                        analytics, duration = analyze_image(file_path, output_path)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    analytics = None

                if analytics:
                    st.session_state.analytics = analytics
                    st.session_state.duration = duration
                    st.session_state.output_path = output_path
                    st.session_state.file_type = file_type
                    st.session_state.processed = True

                    graph_path = os.path.join("temp", f"{file_id}_graph.png")
                    generate_class_graph(analytics["class_count"], graph_path)

                    st.session_state.graph_path = graph_path

# -----------------------------------------------------------------------------
# 7. RESULTS
# -----------------------------------------------------------------------------
if st.session_state.processed:

    st.subheader("Results")

    analytics = st.session_state.analytics

    total = sum(analytics["class_count"].values())
    st.metric("Total Objects", total)

    # Output
    if st.session_state.file_type == "video":
        st.video(st.session_state.output_path)
    else:
        st.image(st.session_state.output_path)

    # Graph
    if "graph_path" in st.session_state:
        st.image(st.session_state.graph_path)

    # -----------------------------------------------------------------------------
    # 8. PDF REPORT
    # -----------------------------------------------------------------------------
    if st.button("Generate PDF Report"):

        if not api_key:
            st.warning("API key required")
            st.stop()

        with st.spinner("Generating report..."):

            try:
                ai_text = get_ai_summary(
                    api_key,
                    analytics,
                    st.session_state.duration or 0,
                    ai_provider,
                    "default"
                )

                report_path = os.path.join("temp", "report.pdf")

                generate_pdf(
                    report_path,
                    analytics,
                    st.session_state.duration or 0,
                    st.session_state.graph_path,
                    ai_text
                )

                with open(report_path, "rb") as f:
                    st.download_button("Download Report", f, "report.pdf")

            except Exception as e:
                st.error(f"Report Error: {str(e)}")
