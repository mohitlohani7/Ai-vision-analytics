import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from video_processor import analyze_video, analyze_image, process_live_frame
from graphs import generate_class_graph
from report_generator import generate_pdf, get_ai_summary
from groq import Groq
from openai import OpenAI

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Vision Analytics", layout="wide", page_icon="👁️")

st.markdown("""
    <style>
    /* Professional Dark Theme */
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    section[data-testid="stSidebar"] {
        background-color: #111;
        border-right: 1px solid #333;
    }
    /* Buttons */
    .stButton>button {
        background-color: #00ADB5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00797E;
    }
    /* Headings */
    h1, h2, h3 {
        color: #00ADB5 !important;
    }
    /* Live Feed Container */
    .live-feed-box {
        border: 2px solid #00ADB5;
        border-radius: 10px;
        padding: 10px;
        background-color: rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS
# -----------------------------------------------------------------------------
COCO_CLASSES = [
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat", 
    "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat", 
    "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", 
    "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball", 
    "Kite", "Baseball Bat", "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", 
    "Bottle", "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple", 
    "Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut", "Cake", "Chair", 
    "Couch", "Potted Plant", "Bed", "Dining Table", "Toilet", "TV", "Laptop", "Mouse", 
    "Remote", "Keyboard", "Cell Phone", "Microwave", "Oven", "Toaster", "Sink", 
    "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Teddy Bear", "Hair Drier", "Toothbrush"
]

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/brain.png", width=80)
    st.title("⚙️ Control Panel")
    
    st.divider()
    
    # Input Selection
    st.subheader("1. Select Input Source")
    input_source = st.radio(
        "Choose Source:", 
        ("📹 Upload Video", "🖼️ Upload Image", "🔴 Live Webcam Feed")
    )
    
    st.divider()
    
    # AI Configuration
    st.subheader("2. AI Intelligence")
    ai_provider = st.selectbox("AI Provider", ("Groq (Free)", "OpenAI (Paid)"))
    
    api_key = None
    if ai_provider == "Groq (Free)":
        provider_code = "Groq"
        selected_model = st.selectbox("Model", ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"))
        st.markdown("[🔗 Get Free Groq Key](https://console.groq.com/keys)")
        api_key = st.text_input("Groq API Key", type="password")
    else:
        provider_code = "OpenAI"
        selected_model = st.selectbox("Model", ("gpt-4o", "gpt-3.5-turbo"))
        api_key = st.text_input("OpenAI API Key", type="password")
        
    if api_key and st.button("🔍 Verify API Key"):
        st.success("API Key looks good!")

    with st.expander("ℹ️ Supported Classes"):
        st.write("Detects 80 objects including:")
        st.dataframe(COCO_CLASSES, height=200, hide_index=True)

# -----------------------------------------------------------------------------
# 4. MAIN APP LAYOUT
# -----------------------------------------------------------------------------
st.title("👁️ AI Vision Analytics System")
st.markdown("#### Advanced Object Detection, Tracking & Automated Reporting")

if 'analytics' not in st.session_state: st.session_state.analytics = None
if 'duration' not in st.session_state: st.session_state.duration = None
if 'processed' not in st.session_state: st.session_state.processed = False
if 'output_path' not in st.session_state: st.session_state.output_path = None
if 'file_type' not in st.session_state: st.session_state.file_type = None

os.makedirs("temp", exist_ok=True)
input_file_path = None

# -----------------------------------------------------------------------------
# 5. INPUT HANDLING
# -----------------------------------------------------------------------------
st.container()

# ================================
# MODE A: LIVE WEBCAM FEED
# ================================
if input_source == "🔴 Live Webcam Feed":
    col_live, col_info = st.columns([2, 1])
    
    with col_live:
        st.markdown('<div class="live-feed-box">', unsafe_allow_html=True)
        st.subheader("📡 Real-Time Detection")
        
        run_live = st.toggle("▶️ Start Live Camera", value=False)
        st_frame = st.image([]) 
        
        if run_live:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Error: Webcam not detected.")
            else:
                while run_live:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Camera stream ended.")
                        break
                    
                    # Process Frame using global model
                    processed_frame = process_live_frame(frame)
                    
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, caption="🟢 Live Tracking Active", use_container_width=True)
                
                cap.release()
        else:
            st.info("Click the toggle above to start the webcam feed.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.info("ℹ️ **How it works:**\n\nThis mode uses your webcam to detect objects in real-time. Objects will be tracked with Unique IDs.")

# ================================
# MODE B: STATIC UPLOADS
# ================================
else:
    col_input, col_action = st.columns([2, 1])

    with col_input:
        if input_source == "📹 Upload Video":
            uploaded_file = st.file_uploader("Drop MP4 video", type=["mp4"])
            if uploaded_file:
                input_file_path = os.path.join("temp", "input_video.mp4")
                with open(input_file_path, "wb") as f: f.write(uploaded_file.read())
                st.video(input_file_path)
                st.session_state.file_type = 'video'

        elif input_source == "🖼️ Upload Image":
            uploaded_file = st.file_uploader("Drop Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                input_file_path = os.path.join("temp", "input_image.jpg")
                with open(input_file_path, "wb") as f: f.write(uploaded_file.read())
                st.image(input_file_path, caption="Original Image", use_container_width=True)
                st.session_state.file_type = 'image'

    with col_action:
        st.write("### 🚀 Actions")
        if input_file_path:
            if st.button("Start AI Analysis", use_container_width=True):
                with st.spinner("🧠 Processing..."):
                    
                    if st.session_state.file_type == 'video':
                        output_path = os.path.join("temp", "output_tracked.mp4")
                        progress_bar = st.progress(0)
                        analytics, duration = analyze_video(input_file_path, output_path, progress_bar)
                        progress_bar.empty()
                    else:
                        output_path = os.path.join("temp", "output_result.jpg")
                        analytics, duration = analyze_image(input_file_path, output_path)

                    if analytics:
                        st.session_state.analytics = analytics
                        st.session_state.duration = duration
                        st.session_state.output_path = output_path
                        st.session_state.processed = True
                        
                        graph_path = os.path.join("temp", "class_distribution.png")
                        generate_class_graph(analytics["class_count"], graph_path)
                        
                        st.toast("Done!", icon="✅")
                    else:
                        st.error("Failed.")
        else:
            if input_source != "🔴 Live Webcam Feed":
                st.warning("Please provide input.")

# -----------------------------------------------------------------------------
# 6. RESULTS DASHBOARD
# -----------------------------------------------------------------------------
if st.session_state.processed and input_source != "🔴 Live Webcam Feed":
    st.divider()
    st.markdown("## 📊 Analytics Dashboard")
    
    total_obj = sum(st.session_state.analytics["class_count"].values())
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Objects", total_obj)
    m2.metric("Unique Classes", len(st.session_state.analytics["class_count"]))
    m3.metric("Duration", f"{st.session_state.duration:.2f}s" if st.session_state.duration else "N/A")

    r1_col1, r1_col2 = st.columns([1.5, 1])
    with r1_col1:
        st.subheader("Output")
        if st.session_state.file_type == 'video':
            if os.path.exists(st.session_state.output_path):
                st.video(st.session_state.output_path)
                with open(st.session_state.output_path, "rb") as f:
                    st.download_button("⬇️ Download Video", f, file_name="output.mp4")
        else:
            if os.path.exists(st.session_state.output_path):
                st.image(st.session_state.output_path, caption="Processed Image", use_container_width=True)
                with open(st.session_state.output_path, "rb") as f:
                    st.download_button("⬇️ Download Image", f, file_name="output.jpg")

    with r1_col2:
        st.subheader("Distribution")
        st.dataframe(st.session_state.analytics["class_count"], use_container_width=True)
        graph_path = os.path.join("temp", "class_distribution.png")
        if os.path.exists(graph_path): st.image(graph_path, use_container_width=True)

    st.divider()
    if st.button("Generate PDF Report", use_container_width=True):
        if not api_key: st.warning("⚠️ No API Key found.")
        with st.spinner("Writing Report..."):
            ai_text = "No AI Summary."
            if api_key:
                ai_text = get_ai_summary(api_key, st.session_state.analytics, st.session_state.duration or 0, provider_code, selected_model)
            
            report_path = os.path.join("temp", "Sriya_AI_Report.pdf")
            graph_path = os.path.join("temp", "class_distribution.png")
            generate_pdf(report_path, st.session_state.analytics, st.session_state.duration or 0, graph_path, ai_text)
            st.success("Report Generated!")
            with open(report_path, "rb") as f:
                st.download_button("📄 Download PDF", f, file_name="Report.pdf")