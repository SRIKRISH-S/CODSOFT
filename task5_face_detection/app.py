"""
Task 5 — Face Detection & Recognition UI (Streamlit)
CODSOFT AI Internship | Author: SRIKRISH-S
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path
from face_engine import FaceDetector, FaceRecognizer, FaceProcessor, FaceDetection

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceVision AI — Detection & Recognition",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #030712 0%, #0c0c1a 50%, #030712 100%);
    color: #e2e8f0;
}

.hero-banner {
    background: linear-gradient(135deg, #059669 0%, #0891b2 50%, #6366f1 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(5, 150, 105, 0.4);
    position: relative;
    overflow: hidden;
}

.hero-title { font-size: 2.8rem; font-weight: 800; color: white; margin: 0; }
.hero-subtitle { font-size: 1rem; color: rgba(255,255,255,0.85); margin-top: 0.5rem; }

.face-card {
    background: rgba(5, 150, 105, 0.08);
    border: 1px solid rgba(5, 150, 105, 0.3);
    border-radius: 16px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.face-card:hover {
    border-color: rgba(5, 150, 105, 0.6);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(5, 150, 105, 0.2);
}

.badge-known {
    background: linear-gradient(135deg, #059669, #0891b2);
    color: white; border-radius: 8px; padding: 3px 12px;
    font-size: 0.75rem; font-weight: 700;
}

.badge-unknown {
    background: linear-gradient(135deg, #dc2626, #f59e0b);
    color: white; border-radius: 8px; padding: 3px 12px;
    font-size: 0.75rem; font-weight: 700;
}

.metric-card {
    background: rgba(5, 150, 105, 0.1);
    border: 1px solid rgba(5, 150, 105, 0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.section-header {
    font-size: 1.4rem; font-weight: 700; color: #34d399;
    border-bottom: 2px solid rgba(5, 150, 105, 0.3);
    padding-bottom: 0.5rem; margin-bottom: 1rem;
}

.info-box {
    background: rgba(8, 145, 178, 0.1);
    border: 1px solid rgba(8, 145, 178, 0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}

[data-testid="stSidebar"] {
    background: rgba(3, 7, 18, 0.98) !important;
    border-right: 1px solid rgba(5, 150, 105, 0.2);
}

.stButton > button {
    background: linear-gradient(135deg, #059669, #0891b2) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(5, 150, 105, 0.5) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.stProgress > div > div { background: linear-gradient(135deg, #059669, #0891b2) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine(method="haar", scale=1.1, neighbors=5, min_size=30, confidence=0.5, tolerance=0.5):
    detector = FaceDetector(method, scale, neighbors, min_size, confidence)
    recognizer = FaceRecognizer(tolerance)
    processor = FaceProcessor(detector, recognizer)
    return detector, recognizer, processor

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👁️ FaceVision AI")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🖼️ Image Detection", "📹 Webcam Live", "👤 Register Face", "📂 Face Database"],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")
    
    detect_method = st.selectbox("Detection Method", ["haar", "dnn"], 
                                  format_func=lambda x: "Haar Cascade (Fast)" if x == "haar" else "DNN (Accurate)")
    scale_factor = st.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05)
    min_neighbors = st.slider("Min Neighbors", 1, 15, 5)
    min_face_size = st.slider("Min Face Size (px)", 10, 100, 30)
    
    st.markdown("---")
    st.markdown("### 🎯 Recognition Settings")
    tolerance = st.slider("Recognition Tolerance", 0.3, 0.8, 0.5, 0.05,
                          help="Lower = stricter matching")
    do_recognize = st.toggle("Enable Recognition", value=True)
    
    st.markdown("---")
    st.markdown(
        '<div style="color: #059669; font-size: 0.8rem; text-align: center;">'
        "CODSOFT AI Internship<br>Task 5 — Face Detection<br>"
        '<a href="https://github.com/SRIKRISH-S/CODSOFT" style="color: #34d399;">GitHub: SRIKRISH-S</a>'
        "</div>",
        unsafe_allow_html=True,
    )

# Load engine
detector, recognizer, processor = load_engine(
    detect_method, scale_factor, min_neighbors, min_face_size, 0.5, tolerance
)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">👁️ FaceVision AI</div>
    <div class="hero-subtitle">Real-Time Face Detection & Recognition · Haar Cascades · OpenCV DNN · dlib Face Encodings</div>
</div>
""", unsafe_allow_html=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def render_detections_table(detections):
    if not detections:
        st.info("No faces detected.")
        return
    for i, det in enumerate(detections):
        badge = (
            f'<span class="badge-known">✅ {det.name} ({det.similarity:.0%})</span>'
            if det.name != "Unknown"
            else '<span class="badge-unknown">❓ Unknown</span>'
        )
        eyes = "👁️ Yes" if det.has_eyes else "—"
        smile = "😊 Yes" if det.has_smile else "—"
        st.markdown(
            f"""
            <div class="face-card">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div>
                        <strong style="color:#34d399">Face #{i+1}</strong> &nbsp; {badge}
                    </div>
                    <div style="color:#94a3b8; font-size:0.8rem;">
                        Pos: ({det.x}, {det.y}) &nbsp;|&nbsp; Size: {det.w}×{det.h}px &nbsp;|&nbsp;
                        Conf: {det.confidence:.0%} &nbsp;|&nbsp; Eyes: {eyes} &nbsp;|&nbsp; Smile: {smile}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    col1, col2, col3, col4 = st.columns(4)
    
    for col, icon, val, label in [
        (col1, "🔍", "Haar + DNN", "Detection Methods"),
        (col2, "🧠", "dlib 128-D", "Face Encodings"),
        (col3, "👤", f"{len(recognizer.registered_names)}", "Registered Faces"),
        (col4, "⚡", "Real-Time", "Webcam Support"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div style="font-size:2rem">{icon}</div>'
            f'<div style="font-size:1.3rem;font-weight:700;color:#34d399">{val}</div>'
            f'<div style="color:#94a3b8;font-size:0.8rem">{label}</div></div>',
            unsafe_allow_html=True,
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">🧪 How It Works</div>', unsafe_allow_html=True)
        steps = [
            ("1. Detection", "Haar Cascade / DNN scans image for face regions using sliding window + feature matching"),
            ("2. Feature Extraction", "128-dimensional face encoding extracted using dlib's ResNet model"),
            ("3. Recognition", "Euclidean distance between encodings determines identity (tolerance threshold)"),
            ("4. Annotation", "Bounding boxes, confidence scores, names, and facial features drawn on output"),
        ]
        for title, desc in steps:
            st.markdown(
                f'<div class="info-box"><strong style="color:#34d399">{title}</strong><br>'
                f'<span style="color:#94a3b8;font-size:0.85rem">{desc}</span></div>',
                unsafe_allow_html=True,
            )
    
    with col_r:
        st.markdown('<div class="section-header">📊 Algorithm Comparison</div>', unsafe_allow_html=True)
        import plotly.graph_objects as go
        
        categories = ["Speed", "Accuracy", "Memory", "Robustness", "Occlusion\nHandling"]
        haar_scores =  [95, 70, 90, 65, 50]
        dnn_scores  =  [75, 92, 65, 90, 80]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=haar_scores, theta=categories, fill='toself',
                                       name='Haar Cascade', line_color='#34d399'))
        fig.add_trace(go.Scatterpolar(r=dnn_scores, theta=categories, fill='toself',
                                       name='DNN Detector', line_color='#60a5fa'))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="rgba(52,211,153,0.2)", color="#94a3b8"),
                angularaxis=dict(gridcolor="rgba(52,211,153,0.2)", color="#e2e8f0"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if recognizer.registered_names:
            st.markdown('<div class="section-header">👤 Registered Faces</div>', unsafe_allow_html=True)
            for name in recognizer.registered_names:
                count = len(recognizer.known_encodings[name])
                st.markdown(
                    f'<div class="face-card"><strong style="color:#34d399">{name}</strong>'
                    f' <span style="color:#94a3b8">— {count} encoding(s)</span></div>',
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: IMAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🖼️ Image Detection":
    st.markdown('<div class="section-header">🖼️ Image Face Detection & Recognition</div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Upload an image (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload any photo to detect and recognize faces",
    )
    
    use_sample = st.checkbox("Use sample webcam-style image (generated)")
    
    if use_sample or uploaded:
        if uploaded:
            pil_img = Image.open(uploaded)
        else:
            # Generate a test pattern image
            blank = np.ones((480, 640, 3), dtype=np.uint8) * 30
            cv2.putText(blank, "Upload a real image with faces", (80, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (52, 211, 153), 2)
            pil_img = cv2_to_pil(blank)
        
        col_orig, col_result = st.columns(2)
        
        with col_orig:
            st.markdown("**📷 Original Image**")
            st.image(pil_img, use_container_width=True)
        
        with st.spinner("🔍 Running face detection AI..."):
            cv2_img = pil_to_cv2(pil_img)
            start = time.time()
            annotated, detections = processor.process(cv2_img, do_recognize)
            elapsed = (time.time() - start) * 1000

        with col_result:
            st.markdown(f"**🎯 Detection Result** — `{elapsed:.1f}ms`")
            st.image(cv2_to_pil(annotated), use_container_width=True)
        
        # Face crops
        crops = processor.extract_face_crops(cv2_img, detections)
        
        st.markdown(f'<div class="section-header">📋 Detection Details — {len(detections)} face(s) found</div>', unsafe_allow_html=True)
        render_detections_table(detections)
        
        if crops:
            st.markdown('<div class="section-header">✂️ Extracted Face Crops</div>', unsafe_allow_html=True)
            crop_cols = st.columns(min(len(crops), 6))
            for i, (crop, det) in enumerate(zip(crops, detections)):
                with crop_cols[i % 6]:
                    label = det.name if det.name != "Unknown" else "❓ Unknown"
                    st.image(cv2_to_pil(crop), caption=label, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: WEBCAM LIVE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📹 Webcam Live":
    st.markdown('<div class="section-header">📹 Live Webcam Detection</div>', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="info-box">📸 <strong>Live Detection Mode</strong>: Uses your webcam to detect and recognize faces in real-time. '
        "Click <strong>Start Camera</strong> to begin.</div>",
        unsafe_allow_html=True,
    )
    
    col_ctrl, col_info = st.columns([2, 1])
    with col_ctrl:
        run = st.toggle("🎥 Start Camera", value=False)
        frame_placeholder = st.empty()
    with col_info:
        stats_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not open webcam. Make sure a camera is connected and not in use by another app.")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            frame_count = 0
            fps_time = time.time()
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera.")
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror
                annotated, detections = processor.process(frame, do_recognize)
                
                # FPS calculation
                frame_count += 1
                elapsed = time.time() - fps_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # FPS overlay
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, annotated.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (52, 211, 153), 2)
                
                frame_placeholder.image(cv2_to_pil(annotated), use_container_width=True)
                
                known = [d for d in detections if d.name != "Unknown"]
                stats_placeholder.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size:1.5rem;font-weight:700;color:#34d399">{len(detections)}</div>
                        <div style="color:#94a3b8">Faces Detected</div>
                        <br>
                        <div style="font-size:1.2rem;font-weight:700;color:#60a5fa">{len(known)}</div>
                        <div style="color:#94a3b8">Recognized</div>
                        <br>
                        <div style="font-size:1rem;font-weight:600;color:#a78bfa">{fps:.1f} FPS</div>
                        <div style="color:#94a3b8">Frame Rate</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                time.sleep(0.03)  # ~30 FPS cap
            
            cap.release()
    else:
        frame_placeholder.markdown(
            '<div style="background:rgba(5,150,105,0.05);border:2px dashed rgba(5,150,105,0.3);'
            'border-radius:12px;padding:4rem;text-align:center;color:#94a3b8">'
            '<div style="font-size:4rem">📷</div>'
            '<div>Toggle the camera switch to start live detection</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: REGISTER FACE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "👤 Register Face":
    st.markdown('<div class="section-header">👤 Register a New Face</div>', unsafe_allow_html=True)
    
    if not recognizer.available:
        st.warning(
            "⚠️ The `face_recognition` library is not installed. "
            "Run: `pip install face-recognition` to enable face registration and recognition."
        )
    
    st.markdown(
        '<div class="info-box">📸 Upload a clear, front-facing photo to register a person. '
        "For best results, use well-lit images with a single face visible.</div>",
        unsafe_allow_html=True,
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        name = st.text_input("👤 Person's Name", placeholder="e.g. John Doe")
        uploaded = st.file_uploader("📸 Upload face photo", type=["jpg", "jpeg", "png"])
        register_btn = st.button("✅ Register Face", use_container_width=True)
    
    with col2:
        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="Preview", use_container_width=True)
    
    if register_btn and name and uploaded:
        pil_img = Image.open(uploaded)
        cv2_img = pil_to_cv2(pil_img)
        
        with st.spinner("Processing face..."):
            # First detect
            detections = detector.detect(cv2_img)
            
        if not detections:
            st.error("❌ No face detected in the image. Please upload a clearer photo.")
        elif len(detections) > 1:
            st.warning(f"⚠️ Multiple faces detected ({len(detections)}). Using the largest face.")
            # Use largest
            detections = [max(detections, key=lambda d: d.w * d.h)]
        
        if detections:
            if recognizer.available:
                success = recognizer.register_face(name.strip(), cv2_img)
                if success:
                    st.success(f"✅ Successfully registered **{name}**!")
                    st.balloons()
                else:
                    st.error("❌ Could not extract face encoding. Try a clearer image.")
            else:
                st.info(
                    f"🔖 Face detected for **{name}** — but encoding requires `face_recognition` library. "
                    "Install it to enable recognition."
                )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FACE DATABASE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📂 Face Database":
    st.markdown('<div class="section-header">📂 Registered Face Database</div>', unsafe_allow_html=True)
    
    if not recognizer.registered_names:
        st.markdown(
            '<div class="info-box" style="text-align:center;padding:3rem;">'
            '<div style="font-size:3rem">👤</div>'
            '<div style="color:#94a3b8;margin-top:1rem">No faces registered yet.<br>'
            'Go to <strong>Register Face</strong> to add people.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.markdown(
            f'<div class="metric-card"><div style="font-size:2rem">👥</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:#34d399">{len(recognizer.registered_names)}</div>'
            f'<div style="color:#94a3b8">People</div></div>',
            unsafe_allow_html=True,
        )
        col2.markdown(
            f'<div class="metric-card"><div style="font-size:2rem">🧠</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:#34d399">{recognizer.registered_count}</div>'
            f'<div style="color:#94a3b8">Total Encodings</div></div>',
            unsafe_allow_html=True,
        )
        col3.markdown(
            f'<div class="metric-card"><div style="font-size:2rem">📐</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:#34d399">128-D</div>'
            f'<div style="color:#94a3b8">Vector Size</div></div>',
            unsafe_allow_html=True,
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        for name in recognizer.registered_names:
            count = len(recognizer.known_encodings.get(name, []))
            col_info, col_del = st.columns([5, 1])
            with col_info:
                st.markdown(
                    f'<div class="face-card">'
                    f'<div style="font-size:1.1rem;font-weight:600;color:#34d399">👤 {name}</div>'
                    f'<div style="color:#94a3b8;font-size:0.85rem">{count} face encoding(s) · 128-dimensional dlib vectors</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Delete", key=f"del_{name}"):
                    recognizer.delete_face(name)
                    st.rerun()
