"""
Task 5 — Face Detection & Recognition  |  CODSOFT AI Internship
Author: SRIKRISH-S | github.com/SRIKRISH-S/CODSOFT
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from face_engine import FaceDetector, FaceRecognizer, FaceProcessor

st.set_page_config(page_title="FaceVision · Detection AI", page_icon="👁️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

*, html, body { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #f8fffe; color: #0d1117; }

[data-testid="stSidebar"] { background: #f0f9f8 !important; border-right: 1px solid #cef2ee; }
[data-testid="stSidebar"] label { color: #1a3a38 !important; }

#MainMenu, footer, header { visibility: hidden; }

.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem; font-weight: 600; color: #0d1117;
    letter-spacing: -0.5px;
}
.page-badge {
    display: inline-block; background: #0d9488; color: white;
    border-radius: 6px; padding: 3px 12px; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace; margin-left: 0.5rem;
}

.terminal-card {
    background: #0d1117; color: #7ee787; border-radius: 12px;
    padding: 1.2rem; margin: 0.6rem 0; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem; border: 1px solid #21262d;
}
.t-label { color: #58a6ff; font-size: 0.7rem; letter-spacing: 1px; text-transform: uppercase; }
.t-val   { color: #7ee787; font-weight: 600; font-size: 0.9rem; }
.t-dim   { color: #3d444d; }

.face-result {
    border-radius: 12px; padding: 1rem 1.2rem; margin: 0.5rem 0;
    border-left: 4px solid;
}
.face-known   { background: rgba(13,148,136,0.06); border-color: #0d9488; }
.face-unknown { background: rgba(220,38,38,0.05);  border-color: #dc2626; }

.name-badge {
    display: inline-block; border-radius: 6px; padding: 3px 12px;
    font-size: 0.78rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace;
}
.badge-known   { background: rgba(13,148,136,0.15); color: #0d9488; border: 1px solid rgba(13,148,136,0.3); }
.badge-unknown { background: rgba(220,38,38,0.1); color: #dc2626; border: 1px solid rgba(220,38,38,0.2); }

.kpi-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
.kpi-item {
    flex: 1; min-width: 80px;
    background: #f0f9f8; border: 1px solid #cef2ee;
    border-radius: 10px; padding: 0.8rem; text-align: center;
}
.kpi-num { font-size: 1.4rem; font-weight: 700; color: #0d9488; font-family: 'IBM Plex Mono', monospace; }
.kpi-lbl { font-size: 0.68rem; color: #5e7e7b; text-transform: uppercase; letter-spacing: 0.5px; }

.step-block {
    border-left: 3px solid #0d9488; padding: 0.7rem 1rem;
    margin: 0.5rem 0; background: #f0f9f8; border-radius: 0 10px 10px 0;
}
.step-title { font-weight: 700; color: #0d1117; font-size: 0.88rem; }
.step-desc  { color: #5e7e7b; font-size: 0.8rem; margin-top: 2px; }

.stButton > button {
    background: #0d9488 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-family: 'IBM Plex Sans', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { background: #0f766e !important;
    box-shadow: 0 4px 16px rgba(13,148,136,0.35) !important;
    transform: translateY(-1px) !important; }

.upload-zone {
    border: 2px dashed #cef2ee; border-radius: 14px;
    padding: 3rem; text-align: center; color: #5e7e7b; background: #f0f9f8;
}
</style>
""", unsafe_allow_html=True)

# ── Load engine ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine(method, scale, neighbors, min_sz, dnn_conf, tol):
    det = FaceDetector(method, scale, neighbors, min_sz, dnn_conf)
    rec_eng = FaceRecognizer(tol)
    return FaceProcessor(det, rec_eng), rec_eng

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👁️ FaceVision")
    st.markdown('<span style="font-family:monospace;font-size:0.75rem;color:#5e7e7b;">Haar · OpenCV DNN · dlib</span>',
                unsafe_allow_html=True)
    st.divider()

    page = st.radio("", ["🏠 Overview", "🖼️ Image Detection", "📹 Webcam", "👤 Register", "📂 Database"],
                    label_visibility="collapsed")
    st.divider()

    st.markdown("**Detection**")
    method    = st.selectbox("Method", ["haar","dnn"],
                              format_func=lambda x: "Haar Cascade (Fast)" if x=="haar" else "DNN Res10 SSD")
    scale     = st.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05)
    neighbors = st.slider("Min Neighbours", 1, 15, 5)
    min_size  = st.slider("Min Face (px)", 10, 100, 30)
    st.divider()
    st.markdown("**Recognition**")
    tolerance   = st.slider("Tolerance", 0.3, 0.8, 0.5, 0.05)
    do_recognize = st.toggle("Enable Recognition", True)
    st.divider()

    st.markdown('<div style="text-align:center;color:#a8d5d1;font-size:0.75rem;">CODSOFT AI · Task 5<br>'
                '<a href="https://github.com/SRIKRISH-S/CODSOFT" style="color:#0d9488;">github.com/SRIKRISH-S</a></div>',
                unsafe_allow_html=True)

proc, rec_eng = load_engine(method, scale, neighbors, min_size, 0.5, tolerance)

# ── Helpers ───────────────────────────────────────────────────────────────────
def pil2cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv2pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def render_detections(detections, elapsed_ms):
    st.markdown(
        f'<div class="kpi-row">'
        f'<div class="kpi-item"><div class="kpi-num">{len(detections)}</div><div class="kpi-lbl">Faces Found</div></div>'
        f'<div class="kpi-item"><div class="kpi-num">{sum(1 for d in detections if d.name!="Unknown")}</div><div class="kpi-lbl">Recognised</div></div>'
        f'<div class="kpi-item"><div class="kpi-num">{elapsed_ms:.0f}ms</div><div class="kpi-lbl">Detect Time</div></div>'
        f'</div>', unsafe_allow_html=True)

    for i, det in enumerate(detections):
        known = det.name != "Unknown"
        cls = "face-known" if known else "face-unknown"
        badge_cls = "badge-known" if known else "badge-unknown"
        sim_txt = f" · {det.similarity:.0%}" if known and det.similarity > 0 else ""
        eyes = " · 👁 Eyes" if det.has_eyes else ""
        smile = " · 😊 Smile" if det.has_smile else ""
        st.markdown(
            f'<div class="face-result {cls}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">'
            f'<div><strong>Face #{i+1}</strong> &nbsp;'
            f'<span class="name-badge {badge_cls}">{det.name}{sim_txt}</span></div>'
            f'<div style="font-size:0.78rem;color:#5e7e7b;">'
            f'Pos ({det.x},{det.y}) · {det.w}×{det.h}px · Conf {det.confidence:.0%}{eyes}{smile}</div>'
            f'</div></div>', unsafe_allow_html=True)

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">FaceVision<span class="page-badge">AI</span></p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**How It Works**")
        for num, title, desc in [
            ("01", "Image Input",      "Load image or capture webcam frame"),
            ("02", "Haar / DNN Scan",  "Sliding window scans for face regions using trained features"),
            ("03", "ROI Extraction",   "Face bounding boxes cropped and normalised to 128-D"),
            ("04", "Encoding Match",   "Euclidean distance to known encodings determines identity"),
        ]:
            st.markdown(
                f'<div class="step-block">'
                f'<div class="step-title"><span style="color:#0d9488;font-family:monospace">[{num}]</span> {title}</div>'
                f'<div class="step-desc">{desc}</div></div>',
                unsafe_allow_html=True)

        st.markdown('<div class="kpi-row">'
                    '<div class="kpi-item"><div class="kpi-num">2</div><div class="kpi-lbl">Detectors</div></div>'
                    f'<div class="kpi-item"><div class="kpi-num">{len(rec_eng.registered_names)}</div><div class="kpi-lbl">Registered</div></div>'
                    '<div class="kpi-item"><div class="kpi-num">128D</div><div class="kpi-lbl">Encoding</div></div>'
                    '<div class="kpi-item"><div class="kpi-num">30fps</div><div class="kpi-lbl">Webcam</div></div>'
                    '</div>', unsafe_allow_html=True)

    with col2:
        import plotly.graph_objects as go
        cats = ["Speed", "Accuracy", "Memory", "Occlusion", "Robustness"]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[95,70,90,50,65], theta=cats, fill="toself",
                                       name="Haar Cascade", line_color="#0d9488",
                                       fillcolor="rgba(13,148,136,0.15)"))
        fig.add_trace(go.Scatterpolar(r=[75,92,65,80,90], theta=cats, fill="toself",
                                       name="DNN Detector", line_color="#0891b2",
                                       fillcolor="rgba(8,145,178,0.1)"))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100], gridcolor="#cef2ee", color="#5e7e7b"),
                angularaxis=dict(gridcolor="#cef2ee", color="#0d1117"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="#f8fffe", font=dict(color="#0d1117"),
            legend=dict(bgcolor="rgba(0,0,0,0)"), height=340,
            margin=dict(l=20,r=20,t=20,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="terminal-card">'
                    '<div class="t-label">System Status</div>'
                    f'<div class="t-val">◉ ONLINE</div>'
                    f'<div class="t-dim">detector: {method.upper()} CASCADE</div>'
                    f'<div class="t-dim">recognition: {"ENABLED" if do_recognize else "DISABLED"}</div>'
                    f'<div class="t-dim">registered: {len(rec_eng.registered_names)} faces</div>'
                    f'<div class="t-dim">tolerance: {tolerance}</div>'
                    '</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
elif page == "🖼️ Image Detection":
    st.markdown("**Upload Image for Face Detection**")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp","bmp"],
                                 label_visibility="collapsed")

    if not uploaded:
        st.markdown('<div class="upload-zone">📤 Drop an image here<br>'
                    '<span style="font-size:0.8rem">JPG · PNG · WEBP · BMP</span></div>',
                    unsafe_allow_html=True)
    else:
        img = Image.open(uploaded).convert("RGB")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("**Original**")
            st.image(img, use_container_width=True)
        with st.spinner("Detecting faces…"):
            cv_img = pil2cv(img)
            t0 = time.perf_counter()
            annotated, dets = proc.process(cv_img, do_recognize)
            elapsed = (time.perf_counter() - t0) * 1000
        with c2:
            st.markdown(f"**Detected — {len(dets)} face(s)**")
            st.image(cv2pil(annotated), use_container_width=True)

        render_detections(dets, elapsed)

        crops = proc.extract_face_crops(cv_img, dets)
        if crops:
            st.markdown("**Extracted Faces**")
            crop_cols = st.columns(min(len(crops), 6))
            for i, (cr, det) in enumerate(zip(crops, dets)):
                with crop_cols[i % 6]:
                    st.image(cv2pil(cr), caption=det.name, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
elif page == "📹 Webcam":
    st.markdown("**Live Webcam Detection**")
    live = st.toggle("▶ Start Camera", False)
    col_cam, col_stat = st.columns([3, 1])
    frame_ph = col_cam.empty()
    stat_ph  = col_stat.empty()

    if live:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam.")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            t0, fc = time.time(), 0
            while live:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                ann, dets = proc.process(frame, do_recognize)
                fc += 1
                fps = fc / max(time.time() - t0, 0.001)
                cv2.putText(ann, f"FPS {fps:.0f}", (10, ann.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (13,148,136), 2)
                frame_ph.image(cv2pil(ann), use_container_width=True)
                stat_ph.markdown(
                    f'<div class="terminal-card">'
                    f'<div class="t-label">Live Stats</div>'
                    f'<div class="t-val">{len(dets)} faces</div>'
                    f'<div class="t-dim">{sum(1 for d in dets if d.name!="Unknown")} recognised</div>'
                    f'<div class="t-dim">{fps:.0f} FPS</div>'
                    f'</div>', unsafe_allow_html=True)
                time.sleep(0.03)
            cap.release()
    else:
        frame_ph.markdown(
            '<div class="upload-zone" style="min-height:300px;">'
            '<div style="font-size:3rem">📷</div><br>Toggle camera to begin</div>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
elif page == "👤 Register":
    st.markdown("**Register a New Face**")
    if not rec_eng.available:
        st.warning("⚠️ Install `face-recognition`: `pip install cmake dlib face-recognition`")

    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full Name", placeholder="e.g. Alice Smith")
        photo = st.file_uploader("Clear front-facing photo", type=["jpg","jpeg","png"])
        reg_btn = st.button("✅ Register Face", use_container_width=True)
    with c2:
        if photo:
            st.image(Image.open(photo), use_container_width=True, caption="Preview")

    if reg_btn and name and photo:
        img = Image.open(photo).convert("RGB")
        cv_img = pil2cv(img)
        dets = proc.detector.detect(cv_img)
        if not dets:
            st.error("❌ No face detected. Try a clearer, well-lit image.")
        elif len(dets) > 1:
            st.warning(f"⚠️ {len(dets)} faces detected — using the largest.")
        if dets:
            if rec_eng.available:
                ok = rec_eng.register_face(name.strip(), cv_img)
                if ok:
                    st.success(f"✅ **{name}** registered successfully!")
                    st.balloons()
                else:
                    st.error("❌ Could not extract encoding. Try a clearer image.")
            else:
                st.info("Face detected but registration requires `face-recognition` library.")

# ─────────────────────────────────────────────────────────────────────────────
elif page == "📂 Database":
    st.markdown("**Registered Face Database**")
    if not rec_eng.registered_names:
        st.markdown('<div class="upload-zone" style="padding:3rem;text-align:center;">'
                    '<div style="font-size:3rem">👤</div><br>'
                    '<div style="color:#5e7e7b">No faces registered yet.<br>Go to <strong>Register</strong> to add people.</div>'
                    '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="kpi-row">'
            f'<div class="kpi-item"><div class="kpi-num">{len(rec_eng.registered_names)}</div><div class="kpi-lbl">People</div></div>'
            f'<div class="kpi-item"><div class="kpi-num">{rec_eng.registered_count}</div><div class="kpi-lbl">Encodings</div></div>'
            f'<div class="kpi-item"><div class="kpi-num">128D</div><div class="kpi-lbl">Vector Size</div></div>'
            f'</div>', unsafe_allow_html=True)

        for nm in rec_eng.registered_names:
            cnt = len(rec_eng.known_encodings.get(nm, []))
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                f'<div class="face-result face-known">'
                f'<span class="name-badge badge-known">👤 {nm}</span>'
                f' <span style="color:#5e7e7b;font-size:0.8rem">{cnt} encoding(s)</span></div>',
                unsafe_allow_html=True)
            with c2:
                if st.button("🗑️", key=f"del_{nm}"):
                    rec_eng.delete_face(nm)
                    st.rerun()
