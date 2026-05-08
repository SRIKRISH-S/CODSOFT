"""
Task 3 — Image Captioning  |  CODSOFT AI Internship
Author: SRIKRISH-S | github.com/SRIKRISH-S/CODSOFT
"""

import streamlit as st
from PIL import Image
import time
import numpy as np
from captioner import ImageCaptioner, CaptionResult

st.set_page_config(page_title="CaptionAI · Image Captioning", page_icon="🖼️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

*, html, body { font-family: 'DM Sans', sans-serif; }
.stApp { background: #fafaf8; color: #1a1a1a; }

[data-testid="stSidebar"] { background: #f0ede8 !important; border-right: 1px solid #e0dbd3; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #3a3530 !important; }

#MainMenu, footer, header { visibility: hidden; }

.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem; color: #1a1a1a; margin: 0;
    letter-spacing: -1px;
}
.page-sub { color: #8a8580; font-size: 0.9rem; margin-top: 0.2rem; font-weight: 400; }

.caption-hero {
    background: linear-gradient(135deg, #ff6b35 0%, #f7c59f 50%, #efefd0 100%);
    border-radius: 20px; padding: 2.5rem;
    box-shadow: 0 8px 32px rgba(255,107,53,0.15);
    margin: 1rem 0;
}
.caption-main {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem; color: #1a1a1a; line-height: 1.4;
}
.caption-quote { font-size: 3rem; color: rgba(255,107,53,0.4); line-height: 0.5; }

.pill {
    display: inline-block; border-radius: 50px;
    padding: 4px 14px; font-size: 0.75rem; font-weight: 600;
    margin: 3px; border: 1px solid;
}
.pill-orange { background: rgba(255,107,53,0.1); border-color: rgba(255,107,53,0.3); color: #ff6b35; }
.pill-gray   { background: rgba(0,0,0,0.05); border-color: rgba(0,0,0,0.12); color: #5a5550; }

.alt-card {
    background: white; border: 1px solid #e8e4de;
    border-radius: 12px; padding: 0.9rem 1.1rem;
    margin: 0.5rem 0; color: #3a3530;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.alt-num { font-size: 0.7rem; font-weight: 700; color: #ff6b35; text-transform: uppercase; letter-spacing: 1px; }

.stat-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.6rem; margin: 0.8rem 0;
}
.stat-cell {
    background: white; border: 1px solid #e8e4de;
    border-radius: 10px; padding: 0.7rem; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.stat-val  { font-size: 1.1rem; font-weight: 700; color: #ff6b35; }
.stat-lbl  { font-size: 0.7rem; color: #8a8580; margin-top: 1px; }

.card {
    background: white; border: 1px solid #e8e4de;
    border-radius: 14px; padding: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin: 0.8rem 0;
}

.stButton > button {
    background: #ff6b35 !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { background: #e55a24 !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(255,107,53,0.4) !important; }

.upload-zone {
    border: 2px dashed #d4cfc8; border-radius: 16px;
    padding: 3rem; text-align: center; background: white;
    color: #8a8580;
}

.step-card {
    background: white; border: 1px solid #e8e4de;
    border-left: 4px solid #ff6b35; border-radius: 0 12px 12px 0;
    padding: 0.9rem 1rem; margin: 0.5rem 0;
}
.step-num  { font-size: 0.7rem; color: #ff6b35; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.step-body { color: #3a3530; font-size: 0.85rem; margin-top: 2px; }

.conf-bar-bg { height: 8px; background: #e8e4de; border-radius: 4px; overflow: hidden; margin: 8px 0; }
.conf-bar    { height: 100%; background: linear-gradient(90deg, #ff6b35, #f7c59f); border-radius: 4px; transition: width 0.6s; }
</style>
""", unsafe_allow_html=True)

# ── Load captioner ────────────────────────────────────────────────────────────
@st.cache_resource
def get_captioner():
    return ImageCaptioner()

cap = get_captioner()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🖼️ CaptionAI")
    st.caption("BLIP · Vision Transformer · Beam Search")
    st.divider()

    page = st.radio("", ["📸 Caption Image", "🔬 Batch Mode", "ℹ️ About BLIP"],
                    label_visibility="collapsed")
    st.divider()

    st.markdown("**Model**")
    model_choice = st.selectbox("", list(ImageCaptioner.MODEL_OPTIONS.keys()),
                                 index=1, label_visibility="collapsed")
    st.divider()

    st.markdown("**Generation**")
    num_captions = st.slider("Captions", 1, 5, 3)
    max_length   = st.slider("Max Length (tokens)", 20, 100, 50)
    num_beams    = st.slider("Beam Width", 1, 8, 5)
    cond_text    = st.text_input("Conditional prompt", placeholder="e.g. 'a photo of'")
    st.divider()

    st.markdown('<div style="text-align:center;color:#b0ab9f;font-size:0.75rem;">CODSOFT AI · Task 3<br>'
                '<a href="https://github.com/SRIKRISH-S/CODSOFT" style="color:#ff6b35;">github.com/SRIKRISH-S</a></div>',
                unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">Caption<span style="color:#ff6b35">AI</span></p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Vision · Language · BLIP Transformer · Beam Search Decoding</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_img(f) -> Image.Image:
    img = Image.open(f)
    return img.convert("RGB") if img.mode != "RGB" else img

def ensure_model():
    if not cap.available:
        st.error("❌ `transformers` or `torch` not installed.\n\nRun: `pip install transformers torch torchvision`")
        st.stop()
    if not cap.is_ready or cap.loaded_model_name != model_choice:
        with st.status(f"⏳ Loading **{model_choice}**…", expanded=True) as s:
            st.write("Downloading from HuggingFace Hub (one-time, ~200–900 MB)…")
            try:
                cap.load_model(model_choice)
                s.update(label=f"✅ {model_choice} ready", state="complete")
            except Exception as e:
                s.update(label=f"❌ Load failed: {e}", state="error")
                st.stop()

def render_result(result: CaptionResult):
    conf_pct = int(result.confidence * 100)
    st.markdown(f"""
    <div class="caption-hero">
        <div class="caption-quote">"</div>
        <div class="caption-main">{result.caption}</div>
        <div class="caption-quote" style="text-align:right">"</div>
        <div style="margin-top:1rem;">
            <span class="pill pill-orange">🤖 {result.model_name}</span>
            <span class="pill pill-gray">⚡ {result.generate_ms:.0f} ms</span>
            <span class="pill pill-gray">📐 {result.image_size[0]}×{result.image_size[1]}</span>
        </div>
        <div style="margin-top:0.6rem;">
            <span style="font-size:0.78rem;color:#5a5550;">Confidence</span>
            <div class="conf-bar-bg"><div class="conf-bar" style="width:{conf_pct}%"></div></div>
            <span style="font-size:0.75rem;color:#ff6b35;font-weight:600;">{conf_pct}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if result.alternatives:
        st.markdown("**Alternative captions:**")
        for i, alt in enumerate(result.alternatives):
            st.markdown(
                f'<div class="alt-card"><div class="alt-num">Variant {i+2}</div>{alt}</div>',
                unsafe_allow_html=True)

    st.text_area("Copy caption ↓", result.caption, height=72, label_visibility="visible")

def render_img_stats(img: Image.Image):
    s = cap.analyze_image(img)
    items = [
        (f"{s['width']}×{s['height']}", "Resolution"),
        (str(s['aspect_ratio']), "Aspect Ratio"),
        (f"{s['brightness']:.0f}", "Brightness"),
        (s['dominant_channel'], "Dominant Color"),
    ]
    cells = "".join(f'<div class="stat-cell"><div class="stat-val">{v}</div><div class="stat-lbl">{l}</div></div>'
                    for v, l in items)
    st.markdown(f'<div class="stat-grid">{cells}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
if page == "📸 Caption Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp","bmp"],
                                 label_visibility="collapsed")

    if not uploaded:
        st.markdown('<div class="upload-zone">📤 Drag & drop or click to upload an image<br>'
                    '<span style="font-size:0.8rem">(JPG · PNG · WEBP · BMP)</span></div>',
                    unsafe_allow_html=True)
    else:
        img = load_img(uploaded)
        col_img, col_cap = st.columns([1, 1.3], gap="large")

        with col_img:
            st.image(img, use_container_width=True, caption="Uploaded image")
            st.markdown("**Image statistics**")
            render_img_stats(img)

        with col_cap:
            gen_btn = st.button("🚀 Generate Caption", use_container_width=True)
            if gen_btn:
                ensure_model()
                with st.spinner("Analysing image…"):
                    try:
                        result = cap.caption(img, conditional_text=cond_text,
                                             num_captions=num_captions,
                                             max_length=max_length, num_beams=num_beams)
                        st.session_state["last_result"] = result
                    except Exception as e:
                        st.error(f"❌ {e}")

            if "last_result" in st.session_state:
                render_result(st.session_state["last_result"])

# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Batch Mode":
    st.markdown("### 🔬 Batch Image Captioning")
    uploads = st.file_uploader("Upload multiple images", type=["jpg","jpeg","png","webp"],
                                accept_multiple_files=True, label_visibility="collapsed")
    if uploads:
        if st.button(f"🚀 Caption All {len(uploads)} Images", use_container_width=True):
            ensure_model()
            prog = st.progress(0)
            for i, f in enumerate(uploads):
                img = load_img(f)
                try:
                    result = cap.caption(img, num_captions=1, max_length=max_length, num_beams=3)
                    with st.expander(f"🖼️ {f.name}", expanded=True):
                        c1, c2 = st.columns([1, 2])
                        with c1: st.image(img, use_container_width=True)
                        with c2: render_result(result)
                except Exception as e:
                    st.error(f"{f.name}: {e}")
                prog.progress((i+1)/len(uploads))
            prog.empty()

# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About BLIP":
    st.markdown("### ℹ️ BLIP — Bootstrapped Language-Image Pre-training")
    col1, col2 = st.columns(2)
    with col1:
        for title, desc in [
            ("1. Patch Embedding", "ViT splits image into 16×16 patches → positional embeddings"),
            ("2. Visual Encoding", "24 transformer layers encode spatial and semantic features"),
            ("3. Cross-Attention", "Language decoder attends over visual tokens"),
            ("4. Beam Decoding", "Generates top-K caption hypotheses in parallel"),
        ]:
            st.markdown(f'<div class="step-card"><div class="step-num">{title}</div>'
                        f'<div class="step-body">{desc}</div></div>', unsafe_allow_html=True)

    with col2:
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=[136.7, 133.1, 91.4],
            y=["BLIP Large", "BLIP Base", "ViT-GPT2"],
            orientation="h",
            marker=dict(color=["#ff6b35","#f7c59f","#e8e4de"],
                        line=dict(color="white", width=1)),
            text=["136.7", "133.1", "91.4"],
            textposition="inside",
        ))
        fig.update_layout(
            title="COCO CIDEr Score (higher = better)",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#1a1a1a", family="DM Sans"),
            xaxis=dict(gridcolor="#f0ede8", title="CIDEr"),
            yaxis=dict(gridcolor="#f0ede8"),
            height=250, margin=dict(l=0,r=0,t=40,b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
| Model | Size | CIDEr | BLEU-4 |
|-------|------|-------|--------|
| BLIP Large | ~990 MB | **136.7** | 38.6 |
| BLIP Base  | ~446 MB | 133.1 | 36.9 |
| ViT-GPT2   | ~330 MB | 91.4  | 26.2 |
""")
