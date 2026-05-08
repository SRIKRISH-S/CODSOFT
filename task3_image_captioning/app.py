"""
Task 3 — Image Captioning UI (Streamlit)
CODSOFT AI Internship | Author: SRIKRISH-S
"""

import streamlit as st
from PIL import Image
import io
import os
import time
import numpy as np
from pathlib import Path
from captioner import ImageCaptioner, CaptionResult

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CaptionAI — Image Captioning",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #020617 0%, #0a0a1a 40%, #020617 100%);
    color: #e2e8f0;
}

.hero-banner {
    background: linear-gradient(135deg, #0891b2 0%, #7c3aed 50%, #db2777 100%);
    border-radius: 20px;
    padding: 2.8rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(124,58,237,0.35);
    position: relative; overflow: hidden;
}
.hero-banner::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.06) 0%, transparent 60%);
}
.hero-title { font-size: 3rem; font-weight: 800; color: white; margin: 0; letter-spacing: -1px; }
.hero-sub { color: rgba(255,255,255,0.8); font-size: 1rem; margin-top: 0.4rem; }

.caption-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(8,145,178,0.08));
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 18px;
    padding: 1.6rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}
.caption-text {
    font-size: 1.35rem;
    font-weight: 600;
    color: #e2e8f0;
    line-height: 1.5;
    font-style: italic;
}
.caption-quote { color: #7c3aed; font-size: 2rem; line-height: 0.8; }

.alt-caption {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    color: #94a3b8;
    font-size: 0.9rem;
}

.stat-chip {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #c4b5fd;
    margin: 3px;
    font-weight: 500;
}

.img-stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 10px;
    padding: 0.7rem;
    text-align: center;
}

.section-header {
    font-size: 1.3rem; font-weight: 700; color: #a78bfa;
    border-bottom: 2px solid rgba(124,58,237,0.3);
    padding-bottom: 0.4rem; margin-bottom: 1rem;
}

.sample-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    cursor: pointer;
    transition: all 0.2s;
}
.sample-card:hover { border-color: rgba(124,58,237,0.6); background: rgba(124,58,237,0.08); }

[data-testid="stSidebar"] {
    background: rgba(2,6,23,0.98) !important;
    border-right: 1px solid rgba(124,58,237,0.2);
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #0891b2) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(124,58,237,0.5) !important; }

.progress-bar-custom { height: 8px; border-radius: 4px; background: rgba(255,255,255,0.1); overflow: hidden; }
.progress-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #7c3aed, #0891b2); transition: width 0.5s ease; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Captioner ────────────────────────────────────────────────────────────
@st.cache_resource
def get_captioner():
    return ImageCaptioner()

captioner = get_captioner()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🖼️ CaptionAI Settings")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📸 Caption Image", "🔬 Batch Captioning", "📊 Model Info"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🤖 Model Selection")
    
    model_choice = st.selectbox(
        "Caption Model",
        list(ImageCaptioner.MODEL_OPTIONS.keys()),
        index=1,  # BLIP Base default
    )
    
    st.markdown("### ⚙️ Generation Settings")
    num_captions = st.slider("Number of Captions", 1, 5, 3)
    max_length   = st.slider("Max Caption Length", 20, 100, 50)
    num_beams    = st.slider("Beam Search Width", 1, 10, 5, help="Higher = better quality, slower")
    conditional  = st.text_input("Conditional Prompt (optional)", placeholder="e.g. 'a photo of'")

    st.markdown("---")
    st.markdown(
        '<div style="color:#7c3aed;font-size:0.78rem;text-align:center;">'
        "CODSOFT AI Internship<br>Task 3 — Image Captioning<br>"
        '<a href="https://github.com/SRIKRISH-S/CODSOFT" style="color:#a78bfa;">GitHub: SRIKRISH-S</a>'
        "</div>",
        unsafe_allow_html=True,
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🖼️ CaptionAI</div>
    <div class="hero-sub">Vision + Language · BLIP Transformer · Beam Search · Conditional Captioning</div>
</div>
""", unsafe_allow_html=True)

# ── Helper: load & validate image ────────────────────────────────────────────
def load_image(uploaded) -> Image.Image:
    img = Image.open(uploaded)
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    return img

def render_caption_result(result: CaptionResult):
    # Primary caption
    st.markdown(
        f"""
        <div class="caption-card">
            <div class="caption-quote">"</div>
            <div class="caption-text">{result.caption}</div>
            <div class="caption-quote" style="text-align:right">"</div>
            <div style="margin-top:0.8rem;">
                <span class="stat-chip">🤖 {result.model_name}</span>
                <span class="stat-chip">⚡ {result.generate_ms:.0f} ms</span>
                <span class="stat-chip">📐 {result.image_size[0]}×{result.image_size[1]}</span>
                <span class="stat-chip">🎯 Confidence: {result.confidence:.0%}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence bar
    conf = result.confidence
    st.markdown(
        f'<div class="progress-bar-custom"><div class="progress-fill" style="width:{conf*100:.0f}%"></div></div>',
        unsafe_allow_html=True,
    )

    # Alternatives
    if result.alternatives:
        st.markdown('<div class="section-header" style="margin-top:1rem;">🔀 Alternative Captions</div>', unsafe_allow_html=True)
        for i, alt in enumerate(result.alternatives):
            st.markdown(
                f'<div class="alt-caption"><strong style="color:#7c3aed">#{i+2}</strong> {alt}</div>',
                unsafe_allow_html=True,
            )

def render_image_stats(img: Image.Image):
    stats = captioner.analyze_image(img)
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label in [
        (c1, "📐", f"{stats['width']}×{stats['height']}", "Resolution"),
        (c2, "☀️", f"{stats['brightness']:.0f}/255", "Brightness"),
        (c3, "🎨", stats['dominant_channel'], "Dominant Color"),
        (c4, "📊", f"{stats['contrast']:.0f}", "Contrast (σ)"),
    ]:
        col.markdown(
            f'<div class="img-stat"><div style="font-size:1.5rem">{icon}</div>'
            f'<div style="font-weight:700;color:#a78bfa">{val}</div>'
            f'<div style="color:#94a3b8;font-size:0.75rem">{label}</div></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown('<div class="section-header">🧠 How CaptionAI Works</div>', unsafe_allow_html=True)
        steps = [
            ("1. Image Encoding", "ViT (Vision Transformer) splits image into 16×16 patches and creates rich feature embeddings"),
            ("2. Cross-Modal Attention", "BLIP's encoder attends to visual tokens, learning the relationship between regions and language"),
            ("3. Language Decoding", "A transformer decoder generates caption tokens one-by-one using beam search"),
            ("4. Beam Search", "Explores multiple caption hypotheses simultaneously, keeping the most probable sequences"),
        ]
        for title, desc in steps:
            st.markdown(
                f'<div class="sample-card"><strong style="color:#a78bfa">{title}</strong><br>'
                f'<span style="color:#94a3b8;font-size:0.85rem">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    with col_r:
        st.markdown('<div class="section-header">📊 Model Architecture</div>', unsafe_allow_html=True)
        import plotly.graph_objects as go

        components = ["Image Input", "ViT Encoder", "Cross-Attention", "Language Decoder", "Caption Output"]
        x_pos = [0, 1, 2, 3, 4]
        colors = ["#0891b2", "#7c3aed", "#db2777", "#7c3aed", "#0891b2"]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_pos, y=[0]*5,
            mode="markers+text",
            marker=dict(size=40, color=colors, symbol="square",
                        line=dict(color="white", width=2)),
            text=["🖼️", "⚡", "🔗", "📝", "💬"],
            textfont=dict(size=20),
            textposition="middle center",
        ))
        # Arrows
        for i in range(4):
            fig.add_annotation(
                x=i+0.55, y=0, ax=i+0.45, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=3, arrowsize=1.5, arrowcolor="rgba(167,139,250,0.7)", arrowwidth=2,
            )
        for i, (comp, xp) in enumerate(zip(components, x_pos)):
            fig.add_annotation(x=xp, y=-0.35, text=comp, showarrow=False,
                               font=dict(color="#94a3b8", size=9), align="center")
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(visible=False, range=[-0.5, 4.5]),
            yaxis=dict(visible=False, range=[-0.7, 0.5]),
            height=200, margin=dict(l=10, r=10, t=10, b=50),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">🏆 Model Comparison</div>', unsafe_allow_html=True)
        model_data = {
            "Model":     ["BLIP Large", "BLIP Base", "ViT-GPT2"],
            "Quality":   [95, 85, 70],
            "Speed":     [60, 80, 95],
            "Size (MB)": [990, 446, 330],
        }
        import pandas as pd
        df = pd.DataFrame(model_data)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Quality", x=df["Model"], y=df["Quality"],
                              marker_color="#7c3aed"))
        fig2.add_trace(go.Bar(name="Speed", x=df["Model"], y=df["Speed"],
                              marker_color="#0891b2"))
        fig2.update_layout(
            barmode="group", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(124,58,237,0.2)"),
            yaxis=dict(gridcolor="rgba(124,58,237,0.2)", range=[0, 110]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=250, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CAPTION IMAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📸 Caption Image":
    st.markdown('<div class="section-header">📸 Upload & Caption</div>', unsafe_allow_html=True)
    
    # Model loading
    if not captioner.is_ready or captioner.loaded_model_name != model_choice:
        if not captioner.available:
            st.error(
                "❌ **transformers** library not installed.\n\n"
                "Run: `pip install transformers torch torchvision`"
            )
            st.stop()
        
        with st.status(f"🔄 Loading {model_choice}... (first run downloads model ~450MB)", expanded=True) as status_box:
            st.write("Downloading model weights from HuggingFace Hub...")
            try:
                captioner.load_model(model_choice)
                status_box.update(label=f"✅ {model_choice} loaded!", state="complete")
            except Exception as e:
                status_box.update(label=f"❌ Failed: {e}", state="error")
                st.stop()
    
    uploaded = st.file_uploader(
        "📤 Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Supports JPG, PNG, WEBP, BMP",
    )

    if uploaded:
        img = load_image(uploaded)
        
        col_img, col_cap = st.columns([1, 1.3])
        
        with col_img:
            st.image(img, caption="Uploaded Image", use_container_width=True)
            st.markdown('<div class="section-header" style="margin-top:1rem;">📊 Image Statistics</div>', unsafe_allow_html=True)
            render_image_stats(img)
        
        with col_cap:
            generate_btn = st.button("🚀 Generate Caption", use_container_width=True)
            
            if generate_btn or "last_result" in st.session_state:
                if generate_btn:
                    with st.spinner("🧠 AI is analyzing the image..."):
                        try:
                            result = captioner.caption(
                                img,
                                conditional_text=conditional,
                                num_captions=num_captions,
                                max_length=max_length,
                                num_beams=num_beams,
                            )
                            st.session_state.last_result = result
                        except Exception as e:
                            st.error(f"❌ Caption generation failed: {e}")
                            st.stop()
                
                result = st.session_state.get("last_result")
                if result:
                    st.markdown('<div class="section-header">✨ Generated Caption</div>', unsafe_allow_html=True)
                    render_caption_result(result)
                    
                    # Copy to clipboard area
                    st.text_area("📋 Copy Caption", value=result.caption, height=80, label_visibility="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BATCH CAPTIONING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Batch Captioning":
    st.markdown('<div class="section-header">🔬 Batch Image Captioning</div>', unsafe_allow_html=True)
    st.markdown("Upload multiple images and caption them all at once.")
    
    if not captioner.is_ready:
        st.warning("⚠️ Please load a model first by going to **Caption Image** page.")
        st.stop()
    
    uploads = st.file_uploader(
        "📤 Upload multiple images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    
    if uploads:
        run_batch = st.button(f"🚀 Caption All ({len(uploads)} images)", use_container_width=True)
        
        if run_batch:
            results = []
            prog = st.progress(0, text="Processing images...")
            
            for i, uploaded_file in enumerate(uploads):
                img = load_image(uploaded_file)
                try:
                    result = captioner.caption(img, num_captions=1, max_length=max_length, num_beams=3)
                    results.append((uploaded_file.name, img, result))
                except Exception as e:
                    results.append((uploaded_file.name, img, None))
                prog.progress((i + 1) / len(uploads), text=f"Captioning {uploaded_file.name}...")
            
            prog.empty()
            
            for name, img, result in results:
                with st.expander(f"🖼️ {name}", expanded=True):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.image(img, use_container_width=True)
                    with c2:
                        if result:
                            render_caption_result(result)
                        else:
                            st.error("Failed to generate caption.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Info":
    st.markdown('<div class="section-header">📊 BLIP Model Architecture</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
### Salesforce BLIP

**BLIP** (Bootstrapped Language-Image Pre-training) is a state-of-the-art vision-language model.

#### Architecture
- **Visual Encoder**: ViT-L/16 (Vision Transformer)  
  - Splits image into 16×16 pixel patches
  - 24 transformer layers, 16 attention heads
  - 307M parameters
  
- **Language Model**: BERT-based decoder
  - 12 transformer layers
  - 768 hidden dimensions
  - Cross-attention to visual features

#### Training
- Pre-trained on 129M noisy image-text pairs
- Bootstrapped with a captioner + filter pipeline
- Fine-tuned on COCO Captions (113K images)
- Achieves **CIDEr 136.7** on COCO (SOTA)

#### Beam Search
- Explores `num_beams` parallel hypotheses
- Prunes low-probability paths
- Returns top-k caption candidates
        """)
    
    with col2:
        import plotly.graph_objects as go
        
        # BLIP architecture diagram
        layers = [
            "Image (H×W×3)", "Patch Embedding", "ViT Encoder (24L)", 
            "Cross-Attention", "Text Decoder (12L)", "Caption Tokens"
        ]
        fig = go.Figure(go.Funnel(
            y=layers,
            x=[100, 90, 80, 70, 60, 50],
            textinfo="label",
            marker=dict(color=["#0891b2","#1e40af","#7c3aed","#db2777","#7c3aed","#0891b2"]),
            connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=2)),
        ))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
#### 📈 BLIP vs Other Models (COCO CIDEr)

| Model | CIDEr | BLEU-4 |
|-------|-------|--------|
| **BLIP-Large** | **136.7** | **38.6** |
| BLIP-Base | 133.1 | 36.9 |
| ViT-GPT2 | 91.4 | 26.2 |
| ClipCap | 113.1 | 33.5 |
        """)
