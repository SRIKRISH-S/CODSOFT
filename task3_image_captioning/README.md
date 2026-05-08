# 🖼️ CaptionAI — Image Captioning with BLIP

> **CODSOFT AI Internship | Task 3**  
> Author: [SRIKRISH-S](https://github.com/SRIKRISH-S)

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/Salesforce/blip-image-captioning-large)
[![Model](https://img.shields.io/badge/Model-BLIP--Large-purple)](https://arxiv.org/abs/2201.12086)

---

## 📋 Overview

**CaptionAI** generates natural language descriptions of images using **BLIP** (Bootstrapped Language-Image Pre-training), a state-of-the-art vision-language model from Salesforce Research.

## 🧠 Models Supported

| Model | Size | CIDEr | Speed | Best For |
|-------|------|-------|-------|----------|
| BLIP Large | ~990 MB | 136.7 | Slow | Maximum quality |
| BLIP Base | ~446 MB | 133.1 | Medium | Balanced |
| ViT-GPT2 | ~330 MB | 91.4 | Fast | Quick experiments |

## 🏗️ Architecture

```
Image → ViT Patch Encoder → Cross-Attention → BERT Decoder → Caption
         (16×16 patches)     (visual tokens)   (language model)
```

- **Visual Encoder**: ViT-L/16 — 24 transformer layers, 307M parameters
- **Language Decoder**: BERT-based with cross-attention to image features
- **Beam Search**: Explores multiple caption hypotheses (configurable)
- **Pre-training**: 129M image-text pairs + COCO fine-tuning

## 🚀 Setup & Run

```bash
cd CODSOFT/task3_image_captioning
pip install -r requirements.txt
streamlit run app.py
```

> **First run**: The model (~450MB) is downloaded automatically from HuggingFace Hub.

## 📸 Features

- 📸 **Single Image Captioning** — Upload any photo, get instant AI captions
- 🔀 **Multiple Captions** — Beam search generates diverse alternatives
- 💬 **Conditional Captioning** — Guide captions with a text prompt
- 🔬 **Batch Mode** — Caption multiple images at once
- 📊 **Image Analysis** — Brightness, contrast, dominant color, resolution
- 🏗️ **Architecture Visualizer** — Interactive BLIP pipeline diagram
- 📈 **Model Benchmarks** — COCO CIDEr score comparisons

## 🗂️ Project Structure

```
task3_image_captioning/
├── app.py           # Streamlit UI (4 pages)
├── captioner.py     # BLIP captioning engine
├── requirements.txt
└── README.md
```

## 🔬 Technical Details

- **BLIP Paper**: [Salesforce Research, 2022](https://arxiv.org/abs/2201.12086)
- **COCO CIDEr**: 136.7 (state-of-the-art as of 2022)
- **Beam Search**: `num_beams=5` default, configurable up to 10
- **Max Length**: 50 tokens default, configurable up to 100
- **Conditional**: Provide text prefix to guide caption style
