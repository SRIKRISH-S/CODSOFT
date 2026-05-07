# 👁️ FaceVision AI — Face Detection & Recognition

> **CODSOFT AI Internship | Task 5**  
> Author: [SRIKRISH-S](https://github.com/SRIKRISH-S)

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)](https://opencv.org)
[![dlib](https://img.shields.io/badge/dlib-face--recognition-purple)](https://github.com/ageitgey/face_recognition)

---

## 📋 Overview

**FaceVision AI** is a complete face detection and recognition pipeline built with OpenCV and dlib. It supports both image-based and real-time webcam detection.

## 🧠 AI Techniques Used

| Technique | Library | Description |
|-----------|---------|-------------|
| Haar Cascade Detection | OpenCV | Fast frontal face detection using Haar features |
| DNN Face Detection | OpenCV DNN | Res10 SSD deep neural network detector |
| Face Recognition | dlib (face_recognition) | 128-dimensional ResNet face encodings |
| Eye & Smile Detection | OpenCV | Multi-scale cascade classifiers |

## 🚀 Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/SRIKRISH-S/CODSOFT.git
cd CODSOFT/task5_face_detection

# 2. Install core dependencies
pip install streamlit opencv-python numpy Pillow plotly

# 3. (Optional) Install face recognition — requires cmake
pip install cmake
pip install dlib
pip install face-recognition

# 4. Launch the app
streamlit run app.py
```

> **Windows Note**: If dlib install fails, download a prebuilt wheel from:
> https://github.com/z-mahmud22/Dlib_Windows_Python3.x

## 📸 Features

- 🖼️ **Image Detection** — Upload any photo for instant face detection with bounding boxes
- 📹 **Live Webcam** — Real-time detection at ~30 FPS with face recognition
- 👤 **Register Faces** — Add new people to the recognition database
- 📂 **Face Database** — Manage registered face encodings
- 📊 **Algorithm Comparison** — Radar chart comparing Haar vs DNN approaches
- ✂️ **Face Crops** — Extract individual face regions from images

## 🗂️ Project Structure

```
task5_face_detection/
├── app.py              # Streamlit UI application
├── face_engine.py      # Detection & recognition engine
├── known_faces/        # Face encoding database (auto-created)
│   └── encodings.json  # Registered face encodings
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🔬 Technical Details

- **Haar Cascade**: Viola-Jones algorithm with scale pyramid scanning
- **DNN Detector**: Caffe-based Res10 SSD optimized for face detection
- **Face Encodings**: 128-D vectors from dlib's ResNet-34 trained on LFW dataset
- **Recognition**: Euclidean distance threshold (default 0.5 tolerance)
- **Eye Detection**: Haar cascade on face ROI for landmark detection
- **Smile Detection**: Multi-scale cascade with higher min-neighbors

## 🛠️ Usage

1. **Image Detection**: Go to Image Detection → Upload photo → View annotated results
2. **Register a Face**: Go to Register Face → Enter name → Upload clear photo
3. **Live Recognition**: Go to Webcam Live → Toggle camera → Stand in front of webcam
4. **Manage Database**: Go to Face Database → View / delete registered faces
