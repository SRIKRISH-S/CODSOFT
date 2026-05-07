# 🎬 CineMatch AI — Movie Recommendation System

> **CODSOFT AI Internship | Task 4**  
> Author: [SRIKRISH-S](https://github.com/SRIKRISH-S)

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Overview

**CineMatch AI** is an intelligent movie recommendation system that uses three complementary AI techniques to deliver highly personalized recommendations.

## 🧠 Algorithms Implemented

| Algorithm | Technique | Description |
|-----------|-----------|-------------|
| Content-Based | TF-IDF + Cosine Similarity | Recommends movies based on genre similarity |
| Collaborative (SVD) | Matrix Factorization (TruncatedSVD) | Predicts ratings from the user-item matrix |
| Collaborative (User-KNN) | Cosine Similarity | Finds similar users and recommends their favorites |
| Hybrid | Weighted combination | Blends content + collaborative signals |

## 🚀 Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/SRIKRISH-S/CODSOFT.git
cd CODSOFT/task4_recommendation_system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

## 📸 Features

- 🔍 **Content-Based Filtering** — Search any movie, get similar ones with similarity scores and interactive charts
- 👥 **Collaborative Filtering** — SVD matrix factorization and user-similarity based personalized recommendations
- 🔀 **Hybrid Mode** — Combines both approaches for maximum accuracy
- 📊 **Analytics Dashboard** — Genre distribution, rating histograms, user-movie heatmaps
- 🎨 **Premium Dark UI** — Glassmorphism design with smooth animations

## 🗂️ Project Structure

```
task4_recommendation_system/
├── app.py              # Streamlit UI application
├── recommender.py      # Core recommendation engine
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🔬 Technical Details

- **Dataset**: 100 movies (MovieLens-style) with genre tags and ratings
- **Content Model**: TF-IDF vectorization on genre strings → cosine similarity matrix
- **Collaborative Model**: User-item rating matrix → TruncatedSVD (10 latent factors)
- **User Similarity**: Cosine similarity between user rating vectors
- **Hybrid**: Content similarity (50%) + collaborative boost (50%) re-ranking
