"""
Task 4 — Recommendation System UI (Streamlit)
CODSOFT AI Internship | Author: SRIKRISH-S
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from recommender import MovieRecommender

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch AI — Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
    color: #e2e8f0;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(99, 102, 241, 0.4);
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin: 0;
    text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}

.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
    margin-top: 0.5rem;
}

/* Cards */
.movie-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.movie-card:hover {
    border-color: rgba(99, 102, 241, 0.6);
    background: rgba(99, 102, 241, 0.1);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.2);
}

.movie-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
}

.genre-tag {
    display: inline-block;
    background: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.7rem;
    margin: 2px;
    font-weight: 500;
}

.rating-badge {
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    color: white;
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 0.8rem;
    font-weight: 700;
}

.sim-badge {
    background: linear-gradient(135deg, #10b981, #06b6d4);
    color: white;
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #a5b4fc;
    border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Metric cards */
.metric-card {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 15, 26, 0.95) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

/* Inputs */
.stSelectbox, .stTextInput, .stSlider {
    color: #e2e8f0 !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_recommender():
    return MovieRecommender()

rec = load_recommender()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CineMatch AI")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Content-Based", "👥 Collaborative", "🔀 Hybrid", "📊 Analytics"],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    st.markdown("### 👤 User Profile")
    user_id = st.selectbox("Select User", options=list(range(1, 11)), format_func=lambda x: f"User {x}")
    
    st.markdown("---")
    st.markdown("### 🎯 Settings")
    n_recs = st.slider("Number of Recommendations", 5, 20, 10)
    
    st.markdown("---")
    st.markdown(
        '<div style="color: #6366f1; font-size: 0.8rem; text-align: center;">'
        "CODSOFT AI Internship<br>Task 4 — Recommendation System<br>"
        '<a href="https://github.com/SRIKRISH-S/CODSOFT" style="color: #a5b4fc;">GitHub: SRIKRISH-S</a>'
        "</div>",
        unsafe_allow_html=True,
    )

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🎬 CineMatch AI</div>
    <div class="hero-subtitle">Intelligent Movie Recommendations · Content-Based · Collaborative Filtering · Hybrid AI</div>
</div>
""", unsafe_allow_html=True)

# ── Helper: Render Movie Cards ─────────────────────────────────────────────────
def render_movie_cards(df: pd.DataFrame, badge_col: str = None, badge_label: str = "Score"):
    if df.empty:
        st.warning("No recommendations found. Try a different input.")
        return
    
    cols = st.columns(2)
    for i, row in df.iterrows():
        col = cols[i % 2]
        with col:
            genres_html = "".join(
                f'<span class="genre-tag">{g}</span>'
                for g in row["genres"].split()[:4]
            )
            badge_html = ""
            if badge_col and badge_col in row:
                val = row[badge_col]
                badge_html = f'<span class="sim-badge">{badge_label}: {val}</span>'
            
            stars = "⭐" * int(round(row.get("rating", 0)))
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">{'🏆 ' if i == 0 else ''}{i+1}. {row['title']}</div>
                    <div style="margin: 6px 0;">{genres_html}</div>
                    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:8px;">
                        <span class="rating-badge">⭐ {row.get('rating', 'N/A')}</span>
                        {badge_html}
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
    
    with col1:
        st.markdown(
            '<div class="metric-card"><div style="font-size:2rem">🎬</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:#a5b4fc">100</div>'
            '<div style="color:#94a3b8;font-size:0.85rem">Movies in Database</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div style="font-size:2rem">👥</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:#a5b4fc">10</div>'
            '<div style="color:#94a3b8;font-size:0.85rem">User Profiles</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div style="font-size:2rem">🤖</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:#a5b4fc">3</div>'
            '<div style="color:#94a3b8;font-size:0.85rem">AI Algorithms</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div style="font-size:2rem">🎯</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:#a5b4fc">TF-IDF+SVD</div>'
            '<div style="color:#94a3b8;font-size:0.85rem">Core Technology</div></div>',
            unsafe_allow_html=True,
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.markdown('<div class="section-header">🔥 Top Rated Movies</div>', unsafe_allow_html=True)
        genre_filter = st.selectbox("Filter by genre", ["All"] + rec.get_genres())
        top_movies = rec.get_top_rated(genre_filter, 10)
        render_movie_cards(top_movies)
    
    with col_r:
        st.markdown('<div class="section-header">📊 Genre Distribution</div>', unsafe_allow_html=True)
        from collections import Counter
        all_genres_flat = []
        for g in rec.movies_df["genres"]:
            all_genres_flat.extend(g.split())
        genre_counts = Counter(all_genres_flat)
        top_genres = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:12])
        
        fig = px.bar(
            x=list(top_genres.values()),
            y=list(top_genres.keys()),
            orientation="h",
            color=list(top_genres.values()),
            color_continuous_scale=["#6366f1", "#8b5cf6", "#ec4899"],
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor="rgba(99,102,241,0.2)"),
            yaxis=dict(gridcolor="rgba(99,102,241,0.2)"),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="section-header">📋 Your Watch History</div>', unsafe_allow_html=True)
        history = rec.get_user_history(user_id)
        if not history.empty:
            for _, row in history.iterrows():
                stars = "⭐" * row["user_rating"]
                st.markdown(
                    f'<div class="movie-card"><div class="movie-title">{row["title"]}</div>'
                    f'<div style="color:#f59e0b">{stars}</div></div>',
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CONTENT-BASED
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Content-Based":
    st.markdown('<div class="section-header">🔍 Content-Based Filtering</div>', unsafe_allow_html=True)
    st.markdown(
        "> Uses **TF-IDF genre vectors** and **cosine similarity** to find movies similar to your chosen title."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔎 Search for a movie", placeholder="e.g. Inception, Matrix, Toy Story...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("Get Recommendations", use_container_width=True)
    
    if search_query:
        results = rec.search_movies(search_query)
        if not results.empty:
            st.markdown("**Found movies — select one:**")
            selected = st.selectbox("", results["title"].tolist(), label_visibility="collapsed")
            
            if search_btn or selected:
                with st.spinner("Finding similar movies..."):
                    recs = rec.content_recommend(selected, n_recs)
                
                st.markdown(f'<div class="section-header">🎯 Movies similar to "{selected}"</div>', unsafe_allow_html=True)
                
                # similarity chart
                if not recs.empty:
                    fig = go.Figure(go.Bar(
                        x=recs["similarity"],
                        y=[t[:30] + "..." if len(t) > 30 else t for t in recs["title"]],
                        orientation="h",
                        marker=dict(
                            color=recs["similarity"],
                            colorscale=[[0, "#6366f1"], [0.5, "#8b5cf6"], [1, "#ec4899"]],
                        ),
                    ))
                    fig.update_layout(
                        title="Similarity Scores",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e2e8f0"),
                        margin=dict(l=0, r=0, t=40, b=0),
                        xaxis=dict(gridcolor="rgba(99,102,241,0.2)", title="Cosine Similarity"),
                        yaxis=dict(gridcolor="rgba(99,102,241,0.2)"),
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    render_movie_cards(recs, badge_col="similarity", badge_label="Similarity")
        else:
            st.warning("No movies found. Try a different search term.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: COLLABORATIVE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "👥 Collaborative":
    st.markdown('<div class="section-header">👥 Collaborative Filtering</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📉 SVD / Matrix Factorization", "🤝 User Similarity"])
    
    with tab1:
        st.markdown(
            "> Uses **Singular Value Decomposition (SVD)** to predict what movies you'll love "
            "based on the ratings of similar users."
        )
        with st.spinner("Computing SVD recommendations..."):
            svd_recs = rec.collab_recommend(user_id, n_recs)
        
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown(f"**Recommendations for User {user_id}:**")
            render_movie_cards(svd_recs, badge_col="predicted_score", badge_label="Predicted")
        with col_r:
            if not svd_recs.empty:
                fig = px.scatter(
                    svd_recs,
                    x="predicted_score",
                    y="rating",
                    text=[t[:20] for t in svd_recs["title"]],
                    color="predicted_score",
                    color_continuous_scale=["#6366f1", "#ec4899"],
                    size_max=20,
                )
                fig.update_traces(textposition="top center", textfont_size=8)
                fig.update_layout(
                    title="Predicted vs. Actual Ratings",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="rgba(99,102,241,0.2)", title="Predicted Score"),
                    yaxis=dict(gridcolor="rgba(99,102,241,0.2)", title="Avg. Rating"),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown(
            "> Finds users with **similar taste profiles** using cosine similarity, "
            "then recommends what they loved."
        )
        with st.spinner("Finding similar users..."):
            user_recs = rec.user_similarity_recommend(user_id, n_recs)
        
        render_movie_cards(user_recs)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HYBRID
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔀 Hybrid":
    st.markdown('<div class="section-header">🔀 Hybrid Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        "> Combines **content-based** (genre similarity) and **collaborative** (user preferences) "
        "for the most personalized recommendations."
    )
    
    search_query = st.text_input("🔎 Enter a movie you like", placeholder="e.g. The Dark Knight, Inception...")
    
    if search_query:
        results = rec.search_movies(search_query)
        if not results.empty:
            selected = st.selectbox("Select movie", results["title"].tolist(), label_visibility="collapsed")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Content Weight:** Genre similarity score")
            with col2:
                st.markdown(f"**Collaborative Boost:** Based on User {user_id}'s taste")
            
            with st.spinner("Running hybrid AI..."):
                hybrid_recs = rec.hybrid_recommend(selected, user_id, n_recs)
            
            render_movie_cards(
                hybrid_recs,
                badge_col="hybrid_score" if "hybrid_score" in (hybrid_recs.columns if not hybrid_recs.empty else []) else "similarity",
                badge_label="Hybrid Score",
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Analytics":
    st.markdown('<div class="section-header">📊 Dataset Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        ratings = rec.movies_df["rating"].values
        fig = px.histogram(
            x=ratings,
            nbins=20,
            labels={"x": "Rating"},
            color_discrete_sequence=["#6366f1"],
        )
        fig.update_layout(
            title="Rating Distribution",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(99,102,241,0.2)"),
            yaxis=dict(gridcolor="rgba(99,102,241,0.2)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top rated movies radar-style bar
        top10 = rec.movies_df.nlargest(10, "rating")[["title", "rating"]]
        fig = px.bar(
            top10,
            x="rating",
            y=[t[:25] + "..." if len(t) > 25 else t for t in top10["title"]],
            orientation="h",
            color="rating",
            color_continuous_scale=["#6366f1", "#8b5cf6", "#ec4899"],
        )
        fig.update_layout(
            title="Top 10 Rated Movies",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(99,102,241,0.2)"),
            yaxis=dict(gridcolor="rgba(99,102,241,0.2)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # User ratings heatmap
    st.markdown('<div class="section-header">👥 User-Movie Ratings Heatmap</div>', unsafe_allow_html=True)
    movie_ids = list(rec.movies_df.index[:30])
    matrix_data = []
    for uid in range(1, 11):
        row = []
        rated = dict(USER_RATINGS.get(uid, []))
        for mid in movie_ids:
            row.append(rated.get(mid, 0))
        matrix_data.append(row)
    
    movie_titles_short = [rec.movies_df.loc[m, "title"][:20] + "..." 
                         if len(rec.movies_df.loc[m, "title"]) > 20 
                         else rec.movies_df.loc[m, "title"]
                         for m in movie_ids]
    
    fig = go.Figure(go.Heatmap(
        z=matrix_data,
        x=movie_titles_short,
        y=[f"User {i}" for i in range(1, 11)],
        colorscale=[[0, "#0a0a0f"], [0.5, "#6366f1"], [1, "#ec4899"]],
        text=matrix_data,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=9),
        height=350,
        xaxis=dict(tickangle=-45),
        margin=dict(l=80, r=0, t=10, b=120),
    )
    st.plotly_chart(fig, use_container_width=True)
