"""
Task 4 — Recommendation System  |  CODSOFT AI Internship
Author: SRIKRISH-S | github.com/SRIKRISH-S/CODSOFT
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from recommender import MovieRecommender

st.set_page_config(page_title="CineMatch · Recommendations", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');

*, html, body { font-family: 'Manrope', sans-serif; }
.stApp { background: #0f0f11; color: #e4e4e7; }

[data-testid="stSidebar"] { background: #0a0a0c !important; border-right: 1px solid #1c1c22; }
[data-testid="stSidebar"] * { color: #a1a1aa !important; }
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong { color: #e4e4e7 !important; }

#MainMenu, footer, header { visibility: hidden; }

.page-title {
    font-size: 2.4rem; font-weight: 800; color: #fff;
    letter-spacing: -0.5px;
}
.page-sub { color: #52525b; font-size: 0.85rem; margin-top: 0.1rem; }

/* Movie cards */
.movie-card {
    background: #18181b; border: 1px solid #27272a;
    border-radius: 14px; padding: 1rem 1.1rem;
    margin: 0.45rem 0; transition: all 0.2s;
}
.movie-card:hover { border-color: #7c3aed; background: #1c1c23;
    transform: translateX(3px); box-shadow: -3px 0 0 #7c3aed; }
.movie-title { font-weight: 700; color: #e4e4e7; font-size: 0.92rem; margin-bottom: 4px; }

.tag {
    display: inline-block; border-radius: 6px;
    padding: 2px 9px; font-size: 0.68rem; font-weight: 600;
    margin: 2px; letter-spacing: 0.3px;
}
.tag-genre { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.2); }
.tag-score { background: rgba(234,179,8,0.12); color: #fbbf24; border: 1px solid rgba(234,179,8,0.2); }
.tag-sim   { background: rgba(16,185,129,0.12); color: #34d399; border: 1px solid rgba(16,185,129,0.2); }
.tag-rank  { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

.section-title {
    font-size: 1rem; font-weight: 800; color: #e4e4e7;
    text-transform: uppercase; letter-spacing: 1.5px;
    border-bottom: 2px solid #27272a; padding-bottom: 0.5rem; margin-bottom: 0.8rem;
}

.kpi-box {
    background: #18181b; border: 1px solid #27272a;
    border-radius: 12px; padding: 1rem; text-align: center;
}
.kpi-val { font-size: 1.6rem; font-weight: 800; color: #7c3aed; }
.kpi-lbl { font-size: 0.7rem; color: #52525b; text-transform: uppercase; letter-spacing: 1px; }

.stButton > button {
    background: #7c3aed !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; transition: all 0.2s !important;
}
.stButton > button:hover { background: #6d28d9 !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important; transform: translateY(-1px) !important; }

.hist-row {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 10px; border-radius: 8px; background: #18181b;
    border: 1px solid #27272a; margin: 4px 0; font-size: 0.82rem;
}
.star { color: #fbbf24; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rec():
    return MovieRecommender()

rec = load_rec()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 CineMatch AI")
    st.caption("TF-IDF · SVD · Collaborative Filtering")
    st.divider()

    page = st.radio("", ["🏠 Discover", "🔍 Content-Based", "👥 Collaborative", "🔀 Hybrid", "📊 Analytics"],
                    label_visibility="collapsed")
    st.divider()

    st.markdown("**Your Profile**")
    user_id = st.selectbox("", list(range(1, 11)),
                            format_func=lambda x: f"User {x}", label_visibility="collapsed")
    st.divider()

    n_recs = st.slider("Results", 5, 20, 10)
    genre_opts = ["All"] + rec.get_genres()

    st.divider()
    st.markdown('<div style="text-align:center;color:#3f3f46;font-size:0.75rem;">CODSOFT AI · Task 4<br>'
                '<a href="https://github.com/SRIKRISH-S/CODSOFT" style="color:#7c3aed;">github.com/SRIKRISH-S</a></div>',
                unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">🎬 CineMatch</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">CONTENT-BASED · COLLABORATIVE FILTERING · HYBRID AI RECOMMENDATIONS</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Helpers ───────────────────────────────────────────────────────────────────
def render_cards(df: pd.DataFrame, badge_col=None, badge_cls="tag-sim", badge_label="Match"):
    if df.empty:
        st.info("No results — try a different input.")
        return
    cols = st.columns(2)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 2]:
            genres_html = "".join(f'<span class="tag tag-genre">{g}</span>'
                                  for g in str(row.get("genres","")).split()[:4])
            rating = row.get("rating", 0)
            badge_html = ""
            if badge_col and badge_col in row.index and pd.notna(row[badge_col]):
                badge_html = f'<span class="tag {badge_cls}">{badge_label}: {row[badge_col]}</span>'
            rank_icon = "🥇" if i == 0 else f"#{i+1}"
            st.markdown(
                f'<div class="movie-card">'
                f'<div class="movie-title">{rank_icon} {row.get("title","?")}</div>'
                f'<div>{genres_html}</div>'
                f'<div style="margin-top:6px;">'
                f'<span class="tag tag-score">⭐ {rating}</span>{badge_html}</div>'
                f'</div>',
                unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    plot_bgcolor="#18181b", paper_bgcolor="#0f0f11",
    font=dict(color="#a1a1aa", family="Manrope"),
    xaxis=dict(gridcolor="#27272a"), yaxis=dict(gridcolor="#27272a"),
    margin=dict(l=0, r=0, t=40, b=0), height=320,
)

# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Discover":
    kpi_cols = st.columns(4)
    for col, val, lbl in zip(kpi_cols,
        [len(rec.movies_df), "10", "3", "TF-IDF + SVD"],
        ["Movies", "User Profiles", "AI Algorithms", "Core Model"]):
        col.markdown(f'<div class="kpi-box"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-title">🔥 Top Rated</div>', unsafe_allow_html=True)
        gf = st.selectbox("Filter genre", genre_opts, key="gf_home")
        render_cards(rec.get_top_rated(gf, 8))

    with col_r:
        st.markdown('<div class="section-title">Genre Distribution</div>', unsafe_allow_html=True)
        from collections import Counter
        flat = []
        for g in rec.movies_df["genres"]: flat.extend(g.split())
        cnt = Counter(flat)
        top12 = dict(sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:12])
        fig = go.Figure(go.Bar(x=list(top12.values()), y=list(top12.keys()), orientation="h",
                               marker_color="#7c3aed"))
        fig.update_layout(**PLOT_LAYOUT, title="Top Genres")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Watch History — User {}</div>'.format(user_id), unsafe_allow_html=True)
        hist = rec.get_user_history(user_id)
        if not hist.empty:
            for _, row in hist.head(5).iterrows():
                stars = "★" * int(row.get("user_rating", 0))
                st.markdown(
                    f'<div class="hist-row"><span class="star">{stars}</span>'
                    f'<span style="color:#a1a1aa;font-size:0.82rem">{row["title"]}</span></div>',
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Content-Based":
    st.markdown('<div class="section-title">🔍 Content-Based Filtering (TF-IDF + Cosine Similarity)</div>',
                unsafe_allow_html=True)
    q = st.text_input("Search a movie you like", placeholder="e.g. Inception, The Matrix…")
    if q:
        hits = rec.search_movies(q)
        if not hits.empty:
            sel = st.selectbox("Select", hits["title"].tolist(), label_visibility="collapsed")
            with st.spinner("Finding similar movies…"):
                recs = rec.content_recommend(sel, n_recs)
            st.markdown(f'<div class="section-title">Movies similar to "{sel}"</div>', unsafe_allow_html=True)
            if not recs.empty:
                fig = go.Figure(go.Bar(
                    x=recs["similarity"], y=[t[:28]+"…" if len(t)>28 else t for t in recs["title"]],
                    orientation="h", marker_color="#7c3aed",
                    text=[f"{v:.2f}" for v in recs["similarity"]], textposition="inside",
                ))
                fig.update_layout(**PLOT_LAYOUT, title="Similarity Scores (Cosine)")
                st.plotly_chart(fig, use_container_width=True)
            render_cards(recs, "similarity", "tag-sim", "Sim")
        else:
            st.warning("No movies found — try another title.")

# ─────────────────────────────────────────────────────────────────────────────
elif page == "👥 Collaborative":
    st.markdown('<div class="section-title">👥 Collaborative Filtering</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📉 SVD Matrix Factorization", "🤝 User Similarity"])
    with tab1:
        with st.spinner("Running SVD…"):
            svd_r = rec.collab_recommend(user_id, n_recs)
        c1, c2 = st.columns([1,1])
        with c1:
            render_cards(svd_r, "predicted_score", "tag-sim", "Pred")
        with c2:
            if not svd_r.empty:
                fig = px.scatter(svd_r, x="predicted_score", y="rating",
                                 hover_name="title", color="predicted_score",
                                 color_continuous_scale=["#27272a","#7c3aed"],
                                 labels={"predicted_score":"Predicted","rating":"Avg Rating"})
                fig.update_layout(**PLOT_LAYOUT, title="Predicted vs Actual Ratings",
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
    with tab2:
        with st.spinner("Finding similar users…"):
            user_r = rec.user_similarity_recommend(user_id, n_recs)
        render_cards(user_r)

# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔀 Hybrid":
    st.markdown('<div class="section-title">🔀 Hybrid AI Recommendations</div>', unsafe_allow_html=True)
    st.caption("Blends content-based (50%) + collaborative boost (50%) for maximum personalisation.")
    q = st.text_input("Movie you like", placeholder="e.g. Parasite, Interstellar…")
    if q:
        hits = rec.search_movies(q)
        if not hits.empty:
            sel = st.selectbox("Select", hits["title"].tolist(), label_visibility="collapsed")
            with st.spinner("Hybrid AI running…"):
                h_recs = rec.hybrid_recommend(sel, user_id, n_recs)
            badge_col = "hybrid_score" if "hybrid_score" in (h_recs.columns if not h_recs.empty else []) else "similarity"
            render_cards(h_recs, badge_col, "tag-sim", "Hybrid")

# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Analytics":
    st.markdown('<div class="section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(x=rec.movies_df["rating"], nbins=20, color_discrete_sequence=["#7c3aed"],
                           labels={"x":"Rating"}, title="Rating Distribution")
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top10 = rec.movies_df.nlargest(10,"rating")
        fig = px.bar(top10, x="rating", y=[t[:22]+"…" if len(t)>22 else t for t in top10["title"]],
                     orientation="h", color="rating",
                     color_continuous_scale=["#27272a","#7c3aed"],
                     title="Top 10 Rated Movies")
        fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">User × Movie Ratings Heatmap</div>', unsafe_allow_html=True)
    from recommender import USER_RATINGS
    mids = list(rec.movies_df.index[:25])
    mat  = [[dict(USER_RATINGS.get(u,[])).get(m,0) for m in mids] for u in range(1,11)]
    xlbl = [rec.movies_df.loc[m,"title"][:16]+"…" if len(rec.movies_df.loc[m,"title"])>16
            else rec.movies_df.loc[m,"title"] for m in mids]
    fig = go.Figure(go.Heatmap(z=mat, x=xlbl, y=[f"U{i}" for i in range(1,11)],
                               colorscale=[[0,"#0f0f11"],[0.5,"#7c3aed"],[1,"#fbbf24"]],
                               showscale=True))
    fig.update_layout(plot_bgcolor="#18181b", paper_bgcolor="#0f0f11",
                      font=dict(color="#a1a1aa",size=8), height=300,
                      margin=dict(l=40,r=0,t=10,b=100),
                      xaxis=dict(tickangle=-45))
    st.plotly_chart(fig, use_container_width=True)
