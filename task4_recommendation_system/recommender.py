"""
Recommendation Engine — Task 4 (CODSOFT AI Internship)
Implements content-based filtering, collaborative filtering, and hybrid recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pickle
import os

# ─── Dataset (embedded MovieLens-style sample) ───────────────────────────────

MOVIES_DATA = [
    {"movieId": 1, "title": "Toy Story (1995)", "genres": "Adventure Animation Children Comedy Fantasy", "rating": 3.9},
    {"movieId": 2, "title": "Jumanji (1995)", "genres": "Adventure Children Fantasy", "rating": 3.2},
    {"movieId": 3, "title": "Grumpier Old Men (1995)", "genres": "Comedy Romance", "rating": 3.2},
    {"movieId": 4, "title": "Waiting to Exhale (1995)", "genres": "Comedy Drama Romance", "rating": 2.9},
    {"movieId": 5, "title": "Father of the Bride Part II (1995)", "genres": "Comedy", "rating": 3.1},
    {"movieId": 6, "title": "Heat (1995)", "genres": "Action Crime Thriller", "rating": 4.0},
    {"movieId": 7, "title": "Sabrina (1995)", "genres": "Comedy Romance", "rating": 3.2},
    {"movieId": 8, "title": "Tom and Huck (1995)", "genres": "Adventure Children", "rating": 3.0},
    {"movieId": 9, "title": "Sudden Death (1995)", "genres": "Action", "rating": 3.0},
    {"movieId": 10, "title": "GoldenEye (1995)", "genres": "Action Adventure Thriller", "rating": 3.5},
    {"movieId": 11, "title": "The American President (1995)", "genres": "Comedy Drama Romance", "rating": 3.7},
    {"movieId": 12, "title": "Dracula: Dead and Loving It (1995)", "genres": "Comedy Horror", "rating": 2.6},
    {"movieId": 13, "title": "Balto (1995)", "genres": "Adventure Animation Children", "rating": 3.2},
    {"movieId": 14, "title": "Nixon (1995)", "genres": "Drama", "rating": 3.4},
    {"movieId": 15, "title": "Cutthroat Island (1995)", "genres": "Action Adventure Romance", "rating": 2.9},
    {"movieId": 16, "title": "Casino (1995)", "genres": "Crime Drama", "rating": 3.8},
    {"movieId": 17, "title": "Sense and Sensibility (1995)", "genres": "Drama Romance", "rating": 3.9},
    {"movieId": 18, "title": "Four Rooms (1995)", "genres": "Comedy Thriller", "rating": 3.1},
    {"movieId": 19, "title": "Ace Ventura: When Nature Calls (1995)", "genres": "Comedy", "rating": 3.0},
    {"movieId": 20, "title": "Money Train (1995)", "genres": "Action Comedy Crime Drama Thriller", "rating": 2.8},
    {"movieId": 21, "title": "Get Shorty (1995)", "genres": "Comedy Crime Thriller", "rating": 3.6},
    {"movieId": 22, "title": "Copycat (1995)", "genres": "Crime Drama Horror Mystery Thriller", "rating": 3.6},
    {"movieId": 23, "title": "Assassins (1995)", "genres": "Action Crime Thriller", "rating": 2.8},
    {"movieId": 24, "title": "Powder (1995)", "genres": "Drama Sci-Fi", "rating": 3.0},
    {"movieId": 25, "title": "Leaving Las Vegas (1995)", "genres": "Drama Romance", "rating": 3.7},
    {"movieId": 26, "title": "Othello (1995)", "genres": "Drama", "rating": 3.3},
    {"movieId": 27, "title": "Now and Then (1995)", "genres": "Children Drama", "rating": 2.9},
    {"movieId": 28, "title": "Persuasion (1995)", "genres": "Drama Romance", "rating": 3.5},
    {"movieId": 29, "title": "City of Lost Children (1995)", "genres": "Adventure Drama Fantasy Mystery Sci-Fi", "rating": 3.9},
    {"movieId": 30, "title": "Shanghai Triad (1995)", "genres": "Crime Drama", "rating": 3.6},
    {"movieId": 31, "title": "Dangerous Minds (1995)", "genres": "Drama", "rating": 3.0},
    {"movieId": 32, "title": "12 Monkeys (1995)", "genres": "Mystery Sci-Fi Thriller", "rating": 4.1},
    {"movieId": 33, "title": "Wings of Courage (1995)", "genres": "Adventure Romance", "rating": 3.0},
    {"movieId": 34, "title": "Babe (1995)", "genres": "Children Drama", "rating": 3.8},
    {"movieId": 35, "title": "Carrington (1995)", "genres": "Drama Romance", "rating": 3.4},
    {"movieId": 36, "title": "Dead Man (1995)", "genres": "Comedy Drama Western", "rating": 3.4},
    {"movieId": 37, "title": "Across the Sea of Time (1995)", "genres": "Children Drama", "rating": 2.9},
    {"movieId": 38, "title": "It Takes Two (1995)", "genres": "Children Comedy Romance", "rating": 2.7},
    {"movieId": 39, "title": "Clueless (1995)", "genres": "Comedy Romance", "rating": 3.4},
    {"movieId": 40, "title": "Dumb & Dumber (1994)", "genres": "Adventure Comedy", "rating": 3.0},
    {"movieId": 41, "title": "Desperado (1995)", "genres": "Action Romance Thriller", "rating": 3.4},
    {"movieId": 42, "title": "Pulp Fiction (1994)", "genres": "Comedy Crime Drama Thriller", "rating": 4.2},
    {"movieId": 43, "title": "The Shawshank Redemption (1994)", "genres": "Crime Drama", "rating": 4.5},
    {"movieId": 44, "title": "Forrest Gump (1994)", "genres": "Comedy Drama Romance War", "rating": 4.1},
    {"movieId": 45, "title": "The Lion King (1994)", "genres": "Adventure Animation Children Drama Musical", "rating": 4.0},
    {"movieId": 46, "title": "Speed (1994)", "genres": "Action Romance Thriller", "rating": 3.6},
    {"movieId": 47, "title": "True Lies (1994)", "genres": "Action Adventure Comedy Romance Thriller", "rating": 3.5},
    {"movieId": 48, "title": "Interview with the Vampire (1994)", "genres": "Drama Horror", "rating": 3.5},
    {"movieId": 49, "title": "The Mask (1994)", "genres": "Action Comedy Crime Fantasy Romance", "rating": 3.4},
    {"movieId": 50, "title": "Schindler's List (1993)", "genres": "Drama War", "rating": 4.5},
    {"movieId": 51, "title": "Dances with Wolves (1990)", "genres": "Adventure Drama Western", "rating": 4.0},
    {"movieId": 52, "title": "The Silence of the Lambs (1991)", "genres": "Crime Horror Mystery Thriller", "rating": 4.4},
    {"movieId": 53, "title": "Terminator 2 (1991)", "genres": "Action Sci-Fi Thriller", "rating": 4.1},
    {"movieId": 54, "title": "Star Wars: A New Hope (1977)", "genres": "Action Adventure Fantasy Sci-Fi", "rating": 4.4},
    {"movieId": 55, "title": "The Empire Strikes Back (1980)", "genres": "Action Adventure Sci-Fi", "rating": 4.4},
    {"movieId": 56, "title": "Return of the Jedi (1983)", "genres": "Action Adventure Sci-Fi", "rating": 4.1},
    {"movieId": 57, "title": "Raiders of the Lost Ark (1981)", "genres": "Action Adventure", "rating": 4.3},
    {"movieId": 58, "title": "Jurassic Park (1993)", "genres": "Action Adventure Sci-Fi Thriller", "rating": 3.9},
    {"movieId": 59, "title": "The Matrix (1999)", "genres": "Action Sci-Fi Thriller", "rating": 4.3},
    {"movieId": 60, "title": "Inception (2010)", "genres": "Action Adventure Sci-Fi Thriller", "rating": 4.3},
    {"movieId": 61, "title": "The Dark Knight (2008)", "genres": "Action Crime Drama Thriller", "rating": 4.5},
    {"movieId": 62, "title": "Interstellar (2014)", "genres": "Adventure Drama Sci-Fi", "rating": 4.2},
    {"movieId": 63, "title": "Avengers: Endgame (2019)", "genres": "Action Adventure Sci-Fi", "rating": 4.0},
    {"movieId": 64, "title": "Titanic (1997)", "genres": "Drama Romance", "rating": 3.9},
    {"movieId": 65, "title": "The Godfather (1972)", "genres": "Crime Drama", "rating": 4.6},
    {"movieId": 66, "title": "Goodfellas (1990)", "genres": "Crime Drama", "rating": 4.4},
    {"movieId": 67, "title": "Fight Club (1999)", "genres": "Crime Drama Mystery Thriller", "rating": 4.3},
    {"movieId": 68, "title": "Gladiator (2000)", "genres": "Action Adventure Drama", "rating": 4.0},
    {"movieId": 69, "title": "The Lord of the Rings: Fellowship (2001)", "genres": "Adventure Fantasy", "rating": 4.4},
    {"movieId": 70, "title": "Harry Potter and the Sorcerer's Stone (2001)", "genres": "Adventure Fantasy", "rating": 4.0},
    {"movieId": 71, "title": "The Avengers (2012)", "genres": "Action Adventure Sci-Fi", "rating": 4.0},
    {"movieId": 72, "title": "Iron Man (2008)", "genres": "Action Adventure Sci-Fi", "rating": 4.1},
    {"movieId": 73, "title": "Guardians of the Galaxy (2014)", "genres": "Action Adventure Comedy Sci-Fi", "rating": 4.1},
    {"movieId": 74, "title": "Spider-Man: Into the Spider-Verse (2018)", "genres": "Action Adventure Animation Comedy", "rating": 4.4},
    {"movieId": 75, "title": "Coco (2017)", "genres": "Adventure Animation Children Comedy Fantasy Music", "rating": 4.3},
    {"movieId": 76, "title": "Up (2009)", "genres": "Adventure Animation Children Comedy Drama", "rating": 4.3},
    {"movieId": 77, "title": "WALL·E (2008)", "genres": "Adventure Animation Children Romance Sci-Fi", "rating": 4.3},
    {"movieId": 78, "title": "Finding Nemo (2003)", "genres": "Adventure Animation Children Comedy", "rating": 4.1},
    {"movieId": 79, "title": "Shrek (2001)", "genres": "Adventure Animation Children Comedy Fantasy Romance", "rating": 3.9},
    {"movieId": 80, "title": "Frozen (2013)", "genres": "Adventure Animation Children Comedy Fantasy Musical Romance", "rating": 3.9},
    {"movieId": 81, "title": "Moana (2016)", "genres": "Adventure Animation Children Comedy Fantasy Musical", "rating": 4.0},
    {"movieId": 82, "title": "Get Out (2017)", "genres": "Horror Mystery Thriller", "rating": 4.1},
    {"movieId": 83, "title": "A Quiet Place (2018)", "genres": "Drama Horror Sci-Fi Thriller", "rating": 4.0},
    {"movieId": 84, "title": "Hereditary (2018)", "genres": "Drama Horror Mystery Thriller", "rating": 3.8},
    {"movieId": 85, "title": "It (2017)", "genres": "Drama Horror", "rating": 3.9},
    {"movieId": 86, "title": "Parasite (2019)", "genres": "Comedy Drama Thriller", "rating": 4.5},
    {"movieId": 87, "title": "Knives Out (2019)", "genres": "Comedy Crime Drama Mystery Thriller", "rating": 4.3},
    {"movieId": 88, "title": "1917 (2019)", "genres": "Drama War", "rating": 4.3},
    {"movieId": 89, "title": "Joker (2019)", "genres": "Crime Drama Thriller", "rating": 4.2},
    {"movieId": 90, "title": "Once Upon a Time in Hollywood (2019)", "genres": "Comedy Drama", "rating": 3.9},
    {"movieId": 91, "title": "La La Land (2016)", "genres": "Comedy Drama Musical Romance", "rating": 4.1},
    {"movieId": 92, "title": "Whiplash (2014)", "genres": "Drama Music", "rating": 4.4},
    {"movieId": 93, "title": "Birdman (2014)", "genres": "Comedy Drama", "rating": 4.0},
    {"movieId": 94, "title": "The Grand Budapest Hotel (2014)", "genres": "Adventure Comedy Crime Drama Romance", "rating": 4.1},
    {"movieId": 95, "title": "Mad Max: Fury Road (2015)", "genres": "Action Adventure Sci-Fi Thriller", "rating": 4.2},
    {"movieId": 96, "title": "The Revenant (2015)", "genres": "Adventure Drama Western", "rating": 4.0},
    {"movieId": 97, "title": "Room (2015)", "genres": "Drama Thriller", "rating": 4.2},
    {"movieId": 98, "title": "Ex Machina (2014)", "genres": "Drama Sci-Fi Thriller", "rating": 4.1},
    {"movieId": 99, "title": "Her (2013)", "genres": "Drama Romance Sci-Fi", "rating": 4.1},
    {"movieId": 100, "title": "Gravity (2013)", "genres": "Drama Sci-Fi Thriller", "rating": 3.9},
]

# Sample user ratings matrix (userId → list of (movieId, rating))
USER_RATINGS = {
    1:  [(1, 5), (2, 3), (6, 4), (10, 4), (42, 5), (43, 5), (59, 5), (60, 5), (61, 5)],
    2:  [(1, 4), (3, 3), (17, 5), (25, 4), (44, 5), (64, 5), (91, 5), (92, 4)],
    3:  [(6, 5), (42, 5), (43, 4), (52, 5), (61, 5), (65, 5), (66, 5), (67, 5), (89, 4)],
    4:  [(45, 5), (75, 5), (76, 5), (77, 5), (78, 5), (79, 4), (80, 4), (1, 5)],
    5:  [(54, 5), (55, 5), (56, 4), (57, 5), (58, 4), (59, 5), (60, 5), (69, 5)],
    6:  [(82, 5), (83, 4), (84, 5), (85, 4), (52, 5), (67, 4)],
    7:  [(86, 5), (87, 5), (88, 4), (89, 4), (90, 3), (42, 5), (65, 5)],
    8:  [(91, 5), (92, 5), (93, 4), (94, 4), (44, 5), (17, 4), (99, 5)],
    9:  [(59, 5), (60, 5), (62, 5), (95, 5), (98, 4), (99, 5), (100, 4)],
    10: [(43, 5), (50, 5), (65, 5), (66, 5), (86, 5), (88, 5), (61, 5)],
}


class MovieRecommender:
    """Multi-strategy movie recommendation engine."""

    def __init__(self):
        self.movies_df = pd.DataFrame(MOVIES_DATA)
        self.movies_df.set_index("movieId", inplace=True)
        self._build_content_model()
        self._build_collab_model()

    # ── Content-Based Filtering ────────────────────────────────────────────

    def _build_content_model(self):
        """Build TF-IDF matrix from genre tags."""
        tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z\-]+")
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df["genres"])
        self.content_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.content_indices = pd.Series(
            range(len(self.movies_df)), index=self.movies_df["title"]
        )

    def content_recommend(self, title: str, n: int = 10) -> pd.DataFrame:
        """Return top-N similar movies by content (genres)."""
        # fuzzy title match
        matches = [t for t in self.movies_df["title"] if title.lower() in t.lower()]
        if not matches:
            return pd.DataFrame()
        query_title = matches[0]
        idx = self.content_indices[query_title]
        scores = list(enumerate(self.content_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1: n + 1]
        movie_indices = [i[0] for i in scores]
        sim_scores = [round(i[1], 3) for i in scores]
        result = self.movies_df.iloc[movie_indices][["title", "genres", "rating"]].copy()
        result["similarity"] = sim_scores
        return result.reset_index(drop=True)

    # ── Collaborative Filtering ───────────────────────────────────────────

    def _build_collab_model(self):
        """Build user-item matrix and SVD model."""
        movie_ids = list(self.movies_df.index)
        user_ids = list(USER_RATINGS.keys())
        matrix = np.zeros((len(user_ids), len(movie_ids)))
        movie_id_to_col = {mid: i for i, mid in enumerate(movie_ids)}

        for u_idx, uid in enumerate(user_ids):
            for mid, rating in USER_RATINGS[uid]:
                if mid in movie_id_to_col:
                    matrix[u_idx][movie_id_to_col[mid]] = rating

        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.movie_id_to_col = movie_id_to_col
        self.user_item_matrix = matrix

        # SVD decomposition
        sparse = csr_matrix(matrix)
        svd = TruncatedSVD(n_components=min(10, len(user_ids) - 1), random_state=42)
        self.user_factors = svd.fit_transform(sparse)
        self.movie_factors = svd.components_
        self.svd = svd

    def collab_recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """Return top-N movies for a user via collaborative filtering."""
        if user_id not in self.user_ids:
            return pd.DataFrame()

        u_idx = self.user_ids.index(user_id)
        # reconstruct predicted ratings
        predicted = self.user_factors[u_idx] @ self.movie_factors

        # exclude already-rated movies
        rated_mids = {mid for mid, _ in USER_RATINGS.get(user_id, [])}
        scores = []
        for col, mid in enumerate(self.movie_ids):
            if mid not in rated_mids:
                scores.append((mid, predicted[col]))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_mids = [s[0] for s in scores[:n]]
        top_scores = [round(s[1], 3) for s in scores[:n]]

        result = self.movies_df.loc[top_mids][["title", "genres", "rating"]].copy()
        result["predicted_score"] = top_scores
        return result.reset_index(drop=True)

    # ── User-Similarity Collaborative Filtering ───────────────────────────

    def user_similarity_recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """Find similar users and recommend their highly-rated movies."""
        if user_id not in self.user_ids:
            return pd.DataFrame()

        u_idx = self.user_ids.index(user_id)
        user_sims = cosine_similarity(
            self.user_item_matrix[u_idx].reshape(1, -1), self.user_item_matrix
        )[0]

        # top 3 similar users (exclude self)
        similar_users = np.argsort(user_sims)[::-1][1:4]
        rated_by_user = {mid for mid, _ in USER_RATINGS.get(user_id, [])}

        candidates = {}
        for su_idx in similar_users:
            sim = user_sims[su_idx]
            for mid, rating in USER_RATINGS[self.user_ids[su_idx]]:
                if mid not in rated_by_user:
                    if mid not in candidates:
                        candidates[mid] = 0
                    candidates[mid] += rating * sim

        top_mids = sorted(candidates, key=candidates.get, reverse=True)[:n]
        if not top_mids:
            return pd.DataFrame()

        result = self.movies_df.loc[[m for m in top_mids if m in self.movies_df.index]][
            ["title", "genres", "rating"]
        ].copy()
        return result.reset_index(drop=True)

    # ── Hybrid Recommendations ────────────────────────────────────────────

    def hybrid_recommend(self, title: str, user_id: int = None, n: int = 10) -> pd.DataFrame:
        """Blend content-based and collaborative recommendations."""
        content = self.content_recommend(title, n * 2)
        if content.empty:
            return pd.DataFrame()

        if user_id and user_id in self.user_ids:
            collab = self.collab_recommend(user_id, n * 2)
            if not collab.empty:
                # merge and re-rank
                collab_titles = set(collab["title"])
                content["hybrid_score"] = content.apply(
                    lambda r: r["similarity"] * 0.5 + (0.5 if r["title"] in collab_titles else 0),
                    axis=1,
                )
                return content.sort_values("hybrid_score", ascending=False).head(n).reset_index(drop=True)

        return content.head(n)

    # ── Utilities ─────────────────────────────────────────────────────────

    def search_movies(self, query: str) -> pd.DataFrame:
        mask = self.movies_df["title"].str.contains(query, case=False, na=False)
        return self.movies_df[mask][["title", "genres", "rating"]].reset_index()

    def get_top_rated(self, genre: str = None, n: int = 20) -> pd.DataFrame:
        df = self.movies_df.copy()
        if genre and genre != "All":
            df = df[df["genres"].str.contains(genre, case=False, na=False)]
        return df.nlargest(n, "rating")[["title", "genres", "rating"]].reset_index()

    def get_genres(self):
        all_genres = set()
        for g in self.movies_df["genres"]:
            for part in g.split():
                all_genres.add(part)
        return sorted(all_genres)

    def get_user_history(self, user_id: int) -> pd.DataFrame:
        ratings = USER_RATINGS.get(user_id, [])
        rows = []
        for mid, rating in ratings:
            if mid in self.movies_df.index:
                row = self.movies_df.loc[mid].to_dict()
                row["movieId"] = mid
                row["user_rating"] = rating
                rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
