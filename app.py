# restro/app.py
"""
Streamlit Restaurant Recommender App
Place this file at restro/app.py

Requirements:
  pip install streamlit pandas numpy sqlalchemy
Run:
  cd restro
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import pickle
from pathlib import Path
from sqlalchemy import create_engine, text

# -----------------------------
# Configuration / file paths
# -----------------------------
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "restaurant_recommender.db"

MODEL_DIR = BASE_DIR / "recommender_model"
SIMILARITY_PKL = MODEL_DIR / "similarity_matrix.pkl"
RESTAURANT_META_CSV = MODEL_DIR / "restaurant_metadata.csv"
RATING_MATRIX_CSV = MODEL_DIR / "rating_matrix.csv"
USER_PREFS_CSV = MODEL_DIR / "user_preferences.csv"  # optional local fallback

# -----------------------------
# Utilities
# -----------------------------
def get_db_connection():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    return engine

def init_db():
    """Create tables if they don't exist"""
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            password_hash TEXT,
            location TEXT,
            preferences TEXT
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ratings (
            rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            restaurant_id TEXT,
            rating INTEGER,
            review TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """))
    engine.dispose()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

# -----------------------------
# Load model/data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_restaurant_metadata():
    df = pd.read_csv(RESTAURANT_META_CSV)
    # Normalize columns: ensure 'restaurant_id', 'name', 'location', 'cuisine', 'rating', 'price' exist if possible
    # Attempt some common column names
    colmap = {}
    if 'restaurant_id' not in df.columns:
        for candidate in ['id', 'resto_id', 'restaurantId', 'restaurantId']:
            if candidate in df.columns:
                colmap[candidate] = 'restaurant_id'
                break
    if 'name' not in df.columns:
        for candidate in ['restaurant_name', 'rest_name', 'title']:
            if candidate in df.columns:
                colmap[candidate] = 'name'
                break
    if 'location' not in df.columns and 'area' in df.columns:
        colmap['area'] = 'location'
    if 'cuisine' not in df.columns:
        for candidate in ['food_type', 'cuisines', 'category']:
            if candidate in df.columns:
                colmap[candidate] = 'cuisine'
                break
    if colmap:
        df = df.rename(columns=colmap)
    # Fill missing fields to avoid errors
    if 'restaurant_id' not in df.columns:
        df.insert(0, 'restaurant_id', df.index.astype(str))
    if 'name' not in df.columns:
        df['name'] = df['restaurant_id'].astype(str)
    # Ensure rating numeric if exists
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    else:
        df['rating'] = np.nan
    # price normalization
    if 'price' not in df.columns:
        # try to infer
        for c in ['price_range', 'cost']:
            if c in df.columns:
                df = df.rename(columns={c: 'price'})
                break
    return df

@st.cache_data(show_spinner=False)
def load_similarity_matrix():
    # Could be a numpy array or DataFrame pickled.
    if not SIMILARITY_PKL.exists():
        st.warning("No similarity matrix found at recommender_model/similarity_matrix.pkl.")
        return None
    with open(SIMILARITY_PKL, 'rb') as f:
        sim = pickle.load(f)
    # If sim is a DataFrame keep it; if numpy array, return as-is
    return sim

@st.cache_data(show_spinner=False)
def load_rating_matrix():
    if not RATING_MATRIX_CSV.exists():
        return None
    df = pd.read_csv(RATING_MATRIX_CSV, index_col=0)
    # Many rating matrices store users as index and restaurants as columns.
    return df

# -----------------------------
# Recommendation logic
# -----------------------------
def get_user_row_from_rating_matrix(ratings_df: pd.DataFrame, username: str):
    """
    Try to find the user's row in the rating matrix.
    Accepts both where users are index or where columns are users.
    Returns a pandas Series of item ratings (index: restaurant_id).
    """
    if ratings_df is None:
        return None
    # if username matches index
    if str(username) in ratings_df.index.astype(str).tolist():
        row = ratings_df.loc[str(username)]
        return row.dropna()
    # if username present as column
    if str(username) in ratings_df.columns.astype(str).tolist():
        row = ratings_df[str(username)]
        return row.dropna()
    # otherwise return None
    return None

def recommend_for_user(username: str,
                       ratings_df: pd.DataFrame,
                       rest_meta: pd.DataFrame,
                       similarity,
                       top_n=10,
                       location=None,
                       preferences=None):
    """
    Item-based CF:
      - find items user rated positively (>=4 if integer scale) OR top-k
      - aggregate similarity vectors weighted by rating
      - remove items user already rated
      - return top_n restaurants from rest_meta
    If user has no history, fall back to content-based/popularity in location + preferences
    """
    # 1. Try to get user's rated items
    user_row = get_user_row_from_rating_matrix(ratings_df, username)
    if user_row is not None and len(user_row) > 0:
        # consider ratings >= 4 as positive signals; fallback: take top 5 rated by user
        try:
            # ensure numeric
            user_ratings = pd.to_numeric(user_row, errors='coerce').dropna()
        except Exception:
            user_ratings = user_row
        positive = user_ratings[user_ratings >= 4]
        if positive.empty:
            # pick top 5 items the user rated
            positive = user_ratings.sort_values(ascending=False).head(5)

        # similarity can be DataFrame (idx: restaurant_id)
        # We'll accumulate scores in a dict
        scores = {}
        # mapping restaurant ids
        for rest_id, rating in positive.items():
            rest_id = str(rest_id)
            # try to locate similarity vector
            if isinstance(similarity, pd.DataFrame):
                if rest_id not in similarity.index.astype(str).tolist():
                    continue
                sim_vec = similarity.loc[str(rest_id)]
                # convert to Series with rest ids
                sim_series = pd.Series(sim_vec.values, index=similarity.columns.astype(str))
            elif isinstance(similarity, np.ndarray):
                # if numpy array, we need indices mapping -- best-effort: assume order matches metadata
                # We'll map by metadata order: restaurant_meta['restaurant_id']
                sim_series = None
            else:
                sim_series = None

            if sim_series is None:
                continue

            # weight by rating
            weighted = sim_series * float(rating)
            for r, s in weighted.items():
                scores[str(r)] = scores.get(str(r), 0.0) + float(s)

        # create DataFrame of scores
        if not scores:
            # fallback
            candidate = fallback_recommendation(rest_meta, location, preferences, top_n)
            return candidate

        scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['score'])
        # remove items the user already rated
        already_rated = user_row.index.astype(str).tolist()
        scores_df = scores_df[~scores_df.index.isin(already_rated)]
        # join with metadata to filter by location & preferences
        meta = rest_meta.copy()
        meta['restaurant_id'] = meta['restaurant_id'].astype(str)
        merged = scores_df.merge(meta, left_index=True, right_on='restaurant_id', how='left')
        if location:
            merged = merged[merged['location'].astype(str).str.lower() == str(location).lower()]
        if preferences:
            # preferences is list or comma-separated string
            pref_list = preferences if isinstance(preferences, list) else [p.strip().lower() for p in str(preferences).split(',') if p.strip()]
            if pref_list:
                # try match cuisine or tags
                merged = merged[merged['cuisine'].astype(str).str.lower().apply(
                    lambda x: any(p in x for p in pref_list) if isinstance(x, str) else False
                ) | merged['tags'].astype(str).str.lower().apply(
                    lambda x: any(p in x for p in pref_list) if isinstance(x, str) else False
                )]
        merged = merged.sort_values('score', ascending=False)
        # return top_n merged
        return merged.head(top_n).drop(columns=['score']).reset_index(drop=True)
    else:
        # cold start -> fallback
        candidate = fallback_recommendation(rest_meta, location, preferences, top_n)
        return candidate

def fallback_recommendation(rest_meta: pd.DataFrame, location=None, preferences=None, top_n=10):
    """
    Returns top restaurants by metadata rating, filtered by location and preferences.
    """
    meta = rest_meta.copy()
    if location:
        meta = meta[meta['location'].astype(str).str.lower() == str(location).lower()]
    if preferences:
        pref_list = preferences if isinstance(preferences, list) else [p.strip().lower() for p in str(preferences).split(',') if p.strip()]
        if pref_list:
            meta = meta[meta['cuisine'].astype(str).str.lower().apply(
                lambda x: any(p in x for p in pref_list) if isinstance(x, str) else False
            ) | meta.get('tags', pd.Series([])).astype(str).str.lower().apply(
                lambda x: any(p in x for p in pref_list) if isinstance(x, str) else False
            )]
    # sort by rating if available, else random
    if 'rating' in meta.columns and not meta['rating'].isna().all():
        meta_sorted = meta.sort_values('rating', ascending=False)
    else:
        meta_sorted = meta.sample(frac=1.0, random_state=42)  # random shuffle
    return meta_sorted.head(top_n).reset_index(drop=True)

def get_similar_restaurants(restaurant_id: str, similarity, rest_meta: pd.DataFrame, top_n=6):
    """
    Return top-n similar restaurants to restaurant_id using similarity matrix.
    """
    if similarity is None:
        return pd.DataFrame()
    if isinstance(similarity, pd.DataFrame):
        rid = str(restaurant_id)
        if rid not in similarity.index.astype(str).tolist():
            return pd.DataFrame()
        sim_series = pd.Series(similarity.loc[rid].values, index=similarity.columns.astype(str))
        sim_series = sim_series.sort_values(ascending=False)
        sim_series = sim_series[sim_series.index != rid]  # exclude itself
        top_ids = sim_series.head(top_n).index.astype(str).tolist()
        meta = rest_meta.copy()
        meta['restaurant_id'] = meta['restaurant_id'].astype(str)
        result = meta[meta['restaurant_id'].isin(top_ids)]
        # preserve order of top_ids
        result = pd.DataFrame({ 'restaurant_id': top_ids }).merge(result, on='restaurant_id', how='left')
        return result
    else:
        return pd.DataFrame()

# -----------------------------
# DB user functions
# -----------------------------
def get_user(username: str):
    engine = get_db_connection()
    with engine.connect() as conn:
        res = conn.execute(text("SELECT username, email, location, preferences FROM users WHERE username = :u"), {"u": username}).fetchone()
    engine.dispose()
    return res

def create_user(username: str, email: str, password: str):
    password_hash = hash_password(password)
    engine = get_db_connection()
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO users (username, email, password_hash) VALUES (:u, :e, :p)"),
                         {"u": username, "e": email, "p": password_hash})
        engine.dispose()
        return True, "User created"
    except Exception as e:
        engine.dispose()
        return False, str(e)

def verify_user(username: str, password: str):
    password_hash = hash_password(password)
    engine = get_db_connection()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT username FROM users WHERE username = :u AND password_hash = :p"), {"u": username, "p": password_hash}).fetchone()
    engine.dispose()
    return row is not None

def update_user_location(username: str, location: str):
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("UPDATE users SET location = :loc WHERE username = :u"), {"loc": location, "u": username})
    engine.dispose()

def update_user_preferences(username: str, preferences: str):
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("UPDATE users SET preferences = :prefs WHERE username = :u"), {"prefs": preferences, "u": username})
    engine.dispose()

def save_user_rating(username: str, restaurant_id: str, rating: int, review: str = ""):
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO ratings (username, restaurant_id, rating, review) VALUES (:u, :rid, :r, :rev)"),
                     {"u": username, "rid": str(restaurant_id), "r": int(rating), "rev": review})
    engine.dispose()

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Kathmandu Restaurant Recommender", layout="wide")
    init_db()

    # Load model/data (cached)
    rest_meta = load_restaurant_metadata()
    sim = load_similarity_matrix()
    ratings_df = load_rating_matrix()

    # --- Sidebar: Authentication + navigation ---
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        st.sidebar.title("Welcome")
        choice = st.sidebar.selectbox("Menu", ["Login", "Sign up", "Continue as Guest"])
        if choice == "Login":
            st.sidebar.subheader("Login")
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login"):
                if verify_user(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    # load user details
                    user_row = get_user(username)
                    if user_row:
                        st.session_state['location'] = user_row['location']
                        st.session_state['preferences'] = user_row['preferences']
                    st.success(f"Logged in as {username}")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
        elif choice == "Sign up":
            st.sidebar.subheader("Create an account")
            new_username = st.sidebar.text_input("Choose a username", key="signup_user")
            email = st.sidebar.text_input("Email", key="signup_email")
            new_password = st.sidebar.text_input("Choose a password", type="password", key="signup_pass")
            if st.sidebar.button("Create account"):
                ok, msg = create_user(new_username, email, new_password)
                if ok:
                    st.success("Account created. You can now login.")
                else:
                    st.error(f"Could not create account: {msg}")
        else:
            # guest
            st.sidebar.write("Continue without signing in.")
            if st.sidebar.button("Proceed as Guest"):
                st.session_state['logged_in'] = True
                st.session_state['username'] = "guest"
                st.session_state['location'] = None
                st.session_state['preferences'] = None
                st.experimental_rerun()
        # show a brief landing
        st.title("🍽 Kathmandu Restaurant Recommender")
        st.markdown("Login or sign up to get personalized recommendations. Or continue as Guest.")
        st.write("---")
        st.subheader("About")
        st.write("This demo uses an item-based collaborative filtering model with precomputed item similarity. Set your location, save preferences, and get recommendations tailored for Kathmandu.")
        return  # do not continue into main app until logged in

    # When logged in:
    username = st.session_state.get('username', 'guest')
    st.sidebar.title(f"Hi, {username}")
    page = st.sidebar.radio("Navigate", ["Home", "Set My Location", "Explore", "Preferences", "My Reviews", "Admin (hidden)"])
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

    # --- Pages ---
    if page == "Home":
        st.title("Home — Recommendations")
        st.subheader(f"Welcome, {username}!")
        user_location = st.session_state.get('location', None)
        if user_location:
            st.info(f"Showing recommendations for location: **{user_location}**")
        else:
            st.warning("Please set your location via 'Set My Location' for nearby recommendations.")

        # Use preferences from session or DB
        prefs = st.session_state.get('preferences', None)
        # show a small preferences summary
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write("Your location:")
            st.write(user_location if user_location else "— not set —")
            st.write("Your preferences:")
            st.write(prefs if prefs else "— not set —")
        with col2:
            # call recommendation engine
            recs = recommend_for_user(username=username,
                                      ratings_df=ratings_df,
                                      rest_meta=rest_meta,
                                      similarity=sim,
                                      top_n=10,
                                      location=user_location,
                                      preferences=prefs)
            if recs is None or recs.empty:
                st.write("No recommendations available. Try updating preferences or adding some ratings in 'My Reviews'.")
            else:
                for idx, row in recs.iterrows():
                    st.markdown(f"### {row.get('name', row.get('restaurant_id'))}")
                    st.text(f"Cuisine: {row.get('cuisine', 'N/A')} | Location: {row.get('location', 'N/A')} | Price: {row.get('price', 'N/A')}")
                    if 'rating' in row and not pd.isna(row['rating']):
                        st.text(f"Rating: ⭐ {row['rating']}")
                    # show similar restaurants button
                    rest_id = str(row.get('restaurant_id'))
                    if st.button(f"Show similar to {row.get('name', '')}", key=f"sim_{idx}"):
                        with st.expander("Similar restaurants"):
                            similar = get_similar_restaurants(rest_id, sim, rest_meta, top_n=6)
                            if similar.empty:
                                st.write("No similar restaurants found.")
                            else:
                                for _, s in similar.iterrows():
                                    st.write(f"- {s.get('name','')} ({s.get('cuisine','')}) — {s.get('location','')}")
                    st.markdown("---")

    elif page == "Set My Location":
        st.title("Set My Location")
        st.write("Select your current area (manual input). This will be saved to your profile.")
        # unique sorted locations from metadata
        all_locations = sorted(rest_meta['location'].dropna().unique().astype(str).tolist())
        if not all_locations:
            st.error("No location data available in restaurant metadata.")
        else:
            loc = st.selectbox("Select an area", [""] + all_locations)
            if st.button("Save Location"):
                if username != "guest":
                    update_user_location(username, loc)
                st.session_state['location'] = loc
                st.success(f"Location set to {loc}")
    elif page == "Explore":
        st.title("Explore Restaurants")
        st.write("Search and filter restaurants.")
        search = st.text_input("Search by name or cuisine")
        cuisine_filter = st.multiselect("Cuisine", sorted(rest_meta['cuisine'].dropna().unique().astype(str).tolist()))
        location_filter = st.selectbox("Area (optional)", ["Any"] + sorted(rest_meta['location'].dropna().unique().astype(str).tolist()))
        price_options = []
        if 'price' in rest_meta.columns:
            price_options = sorted(rest_meta['price'].dropna().unique().astype(str).tolist())
        price_filter = st.selectbox("Price (optional)", ["Any"] + price_options) if price_options else "Any"

        df = rest_meta.copy()
        if search:
            mask = df['name'].astype(str).str.contains(search, case=False, na=False) | df['cuisine'].astype(str).str.contains(search, case=False, na=False)
            df = df[mask]
        if cuisine_filter:
            df = df[df['cuisine'].isin(cuisine_filter)]
        if location_filter and location_filter != "Any":
            df = df[df['location'] == location_filter]
        if price_filter and price_filter != "Any":
            df = df[df['price'].astype(str) == price_filter]

        st.write(f"Found {len(df)} restaurants")
        for _, r in df.head(200).iterrows():  # limit to 200 rows for speed
            st.markdown(f"**{r.get('name','')}** — {r.get('cuisine','')} — {r.get('location','')}")
            st.text(f"Price: {r.get('price','N/A')} | Rating: {r.get('rating','N/A')}")
            if st.button(f"See similar to {r.get('name','')}", key=f"explore_sim_{r.get('restaurant_id')}"):
                simdf = get_similar_restaurants(str(r.get('restaurant_id')), sim, rest_meta, top_n=6)
                if simdf.empty:
                    st.write("No similar restaurants.")
                else:
                    for _, s in simdf.iterrows():
                        st.write(f"- {s.get('name','')} ({s.get('location','')})")
            st.markdown("---")

    elif page == "Preferences":
        st.title("My Preferences")
        st.write("Set your favorite cuisines and budget to improve recommendations.")
        cuisine_list = sorted(rest_meta['cuisine'].dropna().unique().astype(str).tolist())
        chosen = st.multiselect("Favorite cuisines", cuisine_list, default=st.session_state.get('preferences', None))
        budget = st.selectbox("Preferred budget", ["Any", "Low", "Medium", "High"])
        if st.button("Save Preferences"):
            prefs_str = ",".join(chosen) + (f";budget={budget}" if budget and budget != "Any" else "")
            if username != "guest":
                update_user_preferences(username, prefs_str)
            st.session_state['preferences'] = prefs_str
            st.success("Preferences saved.")

    elif page == "My Reviews":
        st.title("My Reviews & Ratings")
        st.write("Submit a rating & short review for a restaurant.")
        # restaurant selection
        choices = rest_meta[['restaurant_id', 'name']].drop_duplicates().to_dict('records')
        id_to_name = {str(r['restaurant_id']): r['name'] for r in choices}
        select_id = st.selectbox("Select restaurant", [""] + list(id_to_name.keys()), format_func=lambda x: id_to_name.get(str(x), "") if x else "")
        rating = st.slider("Rating (1-5)", 1, 5, 4)
        review = st.text_area("Short review (optional)")
        if st.button("Submit Review"):
            if not select_id:
                st.error("Please choose a restaurant.")
            else:
                save_user_rating(username, select_id, rating, review)
                st.success("Thank you — your review has been saved.")
    else:
        # Admin (hidden) - simple stats
        st.title("Admin Dashboard (basic)")
        engine = get_db_connection()
        with engine.connect() as conn:
            user_count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
            rating_count = conn.execute(text("SELECT COUNT(*) FROM ratings")).scalar()
        st.write(f"Users: {user_count}  |  Ratings stored: {rating_count}")
        st.write("Top cuisines in metadata:")
        st.write(rest_meta['cuisine'].value_counts().head(10))

if __name__ == "__main__":
    main()
