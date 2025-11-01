# restro/app.py
"""
Streamlit Restaurant Recommender - Robust Version
Drop into your restro/ folder and run:
    streamlit run app.py
This version:
 - Uses actual locations from restaurant metadata
 - Clears signup inputs after success
 - Defensive checks to avoid sqlite / file errors on Streamlit Cloud
 - Improved Home display (cards/grid) and Explore page with pagination
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import pickle
import math
import traceback
from pathlib import Path

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "restaurant_recommender.db"
MODEL_DIR = BASE_DIR / "recommender_model"

SIMILARITY_PKL = MODEL_DIR / "similarity_matrix.pkl"
RESTAURANT_META_CSV = MODEL_DIR / "restaurant_metadata.csv"
RATING_MATRIX_CSV = MODEL_DIR / "rating_matrix.csv"

# ---------------------------
# Ensure DB and tables exist (run at import so safe)
# ---------------------------
def ensure_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT,
                password_hash TEXT,
                location TEXT,
                preferences TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                restaurant_id TEXT,
                rating INTEGER,
                review TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    finally:
        conn.close()

ensure_db()

# ---------------------------
# Utilities
# ---------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256((password or "").encode("utf-8")).hexdigest()

# Safe read CSV
def safe_read_csv(path: Path, index_col=None):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=index_col)
    except Exception:
        return None

# ---------------------------
# Data loaders (cached)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_restaurant_metadata():
    df = safe_read_csv(RESTAURANT_META_CSV)
    if df is None:
        # create a tiny fallback so app doesn't crash
        df = pd.DataFrame([{
            "restaurant_id": "r1", "name": "Example Resto", "cuisine": "Nepali",
            "location": "Thamel", "rating": 4.2, "price": "Medium", "tags": "vegan"
        }])
    # standardize expected columns
    if "restaurant_id" not in df.columns:
        df.insert(0, "restaurant_id", df.index.astype(str))
    if "name" not in df.columns:
        df["name"] = df["restaurant_id"].astype(str)
    if "cuisine" not in df.columns:
        df["cuisine"] = df.get("category", "Various")
    if "location" not in df.columns:
        df["location"] = df.get("area", "Unknown")
    if "rating" not in df.columns:
        df["rating"] = np.nan
    if "tags" not in df.columns:
        df["tags"] = ""
    # ensure types
    df["restaurant_id"] = df["restaurant_id"].astype(str)
    df["name"] = df["name"].astype(str)
    df["location"] = df["location"].astype(str)
    df["cuisine"] = df["cuisine"].astype(str)
    df["tags"] = df["tags"].astype(str)
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_similarity():
    if not SIMILARITY_PKL.exists():
        return None
    try:
        with open(SIMILARITY_PKL, "rb") as f:
            sim = pickle.load(f)
        # If numpy array, convert to DataFrame keyed to metadata order
        if isinstance(sim, (list, tuple, np.ndarray)) and not isinstance(sim, pd.DataFrame):
            sim = np.array(sim)
            # we cannot reliably map indices here — caller should set up proper DataFrame when training
            return sim
        return sim
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_rating_matrix():
    df = safe_read_csv(RATING_MATRIX_CSV, index_col=0)
    return df  # may be None

# ---------------------------
# DB functions (sqlite3)
# ---------------------------
def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT username, email, location, preferences FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def create_user(username, email, password):
    """Return (ok:bool, message:str)."""
    if not username or not password:
        return False, "Username and password are required."
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            conn.close()
            return False, "Username already taken. Choose another."
        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email or "", hash_password(password)))
        conn.commit()
        conn.close()
        return True, "User created."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, f"DB error: {e}"

def verify_user(username, password):
    if not username or not password:
        return False
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    stored_hash = row[0]
    return stored_hash == hash_password(password)

def update_user_location(username, location):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE users SET location = ? WHERE username = ?", (location, username))
    conn.commit()
    conn.close()

def update_user_preferences(username, prefs):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE users SET preferences = ? WHERE username = ?", (prefs, username))
    conn.commit()
    conn.close()

def save_user_rating(username, restaurant_id, rating, review):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO ratings (username, restaurant_id, rating, review) VALUES (?, ?, ?, ?)",
                (username, restaurant_id, int(rating), review or ""))
    conn.commit()
    conn.close()

# ---------------------------
# Recommendation helpers
# ---------------------------
def get_similar_restaurants(restaurant_id, similarity, rest_meta, top_n=6):
    """
    similarity: either DataFrame indexed by restaurant_id, or None.
    Returns DataFrame of similar restaurants preserving order.
    """
    if similarity is None:
        return pd.DataFrame()
    try:
        if isinstance(similarity, pd.DataFrame):
            rid = str(restaurant_id)
            if rid not in similarity.index.astype(str).tolist():
                return pd.DataFrame()
            sims = similarity.loc[rid]
            sims = pd.Series(sims.values, index=similarity.columns.astype(str))
            sims = sims.sort_values(ascending=False)
            sims = sims[sims.index != rid]
            top_ids = list(sims.head(top_n).index.astype(str))
        elif isinstance(similarity, np.ndarray):
            # fallback: assume similarity aligned to rest_meta order
            idx_map = rest_meta['restaurant_id'].astype(str).tolist()
            if restaurant_id not in idx_map:
                return pd.DataFrame()
            i = idx_map.index(str(restaurant_id))
            vec = similarity[i]
            order = np.argsort(-vec)
            top_idx = [idx_map[j] for j in order if idx_map[j] != str(restaurant_id)]
            top_ids = top_idx[:top_n]
        else:
            return pd.DataFrame()
        res = rest_meta[rest_meta['restaurant_id'].astype(str).isin(top_ids)].copy()
        # keep order
        res['order'] = res['restaurant_id'].apply(lambda x: top_ids.index(str(x)) if str(x) in top_ids else 999)
        res = res.sort_values('order').drop(columns=['order'])
        return res
    except Exception:
        return pd.DataFrame()

def recommend_for_user_simple(username, ratings_df, rest_meta, similarity, location=None, prefs=None, top_n=10):
    """
    Simpler robust recommendation:
     - If user has explicit ratings in ratings table, use those to boost similar items (if similarity present)
     - Otherwise fallback to top-rated filtered by location & prefs
    """
    # Fetch user ratings from DB (ratings table)
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT restaurant_id, rating FROM ratings WHERE username = ?", (username,))
        rows = cur.fetchall()
        conn.close()
    except Exception:
        rows = []

    if rows:
        # aggregate top rated by user
        user_rated = {str(rid): float(r) for rid, r in rows}
        # if similarity provided, score others
        if isinstance(similarity, pd.DataFrame):
            scores = {}
            for rid, rscore in user_rated.items():
                if rid not in similarity.index.astype(str).tolist():
                    continue
                sim_series = pd.Series(similarity.loc[rid].values, index=similarity.columns.astype(str))
                for item_id, sim_val in sim_series.items():
                    scores[item_id] = scores.get(item_id, 0.0) + sim_val * (rscore)
            # remove already rated
            for rated in user_rated.keys():
                scores.pop(rated, None)
            if scores:
                scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['score']).reset_index().rename(columns={'index':'restaurant_id'})
                merged = scores_df.merge(rest_meta, on='restaurant_id', how='left')
                if location:
                    merged = merged[merged['location'].astype(str).str.lower() == str(location).lower()]
                if prefs:
                    pref_list = [p.strip().lower() for p in str(prefs).split(',') if p.strip()]
                    if pref_list:
                        merged = merged[ merged['cuisine'].astype(str).str.lower().apply(lambda x: any(p in x for p in pref_list))
                                        | merged['tags'].astype(str).str.lower().apply(lambda x: any(p in x for p in pref_list)) ]
                merged = merged.sort_values('score', ascending=False)
                return merged.head(top_n).drop(columns=['score']).reset_index(drop=True)
        # fallback to top-rated by restaurant metadata but exclude already rated
        rated_ids = set(user_rated.keys())
        candidates = rest_meta[~rest_meta['restaurant_id'].astype(str).isin(rated_ids)].copy()
    else:
        candidates = rest_meta.copy()

    # apply location / prefs filters
    if location:
        candidates = candidates[candidates['location'].astype(str).str.lower() == str(location).lower()]
    if prefs:
        pref_list = [p.strip().lower() for p in str(prefs).split(',') if p.strip()]
        if pref_list:
            candidates = candidates[ candidates['cuisine'].astype(str).str.lower().apply(lambda x: any(p in x for p in pref_list))
                                   | candidates['tags'].astype(str).str.lower().apply(lambda x: any(p in x for p in pref_list)) ]

    # sort by rating (if available) then name
    if 'rating' in candidates.columns and not candidates['rating'].isna().all():
        candidates = candidates.sort_values(['rating', 'name'], ascending=[False, True])
    else:
        candidates = candidates.sort_values('name')
    return candidates.head(top_n).reset_index(drop=True)

# ---------------------------
# UI helpers
# ---------------------------
def restaurant_card_cols(n_cols=3):
    return st.columns(n_cols)

def show_restaurant_card(row, index_key_prefix="", similarity=None, rest_meta=None):
    # single card display with expander for similar restaurants
    name = row.get('name', 'Unknown')
    cuisine = row.get('cuisine', 'N/A')
    location = row.get('location', 'N/A')
    rating = row.get('rating', None)
    price = row.get('price', 'N/A')
    tags = row.get('tags', '')
    rid = str(row.get('restaurant_id', ''))

    st.markdown(f"**{name}**")
    st.text(f"{cuisine} — {location} — {price}")
    if pd.notna(rating):
        st.caption(f"⭐ {rating}")
    if tags:
        st.write(f"Tags: {tags}")
    with st.expander("Show details & actions"):
        cols = st.columns([2,1,1])
        with cols[0]:
            st.write(row.get('description', 'No description available.'))
        with cols[1]:
            if st.button("Show similar", key=f"sim_btn_{index_key_prefix}_{rid}"):
                sim_df = get_similar_restaurants(rid, similarity, rest_meta, top_n=6) if similarity is not None and rest_meta is not None else pd.DataFrame()
                if sim_df is None or sim_df.empty:
                    st.info("No similar restaurants available.")
                else:
                    for _, s in sim_df.iterrows():
                        st.write(f"- {s.get('name','')} ({s.get('cuisine','')}) — {s.get('location','')}")
        with cols[2]:
            st.write("Rate:")
            r = st.selectbox("Your rating", [5,4,3,2,1], index=0, key=f"rate_select_{index_key_prefix}_{rid}")
            rev = st.text_area("Review (optional)", key=f"rev_{index_key_prefix}_{rid}", height=60)
            if st.button("Submit review", key=f"submit_rev_{index_key_prefix}_{rid}"):
                user = st.session_state.get("username", "guest")
                save_user_rating(user, rid, r, rev)
                st.success("Thanks! Your review was saved.")

# ---------------------------
# Main app
# ---------------------------
def main():
    st.set_page_config(page_title="Kathmandu Restaurant Recommender", layout="wide")
    st.title("Kathmandu Restaurant Recommender 🍽️")

    # Load data (cached)
    rest_meta = load_restaurant_metadata()
    similarity = load_similarity()
    rating_matrix = load_rating_matrix()

    # session defaults
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "location" not in st.session_state:
        st.session_state["location"] = None
    if "preferences" not in st.session_state:
        st.session_state["preferences"] = None

    # Centered login/signup when not logged in
    if not st.session_state["logged_in"]:
        st.markdown("<h3 style='text-align:center;'>Welcome — Sign in or create an account</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            tab_login, tab_signup, tab_guest = st.tabs(["🔐 Login", "🆕 Sign Up", "👀 Guest Mode"])

            with tab_login:
                st.subheader("Login")
                login_user = st.text_input("Username", key="login_user")
                login_pass = st.text_input("Password", type="password", key="login_pass")
                if st.button("Login", use_container_width=True):
                    try:
                        if verify_user(login_user, login_pass):
                            st.session_state["logged_in"] = True
                            st.session_state["username"] = login_user
                            user = get_user(login_user)
                            if user:
                                st.session_state["location"] = user.get("location", None)
                                st.session_state["preferences"] = user.get("preferences", None)
                            st.success(f"Welcome back, {login_user}!")
                            st.experimental_rerun()
                        else:
                            st.error("Invalid username or password.")
                    except Exception as e:
                        st.error("Error while logging in. Try again.")
                        st.write(str(e))

            with tab_signup:
                st.subheader("Create account")
                signup_user = st.text_input("Choose username", key="signup_user")
                signup_email = st.text_input("Email (optional)", key="signup_email")
                signup_pass = st.text_input("Choose password", type="password", key="signup_pass")
                if st.button("Sign Up", use_container_width=True):
                    ok, msg = create_user(signup_user, signup_email, signup_pass)
                    if ok:
                        # clear inputs
                        st.session_state["signup_user"] = ""
                        st.session_state["signup_email"] = ""
                        st.session_state["signup_pass"] = ""
                        st.success("Account created. Please login using the Login tab.")
                    else:
                        st.error(msg)

            with tab_guest:
                st.subheader("Continue as guest")
                st.write("You can explore restaurants without creating an account.")
                if st.button("Continue as Guest", use_container_width=True):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = "guest"
                    st.experimental_rerun()
        st.stop()

    # --- logged-in UI with sidebar navigation ---
    username = st.session_state.get("username", "guest")
    st.sidebar.title(f"Hi, {username}")
    page = st.sidebar.radio("Navigation", ["Home", "Set My Location", "Explore", "Preferences", "My Reviews", "Logout"])

    if page == "Logout":
        st.session_state.clear()
        st.success("You have been logged out.")
        st.experimental_rerun()
        return

    # --- SET MY LOCATION page: populate from rest_meta locations (unique) ---
    if page == "Set My Location":
        st.header("Set My Location")
        locations = sorted(rest_meta['location'].dropna().unique().astype(str).tolist())
        if not locations:
            st.error("No location data available in restaurant metadata.")
        else:
            chosen = st.selectbox("Choose your area", [""] + locations, index=0)
            if st.button("Save Location"):
                if chosen == "" or chosen is None:
                    st.warning("Choose a valid location from the dropdown.")
                else:
                    st.session_state['location'] = chosen
                    if username != "guest":
                        try:
                            update_user_location(username, chosen)
                        except Exception:
                            st.error("Could not save location to DB (check logs).")
                    st.success(f"Location set to: {chosen}")

    # --- HOME page: nicer card/grid view ---
    elif page == "Home":
        st.header("Recommended for you")
        user_loc = st.session_state.get("location", None)
        prefs = st.session_state.get("preferences", None)
        if user_loc is None or user_loc == "":
            st.info("Set your location first under 'Set My Location' to see nearby recommendations.")
        # Build recommendations
        try:
            recs = recommend_for_user_simple(username, rating_matrix, rest_meta, similarity, location=user_loc, prefs=prefs, top_n=18)
        except Exception:
            recs = rest_meta.head(18)
        if recs is None or recs.empty:
            st.write("No recommendations found. Try Explore or update your preferences.")
        else:
            # show in a responsive grid: 3 columns
            cols = st.columns(3)
            for i, (_, row) in enumerate(recs.iterrows()):
                col = cols[i % 3]
                with col:
                    # simple card
                    st.markdown(f"### {row.get('name','')}")
                    st.write(f"**{row.get('cuisine','')}** — {row.get('location','')}")
                    r = row.get('rating', None)
                    if pd.notna(r):
                        st.write(f"⭐ {r}")
                    if row.get('tags'):
                        st.write(f"Tags: {row.get('tags')}")
                    # actions
                    if st.button("Show similar", key=f"home_sim_{i}"):
                        sim_df = get_similar_restaurants(str(row['restaurant_id']), similarity, rest_meta, top_n=6)
                        if sim_df is None or sim_df.empty:
                            st.info("No similar restaurants available.")
                        else:
                            for _, s in sim_df.iterrows():
                                st.write(f"- {s.get('name','')} ({s.get('location','')})")
                    # quick review button (opens expander)
                    with st.expander("Leave a rating / review"):
                        rt = st.selectbox("Rating", [5,4,3,2,1], key=f"home_rate_{i}")
                        rv = st.text_area("Review", key=f"home_rev_{i}", height=80)
                        if st.button("Submit review", key=f"home_rev_submit_{i}"):
                            try:
                                save_user_rating(username, str(row['restaurant_id']), rt, rv)
                                st.success("Thanks — review saved.")
                            except Exception:
                                st.error("Could not save review (DB error).")

    # --- EXPLORE page with filters + pagination ---
    elif page == "Explore":
        st.header("Explore Restaurants")
        search = st.text_input("Search by name, cuisine, tag or location")
        all_cuisines = sorted(rest_meta['cuisine'].dropna().unique().astype(str).tolist())
        sel_cuisine = st.multiselect("Filter by cuisine", all_cuisines)
        all_locations = sorted(rest_meta['location'].dropna().unique().astype(str).tolist())
        sel_location = st.selectbox("Filter by location", ["Any"] + all_locations)
        # apply filters
        df = rest_meta.copy()
        if search:
            s = search.strip()
            mask = df['name'].str.contains(s, case=False, na=False) | \
                   df['cuisine'].str.contains(s, case=False, na=False) | \
                   df['tags'].str.contains(s, case=False, na=False) | \
                   df['location'].str.contains(s, case=False, na=False)
            df = df[mask]
        if sel_cuisine:
            df = df[df['cuisine'].isin(sel_cuisine)]
        if sel_location and sel_location != "Any":
            df = df[df['location'] == sel_location]
        st.write(f"Found {len(df)} restaurants")
        # pagination
        per_page = 12
        total_pages = max(1, math.ceil(len(df) / per_page))
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start = (page_num - 1) * per_page
        end = start + per_page
        page_df = df.iloc[start:end]
        # display paginated grid
        cols = st.columns(3)
        for i, (_, row) in enumerate(page_df.iterrows()):
            col = cols[i % 3]
            with col:
                st.markdown(f"**{row.get('name','')}**")
                st.write(f"{row.get('cuisine','')} — {row.get('location','')}")
                if pd.notna(row.get('rating', None)):
                    st.write(f"⭐ {row.get('rating')}")
                if st.button("Show similar", key=f"explore_sim_{i}"):
                    sim_df = get_similar_restaurants(str(row['restaurant_id']), similarity, rest_meta, top_n=6)
                    if sim_df is None or sim_df.empty:
                        st.info("No similar restaurants.")
                    else:
                        for _, s in sim_df.iterrows():
                            st.write(f"- {s.get('name','')} ({s.get('location','')})")

    # --- PREFERENCES page ---
    elif page == "Preferences":
        st.header("My Preferences")
        # read existing preferences if any
        prefs_current = st.session_state.get('preferences', "")
        cuisines = sorted(rest_meta['cuisine'].dropna().unique().astype(str).tolist())
        chosen = st.multiselect("Favorite cuisines (select one or more)", cuisines, default=[c for c in cuisines if c in (prefs_current or "")][:0])
        budget = st.selectbox("Preferred budget (optional)", ["Any", "Low", "Medium", "High"])
        if st.button("Save Preferences"):
            prefs_str = ",".join(chosen)
            st.session_state['preferences'] = prefs_str
            if username != "guest":
                try:
                    update_user_preferences(username, prefs_str)
                except Exception:
                    st.error("Could not save preferences to DB.")
            st.success("Preferences saved.")

    # --- MY REVIEWS page ---
    elif page == "My Reviews":
        st.header("Submit a rating / review")
        names = rest_meta['name'].tolist()
        chosen = st.selectbox("Select restaurant", names)
        rating = st.slider("Rating (1-5)", 1, 5, 4)
        review = st.text_area("Write a short review (optional)")
        if st.button("Submit Review"):
            rid = rest_meta.loc[rest_meta['name'] == chosen, 'restaurant_id'].iloc[0]
            try:
                save_user_rating(username, str(rid), rating, review)
                st.success("Thanks — your review was saved.")
            except Exception:
                st.error("Could not save review — DB error.")

    # end main

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # catch-all to prevent app crash; show friendly error and log minimal info
        st.error("An unexpected error occurred. Check logs for details.")
        st.write("Error summary:")
        st.text(traceback.format_exc().splitlines()[-1])
