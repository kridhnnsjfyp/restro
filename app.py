"""
FINAL FYP APP — KRISH CHAKRADHAR (00020758)
Restaurant Recommender — Enhanced Rating & Reviews UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import pickle
import math
from pathlib import Path

# ============================
# CONFIG & PATHS
# ============================
st.set_page_config(page_title="Kathmandu Restaurant Recommender", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "restaurant_recommender.db"
MODEL_DIR = BASE_DIR / "recommender_model"

SIMILARITY_PKL = MODEL_DIR / "similarity_matrix.pkl"
RESTAURANT_META_CSV = MODEL_DIR / "restaurant_metadata.csv"

# ============================
# INIT DB
# ============================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            location TEXT,
            preferences TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            restaurant_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            review TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ============================
# UTILS
# ============================
def hash_password(pw): 
    return hashlib.sha256(pw.encode()).hexdigest()

def safe_read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_metadata():
    df = safe_read_csv(RESTAURANT_META_CSV)
    if df.empty:
        return pd.DataFrame([{
            "restaurant_id": "r1", "name": "Sample Cafe", "cuisine": "Multi-Cuisine",
            "location": "Thamel", "rating": 4.0, "price": "Medium", "tags": ""
        }])

    rename_map = {
        "Restaurant Name": "name", "restaurant_name": "name", "name": "name",
        "Cuisine Type": "cuisine", "cuisine": "cuisine",
        "Location": "location", "location": "location",
        "rating": "rating", "Rating": "rating", "Ratings": "rating", 
        "Average Rating": "rating", "avg_rating": "rating",
        "price": "price", "Price Range": "price", "Price": "price",
        "tags": "tags"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    if "restaurant_id" not in df.columns:
        df["restaurant_id"] = df.index.astype(str)
    if "name" not in df.columns:
        df["name"] = "Restaurant " + df["restaurant_id"]
    if "cuisine" not in df.columns:
        df["cuisine"] = "Various"
    if "location" not in df.columns:
        df["location"] = "Kathmandu"
    if "price" not in df.columns:
        df["price"] = "Medium"
    if "tags" not in df.columns:
        df["tags"] = ""

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors='coerce')

    df = df.fillna("")
    df["restaurant_id"] = df["restaurant_id"].astype(str).str.strip()
    return df.reset_index(drop=True)

@st.cache_data
def load_similarity():
    if not SIMILARITY_PKL.exists():
        return None
    try:
        with open(SIMILARITY_PKL, "rb") as f:
            sim = pickle.load(f)
        return sim
    except:
        return None

# ============================
# DB FUNCTIONS
# ============================
def get_user(username):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT username, email, location, preferences FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()
        return dict(zip(["username", "email", "location", "preferences"], row)) if row else None
    except:
        return None

def create_user(username, email, password):
    if not username or not password:
        return False, "Username and password required."
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
        if cur.fetchone():
            conn.close()
            return False, "Username taken."
        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email or "", hash_password(password)))
        conn.commit()
        conn.close()
        return True, "Account created."
    except Exception as e:
        return False, f"Error: {e}"

def verify_user(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()
        return row and row[0] == hash_password(password)
    except:
        return False

def update_user_location(username, location):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE users SET location=? WHERE username=?", (location, username))
        conn.commit()
        conn.close()
    except:
        pass

def update_user_preferences(username, prefs):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE users SET preferences=? WHERE username=?", (prefs, username))
        conn.commit()
        conn.close()
    except:
        pass

def save_user_rating(username, restaurant_id, rating, review=""):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ratings (username, restaurant_id, rating, review) VALUES (?, ?, ?, ?)",
            (username, restaurant_id, int(rating), review or "")
        )
        conn.commit()
        conn.close()
    except:
        pass

def get_reviews(restaurant_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT username, rating, review, created_at FROM ratings WHERE restaurant_id=? ORDER BY created_at DESC",
            conn, params=(restaurant_id,)
        )
        conn.close()
        return df
    except:
        return pd.DataFrame()

# ============================
# RECOMMENDATION
# ============================
def recommend_user(username, meta, similarity, location=None, prefs=None, top_n=12):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT restaurant_id, rating FROM ratings WHERE username=?", (username,))
        rows = cur.fetchall()
        conn.close()
    except:
        rows = []

    rated = {r[0]: r[1] for r in rows} if rows else {}
    candidates = meta[~meta['restaurant_id'].isin(rated.keys())].copy() if rated else meta.copy()

    if location and location.strip():
        candidates = candidates[candidates['location'].astype(str).str.contains(location, case=False, na=False)]

    if prefs and prefs.strip():
        prefs_list = [p.strip().lower() for p in prefs.split(",") if p.strip()]
        if prefs_list:
            mask = pd.Series(False, index=candidates.index)
            for pref in prefs_list:
                mask |= candidates['name'].str.lower().str.contains(pref, na=False)
                mask |= candidates['cuisine'].str.lower().str.contains(pref, na=False)
                if 'tags' in candidates.columns:
                    mask |= candidates['tags'].str.lower().str.contains(pref, na=False)
            candidates = candidates[mask]

    if 'rating' in candidates.columns:
        numeric_mask = pd.to_numeric(candidates['rating'], errors='coerce').notna()
        if numeric_mask.any():
            candidates = candidates[numeric_mask].sort_values('rating', ascending=False)
            candidates = pd.concat([candidates, candidates[~numeric_mask]])
        else:
            candidates = candidates.sort_values('name')
    else:
        candidates = candidates.sort_values('name')

    return candidates.head(top_n).reset_index(drop=True)

# ============================
# RESTAURANT CARD (⭐ ENHANCED)
# ============================
def restaurant_card(row, key_prefix, meta, similarity):
    with st.container():
        st.markdown(f"### {row['name']}")
        st.caption(f"{row.get('cuisine','')} • {row.get('location','')} • {row.get('price','N/A')}")

        if pd.notna(row.get("rating", "")):
            rv = float(row["rating"])
            stars = "⭐" * int(round(rv))
            st.markdown(f"{stars} **{rv:.1f}**")

        if row.get("tags"):
            tags = " ".join([f"`{t.strip()}`" for t in row["tags"].split(",") if t.strip()])
            st.markdown(tags)

        with st.expander("Rate & Reviews"):
            # Rating input — star slider
            st.markdown("#### ⭐ Rate this restaurant")
            rating = st.slider(
                "Select rating", 
                min_value=1, max_value=5, step=1, value=5,
                key=f"rating_slider_{key_prefix}_{row['restaurant_id']}"
            )
            review = st.text_area(
                "Write your review (optional)", 
                key=f"review_text_{key_prefix}_{row['restaurant_id']}", 
                height=80
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Submit Review", key=f"submit_{key_prefix}_{row['restaurant_id']}"):
                    save_user_rating(st.session_state.username, row['restaurant_id'], rating, review)
                    st.success("✅ Review submitted successfully!")

            with col2:
                if st.button("Show Reviews", key=f"show_{key_prefix}_{row['restaurant_id']}"):
                    reviews_df = get_reviews(row['restaurant_id'])
                    if reviews_df.empty:
                        st.info("No reviews yet for this restaurant.")
                    else:
                        st.markdown("### 📝 Recent Reviews")
                        for _, r in reviews_df.iterrows():
                            stars = "⭐" * int(r['rating'])
                            st.markdown(f"**{r['username']}** — {stars} • {r['created_at'][:10]}")
                            if r['review']:
                                st.caption(r['review'])
                            st.markdown("---")

# ============================
# MAIN APP
# ============================
def main():
    st.title("Kathmandu Restaurant Recommender")
    st.caption("Find your perfect meal with smart, personalized recommendations.")

    meta = load_metadata()
    similarity = load_similarity()

    for k, v in {"logged_in": False, "username": None, "location": "", "preferences": ""}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Guest"])
            with tab1:
                with st.form("login_form"):
                    user = st.text_input("Username")
                    pw = st.text_input("Password", type="password")
                    submit = st.form_submit_button("Login")
                    if submit:
                        if verify_user(user, pw):
                            st.session_state.logged_in = True
                            st.session_state.username = user
                            u = get_user(user)
                            if u:
                                st.session_state.location = u.get("location", "") or ""
                                st.session_state.preferences = u.get("preferences", "") or ""
                            st.success("Welcome back!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
            with tab2:
                with st.form("signup_form"):
                    new_user = st.text_input("Username")
                    email = st.text_input("Email (optional)")
                    new_pw = st.text_input("Password", type="password")
                    submit = st.form_submit_button("Create Account")
                    if submit:
                        ok, msg = create_user(new_user, email, new_pw)
                        if ok:
                            st.success(msg + " Please login.")
                        else:
                            st.error(msg)
            with tab3:
                if st.button("Continue as Guest"):
                    st.session_state.logged_in = True
                    st.session_state.username = "guest"
                    st.rerun()
        return

    # SIDEBAR
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.write(f"📍 **{st.session_state.location or 'Not set'}**")
        prefs = st.session_state.preferences
        if prefs:
            pref_list = [c.strip() for c in prefs.split(",") if c.strip()]
            chips = " ".join([f"`{c}`" for c in pref_list])
            st.markdown(chips)
        st.markdown("---")
        page = st.radio("Menu", ["Home", "Explore", "Location", "Preferences", "Reviews", "Logout"])
        if page == "Logout":
            st.session_state.clear()
            st.rerun()

    if page == "Home":
        st.header("Recommended for You")
        recs = recommend_user(st.session_state.username, meta, similarity, st.session_state.location, st.session_state.preferences)
        if recs.empty:
            st.info("No recommendations yet. Try setting your preferences.")
        else:
            cols = st.columns(3)
            for i, row in recs.iterrows():
                with cols[i % 3]:
                    restaurant_card(row, f"home_{i}", meta, similarity)

    elif page == "Explore":
        st.header("Explore Restaurants")
        search = st.text_input("Search")
        cuisine_filter = st.multiselect("Cuisine", sorted(meta['cuisine'].unique()))
        loc_filter = st.selectbox("Location", ["Any"] + sorted(meta['location'].unique()))
        sort_by = st.selectbox("Sort by", ["Top rated", "Name (A-Z)", "Name (Z-A)"])

        df = meta.copy()
        if search:
            s = search.lower()
            mask = (
                df['name'].str.lower().str.contains(s, na=False)
                | df['cuisine'].str.lower().str.contains(s, na=False)
                | df['tags'].str.lower().str.contains(s, na=False)
            )
            df = df[mask]
        if cuisine_filter:
            df = df[df['cuisine'].isin(cuisine_filter)]
        if loc_filter != "Any":
            df = df[df['location'] == loc_filter]

        if sort_by == "Top rated" and 'rating' in df.columns:
            numeric_mask = pd.to_numeric(df['rating'], errors='coerce').notna()
            if numeric_mask.any():
                df = df[numeric_mask].sort_values('rating', ascending=False)
                df = pd.concat([df, df[~numeric_mask]])
            else:
                df = df.sort_values('name')
        elif sort_by == "Name (A-Z)":
            df = df.sort_values('name', ascending=True)
        elif sort_by == "Name (Z-A)":
            df = df.sort_values('name', ascending=False)

        total = len(df)
        per_page = 12
        pages = math.ceil(total / per_page) if total > 0 else 1
        if "explore_page" not in st.session_state:
            st.session_state.explore_page = 1
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            if st.button("Previous") and st.session_state.explore_page > 1:
                st.session_state.explore_page -= 1
                st.rerun()
        with col3:
            if st.button("Next") and st.session_state.explore_page < pages:
                st.session_state.explore_page += 1
                st.rerun()
        start = (st.session_state.explore_page - 1) * per_page
        page_df = df.iloc[start:start + per_page]
        st.write(f"Page {st.session_state.explore_page}/{pages} — {len(page_df)} shown")
        cols = st.columns(3)
        for i, row in page_df.iterrows():
            with cols[i % 3]:
                restaurant_card(row, f"exp_{i}", meta, similarity)

    elif page == "Location":
        st.header("Set Location")
        locs = [""] + sorted(meta['location'].dropna().unique().astype(str).tolist())
        current = st.session_state.location or ""
        choice = st.selectbox("Choose area", locs, index=locs.index(current) if current in locs else 0)
        if st.button("Save"):
            if choice:
                st.session_state.location = choice
                if st.session_state.username != "guest":
                    update_user_location(st.session_state.username, choice)
                st.success(f"Location: **{choice}**")
            else:
                st.warning("Select a location.")

    elif page == "Preferences":
        st.header("Preferences")
        current = st.session_state.preferences or ""
        default = [c.strip() for c in current.split(",") if c.strip()]
        chosen = st.multiselect("Favorite cuisines", sorted(meta['cuisine'].unique()), default=default)
        if st.button("Save"):
            prefs = ",".join(chosen)
            st.session_state.preferences = prefs
            if st.session_state.username != "guest":
                update_user_preferences(st.session_state.username, prefs)
            st.success("Saved!")

    elif page == "Reviews":
        st.header("My Reviews")
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql(
                "SELECT restaurant_id, rating, review, created_at FROM ratings WHERE username=? ORDER BY created_at DESC",
                conn, params=(st.session_state.username,)
            )
            conn.close()
            if df.empty:
                st.info("No reviews yet.")
            else:
                for _, r in df.iterrows():
                    name = meta[meta['restaurant_id'] == r['restaurant_id']]['name'].iloc[0] if not meta[meta['restaurant_id'] == r['restaurant_id']].empty else "Unknown"
                    stars = "⭐" * r['rating']
                    st.markdown(f"**{name}** — {stars} • {r['created_at'][:10]}")
                    if r['review']:
                        st.caption(r['review'])
                    st.markdown("---")
        except:
            st.info("No reviews.")

if __name__ == "__main__":
    main()
