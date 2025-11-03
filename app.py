

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
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            location TEXT,
            preferences TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            restaurant_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            review TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
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
# LOAD METADATA
# ============================
@st.cache_data
def load_metadata():
    df = safe_read_csv(RESTAURANT_META_CSV)
    if df.empty:
        df = pd.DataFrame([{
            "restaurant_id": "r1", "name": "Sample Cafe", "cuisine": "Multi-Cuisine",
            "location": "Thamel", "rating": 4.0, "price": "Medium", "tags": "cozy, coffee, wifi"
        }])
    else:
        rename_map = {
            "Restaurant Name": "name", "restaurant_name": "name", "name": "name",
            "Cuisine Type": "cuisine", "cuisine": "cuisine",
            "Location": "location", "location": "location",
            "rating": "rating", "Rating": "rating", "Average Rating": "rating",
            "price": "price", "Price Range": "price",
            "tags": "tags"
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "restaurant_id" not in df.columns:
            df["restaurant_id"] = df.index.astype(str)
        for col, default in [("name","Restaurant"),("cuisine","Various"),("location","Kathmandu"),("price","Medium"),("tags","")]:
            if col not in df.columns:
                df[col] = default
        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors='coerce')
    df = df.fillna("")
    df["restaurant_id"] = df["restaurant_id"].astype(str).str.strip()
    return df.reset_index(drop=True)

# ============================
# FIXED WEIGHTED SIMILARITY
# ============================
@st.cache_data
def load_or_create_similarity(meta):
    if SIMILARITY_PKL.exists():
        try:
            with open(SIMILARITY_PKL, "rb") as f:
                sim = pickle.load(f)
            if sim.shape[0] == len(meta):
                return sim
        except:
            pass

    with st.spinner("Generating weighted similarity matrix..."):
        n = len(meta)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            row_i = meta.iloc[i]
            tags_i = set([t.strip().lower() for t in str(row_i['tags']).split(",") if t.strip()])
            cuisine_i, loc_i, price_i = map(str.lower, [row_i['cuisine'], row_i['location'], row_i['price']])

            for j in range(n):
                if i == j: 
                    continue
                row_j = meta.iloc[j]
                tags_j = set([t.strip().lower() for t in str(row_j['tags']).split(",") if t.strip()])
                cuisine_j, loc_j, price_j = map(str.lower, [row_j['cuisine'], row_j['location'], row_j['price']])

                tag_overlap = len(tags_i & tags_j)
                cuisine_match = 1 if cuisine_i == cuisine_j else 0
                loc_match = 1 if loc_i == loc_j else 0
                price_match = 1 if price_i == price_j else 0
                tag_penalty = 1 / (1 + abs(len(tags_i) - len(tags_j)))

                # Weighted score
                score = ((tag_overlap * 3) * tag_penalty) + (cuisine_match * 4) + (loc_match * 2) + (price_match * 1)
                sim_matrix[i, j] = score

        max_val = sim_matrix.max()
        if max_val > 0:
            sim_matrix = (sim_matrix / max_val) * 100

        MODEL_DIR.mkdir(exist_ok=True)
        with open(SIMILARITY_PKL, "wb") as f:
            pickle.dump(sim_matrix, f)

    st.success("Similarity matrix ready!")
    return sim_matrix

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
        return dict(zip(["username","email","location","preferences"], row)) if row else None
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
            return False, "Username already taken."
        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email or "", hash_password(password)))
        conn.commit()
        conn.close()
        return True, "Account created. Please login."
    except:
        return False, "Error creating user."

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
    except: pass

def update_user_preferences(username, prefs):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE users SET preferences=? WHERE username=?", (prefs, username))
        conn.commit()
        conn.close()
    except: pass

def save_user_rating(username, restaurant_id, rating, review=""):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO ratings (username, restaurant_id, rating, review) VALUES (?, ?, ?, ?)",
                    (username, restaurant_id, int(rating), review or ""))
        conn.commit()
        conn.close()
    except: pass

def get_reviews(restaurant_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT username, rating, review, created_at FROM ratings WHERE restaurant_id=? ORDER BY created_at DESC", conn, params=(restaurant_id,))
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
        cur.execute("SELECT restaurant_id FROM ratings WHERE username=?", (username,))
        rated = {r[0] for r in cur.fetchall()}
        conn.close()
    except:
        rated = set()

    candidates = meta[~meta['restaurant_id'].isin(rated)].copy() if rated else meta.copy()
    if location:
        candidates = candidates[candidates['location'].astype(str).str.contains(location, case=False, na=False)]
    if prefs:
        prefs_list = [p.strip().lower() for p in prefs.split(",")]
        mask = pd.Series(False, index=candidates.index)
        for p in prefs_list:
            mask |= candidates['name'].str.lower().str.contains(p, na=False)
            mask |= candidates['cuisine'].str.lower().str.contains(p, na=False)
            mask |= candidates['tags'].str.lower().str.contains(p, na=False)
        candidates = candidates[mask]
    if 'rating' in candidates.columns:
        candidates = candidates.sort_values('rating', ascending=False, na_position='last')
    return candidates.head(top_n).reset_index(drop=True)

# ============================
# RESTAURANT CARD UI
# ============================
def restaurant_card(row, key_prefix, meta):
    st.markdown(f"<h4 style='margin:0; color:#1A5F7A'>{row['name']}</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#6c757d; font-size:13px'>{row['cuisine']} • {row['location']} • {row['price']}</div>", unsafe_allow_html=True)

    rating_val = row.get("rating", "")
    if pd.notna(rating_val) and rating_val != "":
        try:
            rv = float(rating_val)
            st.markdown("⭐" * int(round(rv)))
        except:
            pass

    if row.get("tags"):
        tags = " ".join([f"<span style='background:#e9ecef;padding:2px 6px;border-radius:4px;font-size:0.8em;margin:2px'>{t.strip()}</span>" for t in row['tags'].split(",")])
        st.markdown(tags, unsafe_allow_html=True)

    with st.expander("Details & Reviews"):
        st.markdown("**Rate this restaurant**")
        star = st.slider("Rating", 1, 5, 5, key=f"star_{key_prefix}_{row['restaurant_id']}")
        review = st.text_area("Write a review", key=f"rev_{key_prefix}_{row['restaurant_id']}")
        if st.button("Submit Review", key=f"btn_{key_prefix}_{row['restaurant_id']}"):
            save_user_rating(st.session_state.username, row['restaurant_id'], star, review)
            st.success("Review submitted!")

        revs = get_reviews(row['restaurant_id'])
        if not revs.empty:
            with st.expander("Show all reviews"):
                for _, r in revs.iterrows():
                    st.markdown(f"**{r['username']}** — {'⭐'*r['rating']} • {r['created_at'][:10]}")
                    if r['review']:
                        st.caption(r['review'])
                    st.markdown("---")
        else:
            st.caption("No reviews yet.")

# ============================
# MAIN APP
# ============================
def main():
    meta = load_metadata()
    similarity = load_or_create_similarity(meta)

    for k, v in {"logged_in": False, "username": None, "location": "", "preferences": ""}.items():
        st.session_state.setdefault(k, v)

    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Guest"])
            with tab1:
                with st.form("login"):
                    user = st.text_input("Username")
                    pw = st.text_input("Password", type="password")
                    if st.form_submit_button("Login"):
                        if verify_user(user, pw):
                            st.session_state.logged_in = True
                            st.session_state.username = user
                            u = get_user(user)
                            if u:
                                st.session_state.location = u.get("location", "")
                                st.session_state.preferences = u.get("preferences", "")
                            st.success("Welcome back!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
            with tab2:
                with st.form("signup"):
                    u = st.text_input("Username")
                    e = st.text_input("Email (optional)")
                    p = st.text_input("Password", type="password")
                    if st.form_submit_button("Create Account"):
                        ok, msg = create_user(u, e, p)
                        st.success(msg) if ok else st.error(msg)
            with tab3:
                if st.button("Continue as Guest"):
                    st.session_state.logged_in = True
                    st.session_state.username = "guest"
                    st.rerun()
        return

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.write(f"📍 {st.session_state.location or 'Any'}")
        st.write(f"❤️ {st.session_state.preferences or 'None'}")
        page = st.radio("Menu", ["Home", "Explore", "Location", "Preferences", "Reviews", "Logout"])
        if page == "Logout":
            st.session_state.clear()
            st.rerun()

    if page == "Home":
        st.header("Recommended for You")
        recs = recommend_user(st.session_state.username, meta, similarity, st.session_state.location, st.session_state.preferences)
        if recs.empty:
            st.info("No recommendations yet.")
        else:
            cols = st.columns(3)
            for i, row in recs.iterrows():
                with cols[i % 3]:
                    restaurant_card(row, f"home_{i}", meta)

    elif page == "Explore":
        st.header("Explore Restaurants")
        search = st.text_input("Search")
        cuisine_filter = st.multiselect("Cuisine", sorted(meta['cuisine'].unique()))
        loc_filter = st.selectbox("Location", ["Any"] + sorted(meta['location'].unique()))
        df = meta.copy()
        if search:
            s = search.lower()
            df = df[df['name'].str.lower().str.contains(s, na=False) | df['cuisine'].str.lower().str.contains(s, na=False)]
        if cuisine_filter:
            df = df[df['cuisine'].isin(cuisine_filter)]
        if loc_filter != "Any":
            df = df[df['location'] == loc_filter]
        df = df.sort_values('rating', ascending=False, na_position='last')
        total, per_page = len(df), 12
        st.write(f"{total} restaurants found")
        cols = st.columns(3)
        for i, row in df.head(per_page).iterrows():
            with cols[i % 3]:
                restaurant_card(row, f"exp_{i}", meta)

    elif page == "Location":
        st.header("Set Location")
        new_loc = st.text_input("Your location", st.session_state.location)
        if st.button("Save"):
            st.session_state.location = new_loc
            update_user_location(st.session_state.username, new_loc)
            st.success("Updated!")

    elif page == "Preferences":
        st.header("Set Preferences")
        new_pref = st.text_area("Favorite cuisines or vibes (comma separated)", st.session_state.preferences)
        if st.button("Save"):
            st.session_state.preferences = new_pref
            update_user_preferences(st.session_state.username, new_pref)
            st.success("Preferences saved!")

    elif page == "Reviews":
        st.header("Your Reviews")
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT restaurant_id, rating, review, created_at FROM ratings WHERE username=?", conn, params=(st.session_state.username,))
        conn.close()
        if df.empty:
            st.info("No reviews yet.")
        else:
            for _, r in df.iterrows():
                name = meta.loc[meta['restaurant_id'] == r['restaurant_id'], 'name'].iloc[0] if not meta.loc[meta['restaurant_id'] == r['restaurant_id']].empty else "Unknown"
                st.markdown(f"**{name}** — {'⭐'*r['rating']} • {r['created_at'][:10]}")
                if r['review']:
                    st.caption(r['review'])
                st.markdown("---")

if __name__ == "__main__":
    main()
