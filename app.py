"""
FINAL FYP APP — KRISH CHAKRADHAR (00020758)
Restaurant Recommender — AUTO SIMILARITY + NO ERRORS
EC3319 — Nilai University | Supervisor: Subarna Sapkota
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import pickle
import math
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
def hash_password(pw): return hashlib.sha256(pw.encode()).hexdigest()

def safe_read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

# ============================
# LOAD METADATA — SAFE
# ============================
@st.cache_data
def load_metadata():
    df = safe_read_csv(RESTAURANT_META_CSV)
    if df.empty:
        df = pd.DataFrame([{
            "restaurant_id": "r1", "name": "Sample Cafe", "cuisine": "Multi-Cuisine",
            "location": "Thamel", "rating": 4.0, "price": "Medium", "tags": ""
        }])
    else:
        rename_map = {
            "Restaurant Name": "name", "restaurant_name": "name", "name": "name",
            "Cuisine Type": "cuisine", "cuisine": "cuisine",
            "Location": "location", "location": "location",
            "rating": "rating", "Rating": "rating", "Ratings": "rating", "Average Rating": "rating",
            "price": "price", "Price Range": "price",
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

# ============================
# LOAD OR GENERATE SIMILARITY — FIXED
# ============================
@st.cache_data
def load_or_create_similarity(meta):
    if SIMILARITY_PKL.exists():
        try:
            with open(SIMILARITY_PKL, "rb") as f:
                sim = pickle.load(f)
            st.success("Similarity matrix loaded.")
            return sim
        except Exception as e:
            st.warning(f"Corrupted similarity file. Regenerating... ({e})")

    # AUTO-GENERATE SIMILARITY
    st.info("Generating similarity matrix from cuisine & tags...")
    meta["features"] = meta["cuisine"].astype(str) + " " + meta["tags"].astype(str)
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    try:
        tfidf_matrix = vectorizer.fit_transform(meta["features"])
        sim_matrix = cosine_similarity(tfidf_matrix)
        # Save for next time
        MODEL_DIR.mkdir(exist_ok=True)
        with open(SIMILARITY_PKL, "wb") as f:
            pickle.dump(sim_matrix, f)
        st.success("Similarity matrix generated and saved!")
        return sim_matrix
    except Exception as e:
        st.error(f"Failed to generate similarity: {e}")
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
    except: return None

def create_user(username, email, password):
    if not username or not password: return False, "Required fields missing."
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
        if cur.fetchone(): return False, "Username taken."
        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email or "", hash_password(password)))
        conn.commit()
        conn.close()
        return True, "Account created."
    except: return False, "Error creating user."

def verify_user(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()
        return row and row[0] == hash_password(password)
    except: return False

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

# ============================
# RECOMMENDATION & SIMILARITY
# ============================
def recommend_user(username, meta, similarity, location=None, prefs=None, top_n=12):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT restaurant_id, rating FROM ratings WHERE username=?", (username,))
        rows = cur.fetchall()
        conn.close()
    except: rows = []
    rated = {r[0]: r[1] for r in rows}
    candidates = meta[~meta['restaurant_id'].isin(rated.keys())].copy() if rated else meta.copy()

    if location and location.strip():
        candidates = candidates[candidates['location'].astype(str).str.contains(location, case=False, na=False)]
    if prefs and prefs.strip():
        prefs_list = [p.strip().lower() for p in prefs.split(",") if p.strip()]
        mask = pd.Series(False, index=candidates.index)
        for pref in prefs_list:
            mask |= candidates['name'].str.lower().str.contains(pref, na=False)
            mask |= candidates['cuisine'].str.lower().str.contains(pref, na=False)
            if 'tags' in candidates.columns:
                mask |= candidates['tags'].str.lower().str.contains(pref, na=False)
        candidates = candidates[mask]

    if 'rating' in candidates.columns and pd.to_numeric(candidates['rating'], errors='coerce').notna().any():
        candidates = candidates.sort_values('rating', ascending=False, na_position='last')
    else:
        candidates = candidates.sort_values('name')
    return candidates.head(top_n).reset_index(drop=True)

def get_similar(restaurant_id, similarity, meta, top_n=6):
    if similarity is None or meta.empty:
        return pd.DataFrame()
    rid = str(restaurant_id).strip()
    try:
        meta_ids = meta['restaurant_id'].astype(str).str.strip()
        if rid not in meta_ids.values:
            return pd.DataFrame()
        idx = meta_ids[meta_ids == rid].index[0]
        scores = similarity[idx]
        order = np.argsort(-scores)
        top_ids = [str(meta.iloc[i]['restaurant_id']) for i in order if str(meta.iloc[i]['restaurant_id']) != rid][:top_n]
        return meta[meta['restaurant_id'].astype(str).isin(top_ids)].head(top_n)
    except Exception as e:
        st.error(f"Similarity error: {e}")
        return pd.DataFrame()

# ============================
# UI CARD
# ============================
def restaurant_card(row, key_prefix, meta, similarity):
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='margin:0; color:#1A5F7A'>{row['name']}</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#6c757d; font-size:13px'>{row.get('cuisine','')} • {row.get('location','')} • {row.get('price','N/A')}</div>", unsafe_allow_html=True)
        
        rating_val = row.get('rating')
        if pd.notna(rating_val):
            try:
                rv = float(rating_val)
                stars = "⭐" * int(round(rv))
                st.markdown(f"{stars} **{rv:.1f}**")
            except:
                st.markdown(f"⭐ {rating_val}")
        
        tags = row.get('tags', '')
        if tags:
            tag_list = [f"<span class='badge'>{t.strip()}</span>" for t in tags.split(",") if t.strip()]
            st.markdown("".join(tag_list), unsafe_allow_html=True)
        
        with st.expander("Details & Actions"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Rate & Review**")
                rating = st.selectbox("Rating", [5,4,3,2,1], key=f"rate_{key_prefix}_{row['restaurant_id']}")
                review = st.text_area("Review", key=f"rev_{key_prefix}_{row['restaurant_id']}", height=70)
                if st.button("Submit", key=f"submit_{key_prefix}_{row['restaurant_id']}"):
                    save_user_rating(st.session_state.username, str(row['restaurant_id']), rating, review)
                    st.success("Thank you!")
            with col2:
                if st.button("Show Similar", key=f"sim_{key_prefix}_{row['restaurant_id']}"):
                    sims = get_similar(row['restaurant_id'], similarity, meta)
                    if sims.empty:
                        st.info("No similar restaurants.")
                    else:
                        st.markdown("**Similar:**")
                        for _, s in sims.iterrows():
                            st.markdown(f"• **{s['name']}** — {s.get('cuisine','')}")

        st.markdown("</div>", unsafe_allow_html=True)

# ============================
# MAIN APP
# ============================
def main():
    st.markdown("<h1 style='text-align:center; color:#1A5F7A'>Kathmandu Restaurant Recommender</h1>", unsafe_allow_html=True)
    meta = load_metadata()
    similarity = load_or_create_similarity(meta)  # AUTO-FIX

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
                    if st.form_submit_button("Login"):
                        if verify_user(user, pw):
                            st.session_state.logged_in = True
                            st.session_state.username = user
                            u = get_user(user)
                            if u:
                                st.session_state.location = u.get("location", "") or ""
                                st.session_state.preferences = u.get("preferences", "") or ""
                            st.success("Welcome!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
            with tab2:
                with st.form("signup_form"):
                    new_user = st.text_input("Username")
                    email = st.text_input("Email (optional)")
                    new_pw = st.text_input("Password", type="password")
                    if st.form_submit_button("Create Account"):
                        ok, msg = create_user(new_user, email, new_pw)
                        st.write("Success" if ok else "Error", msg)
            with tab3:
                if st.button("Continue as Guest"):
                    st.session_state.logged_in = True
                    st.session_state.username = "guest"
                    st.rerun()
        return

    with st.sidebar:
        st.markdown(f"### {st.session_state.username}")
        st.write(f"Location: **{st.session_state.location or 'Not set'}**")
        st.write(f"Preferences: **{st.session_state.preferences or 'Not set'}**")
        st.markdown("---")
        page = st.radio("Menu", ["Home", "Explore", "Location", "Preferences", "Reviews", "Logout"])
        if page == "Logout":
            st.session_state.clear()
            st.rerun()

    if page == "Home":
        st.header("Recommended for You")
        recs = recommend_user(st.session_state.username, meta, similarity, st.session_state.location, st.session_state.preferences)
        if recs.empty:
            st.info("Set location or preferences.")
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
            mask = (df['name'].str.lower().str.contains(s, na=False) |
                    df['cuisine'].str.lower().str.contains(s, na=False) |
                    df['tags'].str.lower().str.contains(s, na=False))
            df = df[mask]
        if cuisine_filter:
            df = df[df['cuisine'].isin(cuisine_filter)]
        if loc_filter != "Any":
            df = df[df['location'] == loc_filter]

        if sort_by == "Top rated" and 'rating' in df.columns:
            df = df.sort_values('rating', ascending=False, na_position='last')
        elif sort_by == "Name (A-Z)":
            df = df.sort_values('name')
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
            df = pd.read_sql("SELECT restaurant_id, rating, review, created_at FROM ratings WHERE username=? ORDER BY created_at DESC", conn, params=(st.session_state.username,))
            conn.close()
            if df.empty:
                st.info("No reviews yet.")
            else:
                for _, r in df.iterrows():
                    name = meta[meta['restaurant_id'] == r['restaurant_id']]['name'].iloc[0] if not meta[meta['restaurant_id'] == r['restaurant_id']].empty else "Unknown"
                    st.markdown(f"**{name}** — {'⭐'*r['rating']} • {r['created_at'][:10]}")
                    if r['review']:
                        st.caption(r['review'])
                    st.markdown("---")
        except:
            st.info("No reviews.")

    st.markdown("""
    <div style='text-align:center; margin-top:50px; color:#7f8c8d; font-size:0.9rem;'>
        <strong>Nilai University</strong> — Final Year Project by <strong>Krish Chakradhar (00020758)</strong><br>
        EC3319 • Supervisor: Subarna Sapkota
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
