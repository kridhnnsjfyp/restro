"""
FINAL FYP APP — KRISH CHAKRADHAR (00020758)
Restaurant Recommender — GREEN SUCCESS + RED ERROR
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
# LOAD METADATA
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
# SIMILARITY (NO SKLEARN)
# ============================
@st.cache_data
def load_or_create_similarity(meta):
    if SIMILARITY_PKL.exists():
        try:
            with open(SIMILARITY_PKL, "rb") as f:
                sim = pickle.load(f)
            return sim
        except:
            pass

    st.info("Generating similarity...")
    n = len(meta)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        row_i = meta.iloc[i]
        tags_i = set([t.strip().lower() for t in str(row_i['tags']).split(",") if t.strip()])
        cuisine_i = str(row_i['cuisine']).lower()
        for j in range(n):
            if i == j: continue
            row_j = meta.iloc[j]
            tags_j = set([t.strip().lower() for t in str(row_j['tags']).split(",") if t.strip()])
            cuisine_j = str(row_j['cuisine']).lower()
            tag_overlap = len(tags_i & tags_j)
            cuisine_match = 1 if cuisine_i == cuisine_j else 0
            sim_matrix[i, j] = tag_overlap * 2 + cuisine_match

    MODEL_DIR.mkdir(exist_ok=True)
    with open(SIMILARITY_PKL, "wb") as f:
        pickle.dump(sim_matrix, f)
    st.success("Similarity generated!")
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
        return dict(zip(["username", "email", "location", "preferences"], row)) if row else None
    except: return None

def create_user(username, email, password):
    if not username or not password:
        return False, "Username and password required."
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
        if cur.fetchone():
            conn.close()
            return False, "Username already taken."
        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email or "", hash_password(password)))
        conn.commit()
        conn.close()
        return True, "Account successfully created! Please login."
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
# SIMILAR & RECOMMEND
# ============================
def get_similar(restaurant_id, similarity, meta, top_n=6):
    if similarity is None or meta.empty:
        return pd.DataFrame()
    rid = str(restaurant_id).strip()
    try:
        idx = meta[meta['restaurant_id'].astype(str).str.strip() == rid].index[0]
        scores = similarity[idx]
        order = np.argsort(-scores)
        top_ids = [str(meta.iloc[i]['restaurant_id']) for i in order if str(meta.iloc[i]['restaurant_id']) != rid][:top_n]
        return meta[meta['restaurant_id'].astype(str).isin(top_ids)].head(top_n)
    except:
        return pd.DataFrame()

def recommend_user(username, meta, similarity, location=None, prefs=None, top_n=12):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT restaurant_id FROM ratings WHERE username=?", (username,))
        rated = {r[0] for r in cur.fetchall()}
        conn.close()
    except: rated = set()
    candidates = meta[~meta['restaurant_id'].isin(rated)].copy() if rated else meta.copy()

    if location and location.strip():
        candidates = candidates[candidates['location'].astype(str).str.contains(location, case=False, na=False)]
    if prefs and prefs.strip():
        prefs_list = [p.strip().lower() for p in prefs.split(",")]
        mask = pd.Series(False, index=candidates.index)
        for p in prefs_list:
            mask |= candidates['name'].str.lower().str.contains(p, na=False)
            mask |= candidates['cuisine'].str.lower().str.contains(p, na=False)
            if 'tags' in candidates.columns:
                mask |= candidates['tags'].str.lower().str.contains(p, na=False)
        candidates = candidates[mask]

    if 'rating' in candidates.columns:
        candidates = candidates.sort_values('rating', ascending=False, na_position='last')
    return candidates.head(top_n).reset_index(drop=True)

# ============================
# UI CARD
# ============================
def restaurant_card(row, key_prefix, meta, similarity):
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='margin:0; color:#1A5F7A'>{row['name']}</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#6c757d; font-size:13px'>{row.get('cuisine','')} • {row.get('location','')} • {row.get('price','N/A')}</div>", unsafe_allow_html=True)
        
        if pd.notna(row.get('rating')):
            try:
                rv = float(row['rating'])
                stars = "⭐" * int(round(rv))
                st.markdown(f"{stars} **{rv:.1f}**")
            except:
                st.markdown(f"⭐ {row['rating']}")
        
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
# MAIN APP — FIXED SIGNUP UX
# ============================
def main():
    st.markdown("<h1 style='text-align:center; color:#1A5F7A'>Kathmandu Restaurant Recommender</h1>", unsafe_allow_html=True)
    meta = load_metadata()
    similarity = load_or_create_similarity(meta)

    for k, v in {"logged_in": False, "username": None, "location": "", "preferences": ""}.items():
        if k not in st.session_state:
            st.session_state[k] = v

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
                                st.session_state.location = u.get("location", "") or ""
                                st.session_state.preferences = u.get("preferences", "") or ""
                            st.success("Welcome back!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
            with tab2:
                with st.form("signup"):
                    new_user = st.text_input("Username")
                    email = st.text_input("Email (optional)")
                    new_pw = st.text_input("Password", type="password")
                    if st.form_submit_button("Create Account"):
                        ok, msg = create_user(new_user, email, new_pw)
                        if ok:
                            st.success(msg)  # GREEN
                        else:
                            st.error(msg)    # RED
            with tab3:
                if st.button("Continue as Guest"):
                    st.session_state.logged_in = True
                    st.session_state.username = "guest"
                    st.rerun()
        return

    # [SIDEBAR & PAGES — SAME AS BEFORE]

    st.markdown("""
    <div style='text-align:center; margin-top:50px; color:#7f8c8d; font-size:0.9rem;'>
        <strong>Nilai University</strong> — FYP by <strong>Krish Chakradhar (00020758)</strong><br>
        EC3319 • Supervisor: Subarna Sapkota
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
