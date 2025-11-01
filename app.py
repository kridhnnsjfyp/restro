"""
FINAL FYP APP — KRISH CHAKRADHAR (00020758)
Restaurant Recommender with Smart Matching + STUNNING UX
EC3319 — Nilai University
Supervisor: Subarna Sapkota
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import pickle
import math
from pathlib import Path

# ========================
# CONFIG & THEME
# ========================
st.set_page_config(
    page_title="Kathmandu Restaurant Recommender",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BEAUTIFUL MODERN CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        padding: 1rem;
    }
    .stApp {
        background: transparent;
    }
    .card {
        background: white;
        padding: 1.2rem;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        border: 1px solid #e0e6ed;
    }
    .card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    }
    .title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1A5F7A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #5d6d7e;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .badge {
        background: #1A5F7A;
        color: white;
        padding: 0.35rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.25rem;
        display: inline-block;
    }
    .chip {
        background: #3498db;
        color: white;
        padding: 0.4rem 0.9rem;
        border-radius: 25px;
        font-size: 0.85rem;
        margin: 0.25rem;
        display: inline-block;
        box-shadow: 0 2px 6px rgba(52,152,219,0.3);
    }
    .profile {
        background: linear-gradient(135deg, #1A5F7A, #2c8ca3);
        color: white;
        padding: 1.3rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 1.5rem;
        color: #7f8c8d;
        font-size: 0.9rem;
        border-top: 1px solid #e0e6ed;
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiselect > div > div {
        border-radius: 12px !important;
        border: 1px solid #d0d7de !important;
        padding: 0.6rem !important;
    }
    .stButton > button {
        border-radius: 12px !important;
        background: #1A5F7A !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        border: none !important;
    }
    .stButton > button:hover {
        background: #0f3d56 !important;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "restaurant_recommender.db"
MODEL_DIR = BASE_DIR / "recommender_model"

SIMILARITY_PKL = MODEL_DIR / "similarity_matrix.pkl"
RESTAURANT_META_CSV = MODEL_DIR / "restaurant_metadata.csv"

# ========================
# INIT DB
# ========================
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

# ========================
# UTILS
# ========================
def hash_password(pw): return hashlib.sha256(pw.encode()).hexdigest()

def safe_read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

# ========================
# DATA LOADERS
# ========================
@st.cache_data
def load_metadata():
    df = safe_read_csv(RESTAURANT_META_CSV)
    if df.empty:
        df = pd.DataFrame([{
            "restaurant_id": "r1", "name": "Sample Cafe", "cuisine": "Multi-Cuisine",
            "location": "Thamel", "rating": 4.0, "price": "Medium"
        }])
    else:
        rename_map = {
            "Restaurant Name": "name", "name": "name",
            "Cuisine Type": "cuisine", "cuisine": "cuisine",
            "Location": "location", "location": "location",
            "rating": "rating", "Rating": "rating",
            "price": "price", "Price Range": "price"
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
    df["restaurant_id"] = df["restaurant_id"].astype(str)
    return df.reset_index(drop=True)

@st.cache_data
def load_similarity():
    if not SIMILARITY_PKL.exists():
        return None
    try:
        with open(SIMILARITY_PKL, 'rb') as f:
            sim = pickle.load(f)
        return sim
    except:
        return None

# ========================
# DB OPERATIONS
# ========================
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
        return False, f"Error: {str(e)}"

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
                    (username, restaurant_id, rating, review))
        conn.commit()
        conn.close()
    except: pass

# ========================
# SMART RECOMMENDATION
# ========================
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
            keyword_map = {
                "pizza": ["pizza", "pizzeria", "pizzas", "pizza house", "pizza hut", "italian pizza"],
                "italian": ["italian", "pasta", "risotto", "lasagna", "gnocchi"],
                "burger": ["burger", "hamburger", "cheeseburger", "burger joint"],
                "coffee": ["coffee", "cafe", "espresso", "latte"],
                "momo": ["momo", "dumpling", "steam momo", "fried momo"],
                "nepali": ["nepali", "newari", "thakali", "dal bhat"],
            }

            match_mask = pd.Series([False] * len(candidates), index=candidates.index)

            for pref in prefs_list:
                keywords = keyword_map.get(pref, [pref])
                pattern = '|'.join([f"\\b{s}\\b" for s in keywords])
                name_match = candidates['name'].astype(str).str.lower().str.contains(pattern, regex=True, na=False)
                cuisine_match = candidates['cuisine'].astype(str).str.lower().str.contains(pattern, regex=True, na=False)
                tags_match = pd.Series([False] * len(candidates), index=candidates.index)
                if 'tags' in candidates.columns:
                    tags_match = candidates['tags'].astype(str).str.lower().str.contains(pattern, regex=True, na=False)
                match_mask |= name_match | cuisine_match | tags_match

            candidates = candidates[match_mask]

    if 'rating' in candidates.columns and pd.api.types.is_numeric_dtype(candidates['rating']):
        candidates = candidates[candidates['rating'].notna()]
        candidates = candidates.sort_values('rating', ascending=False)
    else:
        candidates = candidates.sort_values('name')

    return candidates.head

# ========================
# ENHANCED CARD
# ========================
def card(row, key_prefix="", meta=None, similarity=None):
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"<div class='card'><h3 style='margin:0; color:#1A5F7A'>{row['name']}</h3>", unsafe_allow_html=True)
            st.markdown(f"**{row['cuisine']}** — {row['location']} • {row.get('price', 'N/A')}")
            
            rating_val = row.get('rating')
            if pd.notna(rating_val) and rating_val != "":
                try:
                    rating_num = float(rating_val)
                    stars = "⭐" * int(round(rating_num))
                    st.write(f"{stars} **{rating_num:.1f}**")
                except:
                    st.write(f"Rating: {rating_val}")

            tags = row.get('tags', '')
            if tags:
                tag_list = [f"<span class='badge'>{t.strip()}</span>" for t in tags.split(",") if t.strip()]
                st.markdown("".join(tag_list), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            with st.expander("Actions"):
                rating = st.selectbox("Rate", [5,4,3,2,1], key=f"rate_{key_prefix}_{row['restaurant_id']}")
                review = st.text_area("Review", key=f"rev_{key_prefix}_{row['restaurant_id']}", height=60)
                if st.button("Submit", key=f"submit_{key_prefix}_{row['restaurant_id']}"):
                    save_user_rating(st.session_state.username, str(row['restaurant_id']), rating, review)
                    st.success("Saved!")
                if st.button("Similar", key=f"sim_{key_prefix}_{row['restaurant_id']}"):
                    sims = get_similar(row['restaurant_id'], similarity, meta)
                    if not sims.empty:
                        for _, s in sims.iterrows():
                            st.write(f"• {s['name']}")

# ========================
# MAIN APP
# ========================
def main():
    st.markdown("<h1 class='title'>Kathmandu Restaurant Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Find your perfect meal with AI-powered recommendations</p>", unsafe_allow_html=True)

    meta = load_metadata()
    similarity = load_similarity()

    for key in ["logged_in", "username", "location", "preferences"]:
        if key not in st.session_state:
            st.session_state[key] = False if key == "logged_in" else ("" if key in ["location", "preferences"] else None)

    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
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
                                st.session_state.location = u["location"] or ""
                                st.session_state.preferences = u["preferences"] or ""
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
            st.markdown("</div>", unsafe_allow_html=True)
        return

    # SIDEBAR
    with st.sidebar:
        st.markdown(f"<div class='profile'><h3>👤 {st.session_state.username}</h3>", unsafe_allow_html=True)
        if st.session_state.location:
            st.markdown(f"📍 **{st.session_state.location}**")
        if st.session_state.preferences:
            prefs = st.session_state.preferences.split(",")
            chips = "".join([f"<span class='chip'>{p.strip()}</span>" for p in prefs if p.strip()])
            st.markdown(chips, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        page = st.radio("Menu", [
            "Home", "Explore", "Location", 
            "Preferences", "Reviews", "Logout"
        ], format_func=lambda x: f"{x}")

        if page == "Logout":
            st.session_state.clear()
            st.rerun()

    # PAGES
    if page == "Home":
        st.markdown("### Recommended for You")
        with st.spinner("Finding best matches..."):
            recs = recommend_user(
                st.session_state.username, meta, similarity,
                st.session_state.location, st.session_state.preferences
            )
        if recs.empty:
            st.info("No recommendations. Try setting location or preferences.")
        else:
            cols = st.columns(2)
            for i, row in recs.iterrows():
                with cols[i % 2]:
                    card(row, f"home_{i}", meta, similarity)

    elif page == "Explore":
        st.markdown("### Explore Restaurants")
        search = st.text_input("Search by name, cuisine, or location")
        cuisine_filter = st.multiselect("Filter by Cuisine", sorted(meta['cuisine'].unique()))
        loc_filter = st.selectbox("Location", ["Any"] + sorted(meta['location'].unique()))
        sort_by = st.selectbox("Sort by", ["Rating", "Name"])

        df = meta.copy()
        if search:
            search_lower = search.lower()
            search_words = [w.strip() for w in search_lower.split() if w.strip()]
            mask = pd.Series([False] * len(df))
            for word in search_words:
                pattern = '|'.join([f"\\b{s}\\b" for s in [word]])
                mask |= (
                    df['name'].str.lower().str.contains(pattern, regex=True, na=False) |
                    df['cuisine'].str.lower().str.contains(pattern, regex=True, na=False)
                )
            df = df[mask]
        if cuisine_filter:
            df = df[df['cuisine'].isin(cuisine_filter)]
        if loc_filter != "Any":
            df = df[df['location'] == loc_filter]

        if sort_by == "Rating" and 'rating' in df.columns:
            df = df.sort_values('rating', ascending=False, na_position='last')
        elif sort_by == "Name":
            df = df.sort_values('name')

        total = len(df)
        per_page = 6
        pages = math.ceil(total / per_page) if total > 0 else 1
        col_prev, col_page, col_next = st.columns([1, 2, 1])
        page_num = col_page.number_input("Page", 1, pages, 1, step=1)
        if col_prev.button("Previous") and page_num > 1:
            page_num -= 1
            st.rerun()
        if col_next.button("Next") and page_num < pages:
            page_num += 1
            st.rerun()

        start = (page_num - 1) * per_page
        page_df = df.iloc[start:start + per_page]

        st.write(f"Showing {len(page_df)} of {total} restaurants")
        cols = st.columns(2)
        for i, row in page_df.iterrows():
            with cols[i % 2]:
                card(row, f"exp_{i}", meta, similarity)

    elif page == "Location":
        st.markdown("### Set Your Location")
        locs = [""] + sorted(meta['location'].unique().tolist())
        current = st.session_state.location or ""
        choice = st.selectbox("Choose your area", locs, index=locs.index(current) if current in locs else 0)
        if st.button("Save Location"):
            st.session_state.location = choice
            if st.session_state.username != "guest":
                update_user_location(st.session_state.username, choice)
            st.success(f"Location set to **{choice}**")

    elif page == "Preferences":
        st.markdown("### Your Food Preferences")
        current_prefs = st.session_state.preferences or ""
        default_prefs = [c.strip() for c in current_prefs.split(",") if c.strip()]
        cuisines = st.multiselect("What do you love to eat?", sorted(meta['cuisine'].unique()), default=default_prefs)
        if st.button("Save Preferences"):
            prefs = ",".join(cuisines)
            st.session_state.preferences = prefs
            if st.session_state.username != "guest":
                update_user_preferences(st.session_state.username, prefs)
            st.success("Preferences updated!")

    elif page == "Reviews":
        st.markdown("### My Reviews")
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT restaurant_id, rating, review, created_at FROM ratings WHERE username=? ORDER BY created_at DESC", conn, params=(st.session_state.username,))
            conn.close()
            if df.empty:
                st.info("You haven't reviewed any restaurants yet.")
            else:
                for _, r in df.iterrows():
                    name = meta[meta['restaurant_id'] == r['restaurant_id']]['name'].iloc[0] if not meta[meta['restaurant_id'] == r['restaurant_id']].empty else "Unknown"
                    stars = "⭐" * r['rating']
                    st.markdown(f"#### {name}")
                    st.write(f"{stars} • {r['created_at'][:10]}")
                    if r['review']:
                        st.caption(f"_{r['review']}_")
                    st.markdown("---")
        except:
            st.info("No reviews.")

    # Footer
    st.markdown("""
    <div class='footer'>
        <p><strong>Nilai University</strong> — Final Year Project by <strong>Krish Chakradhar (00020758)</strong></p>
        <p>EC3319 • Supervisor: Subarna Sapkota • Bachelor of Information Technology (Hons)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
