"""
FINAL FYP APP — KRISH CHAKRADHAR (00020758)
Restaurant Recommender with Smart Matching + WORKING SIMILARITY
UI-upgraded version (dropdown-only location picker)
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

# ----------------------------
# Page config & paths
# ----------------------------
st.set_page_config(page_title="Kathmandu Restaurant Recommender", layout="wide", initial_sidebar_state="expanded")
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "restaurant_recommender.db"
MODEL_DIR = BASE_DIR / "recommender_model"
SIMILARITY_PKL = MODEL_DIR / "similarity_matrix.pkl"
RESTAURANT_META_CSV = MODEL_DIR / "restaurant_metadata.csv"

# ----------------------------
# Ensure DB/tables (safe)
# ----------------------------
def ensure_db():
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

ensure_db()

# ----------------------------
# Utilities & helpers
# ----------------------------
def hash_password(pw: str) -> str:
    return hashlib.sha256((pw or "").encode("utf-8")).hexdigest()

def safe_read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

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
            "rating": "rating", "Rating": "rating",
            "price": "price", "Price Range": "price",
            "tags": "tags"
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "restaurant_id" not in df.columns:
            df["restaurant_id"] = df.index.astype(str)
        if "name" not in df.columns:
            df["name"] = "Restaurant " + df["restaurant_id"].astype(str)
        if "cuisine" not in df.columns:
            df["cuisine"] = "Various"
        if "location" not in df.columns:
            df["location"] = "Unknown"
        if "price" not in df.columns:
            df["price"] = "Medium"
        if "tags" not in df.columns:
            df["tags"] = ""
        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.fillna("")
    df["restaurant_id"] = df["restaurant_id"].astype(str).str.strip()
    df["name"] = df["name"].astype(str)
    df["cuisine"] = df["cuisine"].astype(str)
    df["location"] = df["location"].astype(str)
    df["tags"] = df.get("tags", "").astype(str)
    return df.reset_index(drop=True)

@st.cache_data
def load_similarity():
    if not SIMILARITY_PKL.exists():
        st.warning("Similarity matrix not found. Similar restaurants disabled.")
        return None
    try:
        with open(SIMILARITY_PKL, "rb") as f:
            sim = pickle.load(f)
        return sim
    except Exception as e:
        st.error(f"Failed to load similarity matrix: {e}")
        return None

# ----------------------------
# DB functions (sqlite3)
# ----------------------------
def get_user(username):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT username, email, location, preferences FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()
        if row:
            return {"username": row[0], "email": row[1], "location": row[2], "preferences": row[3]}
        return None
    except Exception:
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
            return False, "Username already taken."
        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email or "", hash_password(password)))
        conn.commit()
        conn.close()
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, f"DB error: {e}"

def verify_user(username, password):
    if not username or not password:
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return False
        return row[0] == hash_password(password)
    except Exception:
        return False

def update_user_location(username, location):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE users SET location=? WHERE username=?", (location, username))
        conn.commit()
        conn.close()
    except Exception:
        pass

def update_user_preferences(username, prefs):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE users SET preferences=? WHERE username=?", (prefs, username))
        conn.commit()
        conn.close()
    except Exception:
        pass

def save_user_rating(username, restaurant_id, rating, review=""):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO ratings (username, restaurant_id, rating, review) VALUES (?, ?, ?, ?)",
                    (username, restaurant_id, int(rating), review or ""))
        conn.commit()
        conn.close()
    except Exception:
        pass

# ----------------------------
# FIXED: ROBUST get_similar()
# ----------------------------
def get_similar(restaurant_id, similarity, meta, top_n=6):
    if similarity is None or meta.empty:
        return pd.DataFrame()
    
    rid = str(restaurant_id).strip()
    try:
        # Ensure meta IDs are clean strings
        meta_ids = meta['restaurant_id'].astype(str).str.strip()
        if rid not in meta_ids.values:
            return pd.DataFrame()

        idx_in_meta = meta_ids[meta_ids == rid].index[0]
        
        if isinstance(similarity, pd.DataFrame):
            sim_idx = similarity.index.astype(str).str.strip()
            if rid not in sim_idx:
                return pd.DataFrame()
            sims = similarity.loc[rid].sort_values(ascending=False)
            sims = sims.drop(rid, errors='ignore')
            top_ids = sims.head(top_n).index.astype(str).tolist()
        else:
            scores = similarity[idx_in_meta]
            order = np.argsort(-scores)
            top_ids = []
            for i in order:
                cand_id = str(meta.iloc[i]['restaurant_id']).strip()
                if cand_id != rid and len(top_ids) < top_n:
                    top_ids.append(cand_id)
        
        result = meta[meta['restaurant_id'].astype(str).isin(top_ids)].copy()
        return result.head(top_n)
        
    except Exception as e:
        st.error(f"Similarity error: {str(e)}")
        return pd.DataFrame()

# ----------------------------
# SMART RECOMMENDATION (with synonym support)
# ----------------------------
def recommend_user(username, meta, similarity, location=None, prefs=None, top_n=12):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT restaurant_id, rating FROM ratings WHERE username=?", (username,))
        rows = cur.fetchall()
        conn.close()
    except Exception:
        rows = []

    rated = {r[0]: r[1] for r in rows} if rows else {}
    candidates = meta[~meta['restaurant_id'].isin(rated.keys())].copy() if rated else meta.copy()

    if location and location.strip():
        candidates = candidates[candidates['location'].astype(str).str.contains(location, case=False, na=False)]

    if prefs and prefs.strip():
        prefs_list = [p.strip().lower() for p in prefs.split(",") if p.strip()]
        if prefs_list:
            keyword_map = {
                "pizza": ["pizza", "pizzeria", "italian"],
                "burger": ["burger", "hamburger"],
                "momo": ["momo", "dumpling"],
                "coffee": ["coffee", "cafe"],
                "nepali": ["nepali", "newari", "thakali"],
            }
            mask = pd.Series(False, index=candidates.index)
            for pref in prefs_list:
                keywords = keyword_map.get(pref, [pref])
                for kw in keywords:
                    mask |= candidates['name'].str.lower().str.contains(kw, na=False)
                    mask |= candidates['cuisine'].str.lower().str.contains(kw, na=False)
                    if 'tags' in candidates.columns:
                        mask |= candidates['tags'].str.lower().str.contains(kw, na=False)
            candidates = candidates[mask]

    if 'rating' in candidates.columns and pd.api.types.is_numeric_dtype(candidates['rating']):
        candidates = candidates[candidates['rating'].notna()].sort_values('rating', ascending=False)
    else:
        candidates = candidates.sort_values('name')

    return candidates.head(top_n).reset_index(drop=True)

# ----------------------------
# UI helpers
# ----------------------------
_STYLES = """
<style>
.card {
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  background-color: #ffffff;
  margin-bottom: 16px;
  border: 1px solid #e0e0e0;
}
.badge {
  display:inline-block;
  padding: 4px 10px;
  margin: 2px 4px 2px 0;
  border-radius: 16px;
  font-size: 12px;
  background: #1A5F7A;
  color: white;
  font-weight: 500;
}
.header-left { display:flex; align-items:center; gap:12px; }
.small-muted { color: #6c757d; font-size:13px; }
</style>
"""
st.markdown(_STYLES, unsafe_allow_html=True)

def render_badges(tags_str):
    tags = [t.strip() for t in str(tags_str).split(",") if t.strip()]
    if not tags:
        return ""
    bads = " ".join([f"<span class='badge'>{t}</span>" for t in tags[:5]])
    return bads

def restaurant_card(row, key_prefix, meta, similarity):
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='margin:0; color:#1A5F7A'>{row['name']}</h4>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>{row.get('cuisine','')} • {row.get('location','')} • {row.get('price','N/A')}</div>", unsafe_allow_html=True)
        
        rating_val = row.get('rating', "")
        if pd.notna(rating_val) and rating_val != "":
            try:
                rv = float(rating_val)
                stars = "⭐" * int(round(rv))
                st.markdown(f"{stars} **{rv:.1f}**")
            except:
                st.markdown(f"⭐ **{rating_val}**")
        
        tags_html = render_badges(row.get('tags', ""))
        if tags_html:
            st.markdown(tags_html, unsafe_allow_html=True)
        
        with st.expander("Details & Actions"):
            col1, col2 = st.columns([2,1])
            with col1:
                desc = row.get('description', '') if 'description' in row else ''
                if desc:
                    st.caption(desc)
            with col2:
                if st.button("Show Similar", key=f"sim_{key_prefix}_{row['restaurant_id']}"):
                    sims = get_similar(row['restaurant_id'], similarity, meta)
                    if sims.empty:
                        st.info("No similar restaurants found.")
                    else:
                        st.markdown("**Similar:**")
                        for _, s in sims.iterrows():
                            st.markdown(f"• **{s['name']}** — {s.get('cuisine','')}")
                
                st.markdown("---")
                st.markdown("**Rate & Review**")
                rating = st.selectbox("Rating", [5,4,3,2,1], key=f"rate_{key_prefix}_{row['restaurant_id']}")
                review = st.text_area("Review (optional)", key=f"rev_{key_prefix}_{row['restaurant_id']}", height=70)
                if st.button("Submit Review", key=f"submit_{key_prefix}_{row['restaurant_id']}"):
                    save_user_rating(st.session_state.username, row['restaurant_id'], rating, review)
                    st.success("Thank you for your review!")

        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Main app
# ----------------------------
def main():
    try:
        meta = load_metadata()
        similarity = load_similarity()
    except Exception as e:
        st.error(f"Data load error: {e}")
        st.stop()

    for k, v in {"logged_in": False, "username": None, "location": "", "preferences": ""}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state.logged_in:
        st.markdown("<h1 style='text-align:center; color:#1A5F7A'>Kathmandu Restaurant Recommender</h1>", unsafe_allow_html=True)
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
                            st.success(f"Welcome back, {user}!")
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
                            st.success(msg)
                        else:
                            st.error(msg)
            with tab3:
                if st.button("Continue as Guest"):
                    st.session_state.logged_in = True
                    st.session_state.username = "guest"
                    st.rerun()
        return

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.write(f"📍 **{st.session_state.location or 'Not set'}**")
        st.write(f"❤️ **{st.session_state.preferences or 'Not set'}**")
        st.markdown("---")
        page = st.radio("Menu", ["Home", "Explore", "Location", "Preferences", "Reviews", "Logout"])
        if page == "Logout":
            st.session_state.clear()
            st.rerun()

    if page == "Home":
        st.header("Recommended for You")
        recs = recommend_user(st.session_state.username, meta, similarity, st.session_state.location, st.session_state.preferences, top_n=18)
        if recs.empty:
            st.info("No recommendations. Set location or preferences.")
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

        df = df.sort_values('rating', ascending=False, na_position='last')
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
        page_df = df.iloc[start:start+per_page]
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
                st.info("No reviews.")
            else:
                for _, r in df.iterrows():
                    name = meta[meta['restaurant_id'] == r['restaurant_id']]['name'].iloc[0] if not meta[meta['restaurant_id'] == r['restaurant_id']].empty else "Unknown"
                    st.markdown(f"**{name}** — {'⭐'*r['rating']} • {r['created_at'][:10]}")
                    if r['review']:
                        st.caption(r['review'])
                    st.markdown("---")
        except:
            st.info("No reviews.")

    # Footer
    st.markdown("""
    <div style='text-align:center; margin-top:50px; color:#7f8c8d; font-size:0.9rem;'>
        <strong>Nilai University</strong> — Final Year Project by <strong>Krish Chakradhar (00020758)</strong><br>
        EC3319 • Supervisor: Subarna Sapkota
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
