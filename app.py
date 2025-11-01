# restro/app.py
"""
FINAL FYP APP — KRISH CHAKRADHAR (00020758)
Restaurant Recommender with Smart Matching
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
        # gentle rename mapping for common variants
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
    df["restaurant_id"] = df["restaurant_id"].astype(str)
    df["name"] = df["name"].astype(str)
    df["cuisine"] = df["cuisine"].astype(str)
    df["location"] = df["location"].astype(str)
    df["tags"] = df.get("tags", "").astype(str)
    return df.reset_index(drop=True)

@st.cache_data
def load_similarity():
    if not SIMILARITY_PKL.exists():
        return None
    try:
        with open(SIMILARITY_PKL, "rb") as f:
            sim = pickle.load(f)
        return sim
    except Exception:
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
# Recommender helpers (kept robust)
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
            # simple keyword match across name, cuisine, tags
            mask = pd.Series(False, index=candidates.index)
            for pref in prefs_list:
                mask |= candidates['name'].str.lower().str.contains(pref, na=False)
                mask |= candidates['cuisine'].str.lower().str.contains(pref, na=False)
                if 'tags' in candidates.columns:
                    mask |= candidates['tags'].str.lower().str.contains(pref, na=False)
            candidates = candidates[mask]

    if 'rating' in candidates.columns and pd.api.types.is_numeric_dtype(candidates['rating']):
        candidates = candidates[candidates['rating'].notna()].sort_values('rating', ascending=False)
    else:
        candidates = candidates.sort_values('name')

    return candidates.head(top_n).reset_index(drop=True)

def get_similar(restaurant_id, similarity, meta, top_n=6):
    if similarity is None or meta.empty:
        return pd.DataFrame()
    try:
        rid = str(restaurant_id)
        if isinstance(similarity, pd.DataFrame):
            if rid not in similarity.index.astype(str).tolist():
                return pd.DataFrame()
            sims = similarity.loc[rid].sort_values(ascending=False)
            sims = sims.drop(rid, errors='ignore')
            top_ids = sims.head(top_n).index.astype(str).tolist()
        else:
            idx = meta[meta['restaurant_id'] == rid].index
            if idx.empty:
                return pd.DataFrame()
            i = idx[0]
            vec = np.array(similarity[i])
            order = np.argsort(-vec)
            ids = meta.iloc[order]['restaurant_id'].astype(str).tolist()
            top_ids = [x for x in ids if x != rid][:top_n]
        return meta[meta['restaurant_id'].isin(top_ids)].copy()
    except Exception:
        return pd.DataFrame()

# ----------------------------
# UI helpers (styling and card)
# ----------------------------
_STYLES = """
<style>
.card {
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  background-color: #ffffff;
  margin-bottom: 12px;
}
.badge {
  display:inline-block;
  padding: 3px 8px;
  margin-right:6px;
  border-radius:12px;
  font-size:12px;
  background:#f1f1f1;
}
.header-left {
  display:flex; align-items:center; gap:12px;
}
.small-muted { color: #6c757d; font-size:13px; }
</style>
"""
st.markdown(_STYLES, unsafe_allow_html=True)

def render_badges(tags_str):
    tags = [t.strip() for t in str(tags_str).split(",") if t.strip()]
    if not tags:
        return ""
    bads = " ".join([f"<span class='badge'>{t}</span>" for t in tags[:6]])
    return bads

def restaurant_card(row, key_prefix, meta, similarity):
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**{row['name']}**")
        st.markdown(f"<div class='small-muted'>{row.get('cuisine','')} • {row.get('location','')} • {row.get('price','N/A')}</div>", unsafe_allow_html=True)
        rating_val = row.get('rating', "")
        if pd.notna(rating_val) and rating_val != "":
            try:
                rv = float(rating_val)
                st.markdown(f"⭐ **{rv:.1f}**")
            except Exception:
                st.markdown(f"⭐ **{rating_val}**")
        tags_html = render_badges(row.get('tags', ""))
        if tags_html:
            st.markdown(tags_html, unsafe_allow_html=True)
        with st.expander("Details & Actions"):
            col1, col2 = st.columns([2,1])
            with col1:
                desc = row.get('description', '') if 'description' in row else ''
                if desc:
                    st.write(desc)
            with col2:
                if st.button("Show similar", key=f"sim_{key_prefix}_{row['restaurant_id']}"):
                    sims = get_similar(row['restaurant_id'], similarity, meta)
                    if sims.empty:
                        st.info("No similar restaurants available.")
                    else:
                        for _, s in sims.iterrows():
                            st.write(f"• {s['name']} — {s.get('cuisine','')}")
                st.write("---")
                st.write("Rate & review")
                rating = st.selectbox("Rating", [5,4,3,2,1], key=f"rate_{key_prefix}_{row['restaurant_id']}")
                review = st.text_area("Review", key=f"rev_{key_prefix}_{row['restaurant_id']}", height=80)
                if st.button("Submit", key=f"submit_{key_prefix}_{row['restaurant_id']}"):
                    save_user_rating(st.session_state.username, row['restaurant_id'], rating, review)
                    st.success("Saved — thank you!")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Main app
# ----------------------------
def main():
    try:
        meta = load_metadata()
        similarity = load_similarity()
    except Exception:
        st.error("Could not load data. Check files in recommender_model/.")
        st.stop()

    # initialize session state
    for k, v in {"logged_in": False, "username": None, "location": "", "preferences": ""}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Centered auth when not logged in
    if not st.session_state.logged_in:
        st.markdown("<div class='header-left'><h2>🍽 Kathmandu Restaurant Recommender</h2></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            tab1, tab2, tab3 = st.tabs(["🔐 Login", "🆕 Sign Up", "👀 Guest"])
            with tab1:
                with st.form("login_form"):
                    user = st.text_input("Username", key="ui_login_user")
                    pw = st.text_input("Password", type="password", key="ui_login_pw")
                    submit = st.form_submit_button("Login")
                    if submit:
                        with st.spinner("Signing in..."):
                            if verify_user(user, pw):
                                st.session_state.logged_in = True
                                st.session_state.username = user
                                u = get_user(user)
                                if u:
                                    st.session_state.location = u.get("location", "") or ""
                                    st.session_state.preferences = u.get("preferences", "") or ""
                                st.success(f"Welcome back, {user}!")
                                st.experimental_rerun()
                            else:
                                st.error("Invalid credentials.")
            with tab2:
                with st.form("signup_form"):
                    new_user = st.text_input("Choose username", key="ui_signup_user")
                    email = st.text_input("Email (optional)", key="ui_signup_email")
                    new_pw = st.text_input("Choose password", type="password", key="ui_signup_pw")
                    submit = st.form_submit_button("Create Account")
                    if submit:
                        with st.spinner("Creating account..."):
                            ok, msg = create_user(new_user, email, new_pw)
                            if ok:
                                st.success(msg)
                                st.balloons()
                                # clear
                                st.session_state.ui_signup_user = ""
                                st.session_state.ui_signup_email = ""
                                st.session_state.ui_signup_pw = ""
                            else:
                                st.error(msg)
            with tab3:
                if st.button("Continue as Guest"):
                    st.session_state.logged_in = True
                    st.session_state.username = "guest"
                    st.experimental_rerun()
        return  # do not continue until logged in

    # Sidebar: profile + quick actions
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.write(f"📍 Location: **{st.session_state.location or 'Not set'}**")
        prefs_display = st.session_state.preferences or "Not set"
        st.write(f"❤️ Preferences: **{prefs_display}**")
        st.markdown("---")
        page = st.radio("Menu", ["Home", "Explore", "Location", "Preferences", "Reviews", "Logout"])
        st.markdown("---")
        st.markdown("🔧 Quick actions")
        if st.button("Set location now"):
            page = "Location"  # local variable only, actual page selection done below via assignment
        st.write("")
        st.caption("Tip: set your location for nearby recommendations.")
        if page == "Logout":
            st.session_state.clear()
            st.experimental_rerun()

    # Use page variable from sidebar (re-read to avoid local override)
    page = st.session_state.get("_sidebar_page", None) or page  # fallback in case of internal change

    # HOME: recommended card grid
    if page == "Home":
        st.header("Recommended for you")
        user_loc = st.session_state.location or ""
        prefs = st.session_state.preferences or ""
        with st.spinner("Generating recommendations..."):
            recs = recommend_user(st.session_state.username, meta, similarity, location=user_loc, prefs=prefs, top_n=18)
        if recs is None or recs.empty:
            st.info("No recommendations found — set your location or preferences and try again.")
        else:
            cols = st.columns(3)
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 3]:
                    restaurant_card(row, f"home_{i}", meta, similarity)

    # EXPLORE: search + filters + sort + pagination (Next/Prev)
    elif page == "Explore":
        st.header("Explore Restaurants")
        # search suggestions: top cuisines & locations for hint
        sample_cuisines = ", ".join(sorted(set(meta['cuisine'].dropna().astype(str).unique()))[:6])
        sample_locs = ", ".join(sorted(set(meta['location'].dropna().astype(str).unique()))[:6])
        st.caption(f"Try cuisines: {sample_cuisines}  •  areas: {sample_locs}")

        col_a, col_b, col_c = st.columns([3,2,1])
        with col_a:
            search = st.text_input("Search (name, cuisine, tag or location)", key="explore_search")
        with col_b:
            cuisine_filter = st.multiselect("Cuisine", sorted(meta['cuisine'].dropna().unique().astype(str).tolist()), key="explore_cuisine")
        with col_c:
            locs = ["Any"] + sorted(meta['location'].dropna().unique().astype(str).tolist())
            loc_filter = st.selectbox("Location", locs, index=0, key="explore_loc")

        df = meta.copy()
        if search:
            s = search.strip().lower()
            mask = df['name'].str.lower().str.contains(s, na=False) | df['cuisine'].str.lower().str.contains(s, na=False) | df['tags'].str.lower().str.contains(s, na=False) | df['location'].str.lower().str.contains(s, na=False)
            df = df[mask]
        if cuisine_filter:
            df = df[df['cuisine'].isin(cuisine_filter)]
        if loc_filter and loc_filter != "Any":
            df = df[df['location'] == loc_filter]

        sort_by = st.selectbox("Sort by", ["Top rated", "Name (A-Z)", "Name (Z-A)"], key="explore_sort")
        if sort_by == "Top rated" and 'rating' in df.columns:
            df = df[df['rating'].notna()].sort_values('rating', ascending=False)
        elif sort_by == "Name (A-Z)":
            df = df.sort_values('name', ascending=True)
        elif sort_by == "Name (Z-A)":
            df = df.sort_values('name', ascending=False)

        # pagination (Next / Prev)
        per_page = 12
        total = len(df)
        pages = max(1, math.ceil(total / per_page))
        if "explore_page" not in st.session_state:
            st.session_state.explore_page = 1
        colp1, colp2, colp3 = st.columns([1,2,1])
        with colp1:
            if st.button("Previous") and st.session_state.explore_page > 1:
                st.session_state.explore_page -= 1
        with colp3:
            if st.button("Next") and st.session_state.explore_page < pages:
                st.session_state.explore_page += 1
        start = (st.session_state.explore_page - 1) * per_page
        page_df = df.iloc[start:start+per_page]
        st.write(f"Showing {len(page_df)} of {total} restaurants — Page {st.session_state.explore_page}/{pages}")
        cols = st.columns(3)
        for i, (_, row) in enumerate(page_df.iterrows()):
            with cols[i % 3]:
                restaurant_card(row, f"explore_{i}", meta, similarity)

    # LOCATION: dropdown-only selection from real dataset
    elif page == "Location":
        st.header("Set Your Location")
        locs = [""] + sorted(meta['location'].dropna().unique().astype(str).tolist())
        current = st.session_state.location or ""
        try:
            index = locs.index(current) if current in locs else 0
        except ValueError:
            index = 0
        choice = st.selectbox("Choose area", locs, index=index)
        if st.button("Save Location"):
            if not choice:
                st.warning("Please select a location from the dropdown.")
            else:
                st.session_state.location = choice
                if st.session_state.username != "guest":
                    update_user_location(st.session_state.username, choice)
                st.success(f"📍 Location set to **{choice}**")
                st.info("Recommendations will prioritize restaurants in this area.")
    # PREFERENCES page
    elif page == "Preferences":
        st.header("Your Preferences")
        current_prefs = st.session_state.preferences or ""
        default_prefs = [c.strip() for c in current_prefs.split(",") if c.strip()]
        cuisines = sorted(meta['cuisine'].dropna().unique().astype(str).tolist())
        chosen = st.multiselect("Favorite cuisines", cuisines, default=default_prefs)
        if st.button("Save Preferences"):
            prefs = ",".join(chosen)
            st.session_state.preferences = prefs
            if st.session_state.username != "guest":
                update_user_preferences(st.session_state.username, prefs)
            st.success("Preferences saved.")
    # REVIEWS page
    elif page == "Reviews":
        st.header("My Reviews")
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT restaurant_id, rating, review, created_at FROM ratings WHERE username=?", conn, params=(st.session_state.username,))
            conn.close()
            if df.empty:
                st.info("You haven't submitted any reviews yet.")
            else:
                for _, r in df.sort_values("created_at", ascending=False).iterrows():
                    name = meta[meta['restaurant_id'] == r['restaurant_id']]['name'].iloc[0] if not meta[meta['restaurant_id'] == r['restaurant_id']].empty else "Unknown"
                    st.markdown(f"**{name}** — {int(r['rating'])} ⭐")
                    if r['review']:
                        st.caption(r['review'])
                    st.write("---")
        except Exception:
            st.info("No reviews found (or a DB error occurred).")

# -- run
if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("An unexpected error occurred. Check logs for details.")
        st.text(traceback.format_exc().splitlines()[-1])
