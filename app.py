"""
FINAL ROBUST VERSION — NO RATING ERROR
Handles string/float ratings, missing columns
Krish Chakradhar – 00020758 – EC3319
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import pickle
import math
from pathlib import Path

st.set_page_config(page_title="Kathmandu Restaurant Recommender", layout="wide")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "restaurant_recommender.db"
MODEL_DIR = BASE_DIR / "recommender_model"

SIMILARITY_PKL = MODEL_DIR / "similarity_matrix.pkl"
RESTAURANT_META_CSV = MODEL_DIR / "restaurant_metadata.csv"

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

def hash_password(pw): return hashlib.sha256(pw.encode()).hexdigest()

def safe_read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

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
            cuisine_match = candidates['cuisine'].astype(str).str.lower().apply(
                lambda x: any(p in x for p in prefs_list)
            )
            tags_match = pd.Series([False] * len(candidates), index=candidates.index)
            if 'tags' in candidates.columns:
                tags_match = candidates['tags'].astype(str).str.lower().apply(
                    lambda x: any(p in x for p in prefs_list) if x else False
                )
            candidates = candidates[cuisine_match | tags_match]

    if 'rating' in candidates.columns:
        if pd.api.types.is_numeric_dtype(candidates['rating']):
            candidates = candidates[candidates['rating'].notna()]
            candidates = candidates.sort_values('rating', ascending=False)
        else:
            candidates = candidates.sort_values('name')
    else:
        candidates = candidates.sort_values('name')

    return candidates.head(top_n).reset_index(drop=True)

def get_similar(restaurant_id, similarity, meta, top_n=6):
    if similarity is None or meta.empty:
        return pd.DataFrame()
    try:
        rid = str(restaurant_id)
        if isinstance(similarity, pd.DataFrame):
            if rid not in similarity.index:
                return pd.DataFrame()
            sims = similarity.loc[rid].sort_values(ascending=False)
            sims = sims.drop(rid, errors='ignore')
            top_ids = sims.head(top_n).index.astype(str).tolist()
        else:
            idx = meta[meta['restaurant_id'] == rid].index
            if idx.empty: return pd.DataFrame()
            i = idx[0]
            scores = similarity[i]
            order = np.argsort(-scores)
            ids = meta.iloc[order]['restaurant_id'].astype(str).tolist()
            top_ids = [x for x in ids if x != rid][:top_n]
        return meta[meta['restaurant_id'].isin(top_ids)].copy()
    except:
        return pd.DataFrame()

def card(row, key_prefix="", meta=None, similarity=None):
    with st.container():
        st.markdown(f"**{row['name']}**")
        st.write(f"*{row['cuisine']}* — {row['location']} — {row.get('price', 'N/A')}")

        rating_val = row.get('rating')
        if pd.notna(rating_val) and rating_val != "":
            try:
                rating_num = float(rating_val)
                st.write(f"Rating: {rating_num:.1f}")
            except (ValueError, TypeError):
                st.write(f"Rating: {rating_val}")

        tags = row.get('tags', '')
        if tags:
            st.caption(f"Tags: {tags}")

        with st.expander("Details & Actions"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Rate & Review**")
                rating = st.selectbox("Rating", [5,4,3,2,1], key=f"rate_{key_prefix}_{row['restaurant_id']}")
                review = st.text_area("Review", key=f"rev_{key_prefix}_{row['restaurant_id']}", height=70)
                if st.button("Submit", key=f"submit_{key_prefix}_{row['restaurant_id']}"):
                    save_user_rating(st.session_state.username, str(row['restaurant_id']), rating, review)
                    st.success("Saved!")
            with col2:
                if st.button("Show Similar", key=f"sim_{key_prefix}_{row['restaurant_id']}"):
                    sims = get_similar(row['restaurant_id'], similarity, meta)
                    if sims.empty:
                        st.info("No similar restaurants.")
                    else:
                        for _, s in sims.iterrows():
                            st.write(f"• {s['name']} ({s['cuisine']})")

def main():
    st.title("Kathmandu Restaurant Recommender")
    st.markdown("---")

    meta = load_metadata()
    similarity = load_similarity()

    for key in ["logged_in", "username", "location", "preferences"]:
        if key not in st.session_state:
            st.session_state[key] = False if key == "logged_in" else ("" if key in ["location", "preferences"] else None)

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
                                st.session_state.location = u["location"] or ""
                                st.session_state.preferences = u["preferences"] or ""
                            st.success("Logged in!")
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

    with st.sidebar:
        st.write(f"**{st.session_state.username}**")
        page = st.radio("Menu", ["Home", "Explore", "Location", "Preferences", "Reviews", "Logout"])
        if page == "Logout":
            st.session_state.clear()
            st.rerun()

    if page == "Home":
        st.header("Recommended for You")
        recs = recommend_user(
            st.session_state.username, meta, similarity,
            st.session_state.location, st.session_state.preferences
        )
        if recs.empty:
            st.info("No recommendations. Try setting location or preferences.")
        else:
            cols = st.columns(3)
            for i, row in recs.iterrows():
                with cols[i % 3]:
                    card(row, f"home_{i}", meta, similarity)

    elif page == "Explore":
        st.header("Explore Restaurants")
        search = st.text_input("Search")
        cuisine_filter = st.multiselect("Cuisine", sorted(meta['cuisine'].unique()))
        loc_filter = st.selectbox("Location", ["Any"] + sorted(meta['location'].unique()))

        df = meta.copy()
        if search:
            df = df[df.apply(lambda x: search.lower() in str(x).lower(), axis=1)]
        if cuisine_filter:
            df = df[df['cuisine'].isin(cuisine_filter)]
        if loc_filter != "Any":
            df = df[df['location'] == loc_filter]

        total = len(df)
        per_page = 12
        pages = math.ceil(total / per_page) if total > 0 else 1
        page_num = st.number_input("Page", 1, pages, 1)
        start = (page_num - 1) * per_page
        page_df = df.iloc[start:start + per_page]

        st.write(f"Showing {len(page_df)} of {total}")
        cols = st.columns(3)
        for i, row in page_df.iterrows():
            with cols[i % 3]:
                card(row, f"exp_{i}", meta, similarity)

    elif page == "Location":
        st.header("Set Your Location")
        locs = [""] + sorted(meta['location'].unique().tolist())
        current = st.session_state.location or ""
        choice = st.selectbox("Choose area", locs, index=locs.index(current) if current in locs else 0)
        if st.button("Save"):
            st.session_state.location = choice
            if st.session_state.username != "guest":
                update_user_location(st.session_state.username, choice)
            st.success("Location saved!")

    elif page == "Preferences":
        st.header("Your Preferences")
        current_prefs = st.session_state.preferences or ""
        default_prefs = [c.strip() for c in current_prefs.split(",") if c.strip()]
        cuisines = st.multiselect("Favorite cuisines", sorted(meta['cuisine'].unique()), default=default_prefs)
        if st.button("Save"):
            prefs = ",".join(cuisines)
            st.session_state.preferences = prefs
            if st.session_state.username != "guest":
                update_user_preferences(st.session_state.username, prefs)
            st.success("Preferences saved!")

    elif page == "Reviews":
        st.header("My Reviews")
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT restaurant_id, rating, review FROM ratings WHERE username=?", conn, params=(st.session_state.username,))
            conn.close()
            if df.empty:
                st.info("No reviews yet.")
            else:
                for _, r in df.iterrows():
                    name = meta[meta['restaurant_id'] == r['restaurant_id']]['name'].iloc[0] if not meta[meta['restaurant_id'] == r['restaurant_id']].empty else "Unknown"
                    st.write(f"**{name}** — {r['rating']} stars")
                    if r['review']:
                        st.caption(r['review'])
        except:
            st.info("No reviews.")

if __name__ == "__main__":
    main()
