

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import hashlib
import requests
import os
from datetime import datetime

# ========================================
# CONFIG
# ========================================
st.set_page_config(page_title="Foodmandu Recommender", layout="wide", initial_sidebar_state="expanded")

# Professional Styling
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #ff6b35; color: white; border-radius: 8px; font-weight: bold;}
    .stButton>button:hover {background-color: #e55a2b;}
    .css-1d391kg {padding: 1rem 1.5rem;}
    .restaurant-card {padding: 1rem; border: 1px solid #ddd; border-radius: 10px; margin: 0.5rem 0; background: white;}
    .title {font-size: 2.5rem; font-weight: 700; color: #1a1a1a; text-align: center; margin-bottom: 0.5rem;}
    .subtitle {text-align: center; color: #666; margin-bottom: 2rem;}
    .nilai-logo {display: block; margin: 0 auto 1rem; width: 150px;}
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "recommender_model"
DB_PATH = "restaurant_recommender.db"

# ========================================
# LOAD MODEL
# ========================================
@st.cache_resource
def load_model():
    sim_path = f"{MODEL_DIR}/similarity_matrix.pkl"
    meta_path = f"{MODEL_DIR}/restaurant_metadata.csv"
    
    if not os.path.exists(sim_path) or not os.path.exists(meta_path):
        st.error("Model files missing. Upload `recommender_model/` folder.")
        st.stop()
    
    with open(sim_path, 'rb') as f:
        sim_df = pickle.load(f)
    meta = pd.read_csv(meta_path)
    meta['Location'] = meta['Location'].str.title()
    meta['Cuisine Type'] = meta['Cuisine Type'].str.title()
    return sim_df, meta

similarity_df, rest_metadata = load_model()

# ========================================
# DATABASE
# ========================================
@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT,
        email TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS preferences (
        user_id INTEGER PRIMARY KEY,
        pref_location TEXT,
        pref_cuisine TEXT,
        last_location TEXT,
        search_count INTEGER DEFAULT 0
    )
    ''')

    # Fix missing columns
    try: cur.execute("ALTER TABLE preferences ADD COLUMN last_location TEXT")
    except: pass
    try: cur.execute("ALTER TABLE preferences ADD COLUMN search_count INTEGER DEFAULT 0")
    except: pass

    cur.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        restaurant TEXT,
        action TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    return conn

conn = get_db()
cur = conn.cursor()

# ========================================
# DB HELPERS
# ========================================
def ensure_preference_row(user_id):
    try:
        cur.execute("INSERT OR IGNORE INTO preferences (user_id, search_count) VALUES (?, 0)", (user_id,))
        conn.commit()
    except: pass

def get_preference(user_id, field):
    ensure_preference_row(user_id)
    try:
        cur.execute(f"SELECT {field} FROM preferences WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else None
    except: return None

def increment_search_count(user_id):
    try:
        cur.execute("UPDATE preferences SET search_count = search_count + 1 WHERE user_id=?", (user_id,))
        conn.commit()
    except: pass

def set_preference(user_id, **kwargs):
    ensure_preference_row(user_id)
    try:
        updates = ", ".join([f"{k}=?" for k in kwargs])
        values = list(kwargs.values()) + [user_id]
        cur.execute(f"UPDATE preferences SET {updates} WHERE user_id=?", values)
        conn.commit()
    except Exception as e:
        st.error(f"DB Error: {e}")

# ========================================
# AUTH
# ========================================
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def register(u, p, e): 
    try:
        cur.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                    (u, hash_pw(p), e))
        conn.commit()
        return True
    except: return False

def login(u, p):
    try:
        cur.execute("SELECT id FROM users WHERE username=? AND password_hash=?", (u, hash_pw(p)))
        row = cur.fetchone()
        return row[0] if row else None
    except: return None

# ========================================
# LOCATIONS
# ========================================
all_locations = sorted(rest_metadata['Location'].unique().tolist())

# ========================================
# RECOMMEND
# ========================================
def recommend_by_location(location, cuisine=None, top_n=5):
    loc_clean = location.lower().strip()
    mask = rest_metadata['Location'].str.lower().str.contains(loc_clean, na=False)
    candidates = rest_metadata[mask]
    
    if cuisine and cuisine != "Any":
        candidates = candidates[candidates['Cuisine Type'].str.contains(cuisine, case=False, na=False)]
    
    if candidates.empty:
        return pd.DataFrame()
    
    scores = candidates['Cuisine Type'].apply(lambda x: 2.0 if cuisine and cuisine.lower() in x.lower() else 1.0)
    candidates = candidates.copy()
    candidates['Score'] = scores
    return candidates.sort_values('Score', ascending=False).head(top_n)

# ========================================
# SIDEBAR
# ========================================
def sidebar_profile():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/5e/Nilai_University_Logo.png", width=120)
        st.markdown(f"### Hi, **{st.session_state.username}**")
        st.markdown("---")
        
        if st.button("Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        
        st.markdown("---")
        st.markdown("#### Stats")
        
        uid = st.session_state.user_id
        try:
            cur.execute("SELECT COUNT(*) FROM interactions WHERE user_id=?", (uid,))
            interactions = cur.fetchone()[0]
            search_count = get_preference(uid, 'search_count') or 0
        except:
            interactions = search_count = 0
        
        last_loc = get_preference(uid, 'last_location') or "Not set"
        
        st.write(f"**Searches:** {search_count}")
        st.write(f"**Interactions:** {interactions}")
        st.write(f"**Last Area:** {last_loc}")

# ========================================
# LOGIN PAGE
# ========================================
def page_login():
    st.markdown('<div class="title">Foodmandu Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Find the best restaurants in your area</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                uid = login(username, password)
                if uid:
                    st.session_state.user_id = uid
                    st.session_state.username = username
                    ensure_preference_row(uid)
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        st.markdown(f"<small>Don't have an account? <a href='#' onclick='document.getElementById(\"register-tab\").click()'>Register now</a></small>", unsafe_allow_html=True)

    with col2:
        st.markdown("### Register")
        with st.form("register_form"):
            reg_u = st.text_input("Username", key="reg_u")
            reg_p = st.text_input("Password", type="password", key="reg_p")
            reg_e = st.text_input("Email", key="reg_e")
            reg_submit = st.form_submit_button("Create Account")
            if reg_submit:
                if register(reg_u, reg_p, reg_e):
                    st.success("Account created! Please login.")
                else:
                    st.error("Username already taken")

# ========================================
# MAIN PAGE
# ========================================
def page_main():
    st.markdown('<div class="title">Find Restaurants</div>', unsafe_allow_html=True)
    uid = st.session_state.user_id

    last_loc = get_preference(uid, 'last_location')
    default_loc = last_loc if last_loc and last_loc in all_locations else all_locations[0]

    col1, col2 = st.columns([3, 1])
    with col1:
        location = st.selectbox(
            "Select your area:",
            options=all_locations,
            index=all_locations.index(default_loc) if default_loc in all_locations else 0
        )
    with col2:
        cuisine = st.selectbox("Cuisine?", 
                               options=["Any"] + sorted(rest_metadata['Cuisine Type'].unique().tolist()))

    if st.button("Search Restaurants", type="primary", use_container_width=True):
        increment_search_count(uid)
        set_preference(uid, last_location=location)
        
        cuisine_filter = cuisine if cuisine != "Any" else None
        recs = recommend_by_location(location, cuisine_filter)

        if recs.empty:
            st.warning(f"No restaurants found in **{location}**")
        else:
            st.success(f"Top {len(recs)} in **{location}**")
            for _, row in recs.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="restaurant-card">
                        <h4>{row['Restaurant Name']}</h4>
                        <p><strong>Cuisine:</strong> {row['Cuisine Type']}<br>
                           <strong>Location:</strong> {row['Location']}<br>
                           <strong>Avg Price:</strong> Rs. {row.get('avg_price', 'N/A'):.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    col_a, col_b = st.columns([3, 1])
                    with col_b:
                        if st.button("Select", key=row['Restaurant Name']):
                            cur.execute("INSERT INTO interactions (user_id, restaurant, action) VALUES (?, ?, ?)",
                                        (uid, row['Restaurant Name'], 'select'))
                            conn.commit()
                            st.success(f"Selected **{row['Restaurant Name']}**!")

# ========================================
# DASHBOARD
# ========================================
def page_dashboard():
    st.markdown('<div class="title">Your Dashboard</div>', unsafe_allow_html=True)
    uid = st.session_state.user_id

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Profile")
        cur.execute("SELECT username, email, created_at FROM users WHERE id=?", (uid,))
        user = cur.fetchone()
        st.write(f"**Name:** {user[0]}")
        st.write(f"**Email:** {user[1]}")
        st.write(f"**Member Since:** {user[2][:10]}")

        st.subheader("Preferences")
        pref_loc = get_preference(uid, 'pref_location') or 'Not set'
        pref_cuisine = get_preference(uid, 'pref_cuisine') or 'Not set'
        last_loc = get_preference(uid, 'last_location') or 'None'
        search_count = get_preference(uid, 'search_count') or 0
        st.write(f"**Favorite Area:** {pref_loc}")
        st.write(f"**Favorite Cuisine:** {pref_cuisine}")
        st.write(f"**Last Search:** {last_loc}")
        st.write(f"**Total Searches:** {search_count}")

    with col2:
        st.subheader("Activity Log")
        cur.execute("""
            SELECT restaurant, action, timestamp 
            FROM interactions 
            WHERE user_id=? 
            ORDER BY timestamp DESC 
            LIMIT 10
        """, (uid,))
        logs = cur.fetchall()
        
        if logs:
            log_df = pd.DataFrame(logs, columns=["Restaurant", "Action", "Time"])
            log_df['Time'] = pd.to_datetime(log_df['Time']).dt.strftime('%b %d, %H:%M')
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info("No activity yet.")

# ========================================
# MAIN
# ========================================
if 'user_id' not in st.session_state:
    page_login()
else:
    sidebar_profile()
    
    tab1, tab2 = st.tabs(["Search", "Dashboard"])
    
    with tab1:
        page_main()
    with tab2:
        page_dashboard()
