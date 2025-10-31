
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

MODEL_DIR = "recommender_model"
DB_PATH = "restaurant_recommender.db"
CSV_PATH = "foodmandu_data_clean.csv"

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
# DATABASE (SAFE)
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
        last_location TEXT
    )
    ''')

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
# SAFE DB HELPERS
# ========================================
def ensure_preference_row(user_id):
    cur.execute("INSERT OR IGNORE INTO preferences (user_id) VALUES (?)", (user_id,))
    conn.commit()

def get_preference(user_id, field):
    ensure_preference_row(user_id)
    cur.execute(f"SELECT {field} FROM preferences WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    return row[0] if row and row[0] else None

def set_preference(user_id, **kwargs):
    ensure_preference_row(user_id)
    updates = ", ".join([f"{k}=?" for k in kwargs])
    values = list(kwargs.values()) + [user_id]
    cur.execute(f"UPDATE preferences SET {updates} WHERE user_id=?", values)
    conn.commit()

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
    cur.execute("SELECT id FROM users WHERE username=? AND password_hash=?", (u, hash_pw(p)))
    row = cur.fetchone()
    return row[0] if row else None

# ========================================
# LOCATION
# ========================================
def get_user_city():
    try:
        resp = requests.get('https://ipapi.co/json/', timeout=3)
        city = resp.json().get('city', 'Kathmandu')
        return city.title() if city else 'Kathmandu'
    except:
        return 'Kathmandu'

# ========================================
# RECOMMEND BY LOCATION
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
# SIDEBAR: PROFILE + LOGOUT
# ========================================
def sidebar_profile():
    with st.sidebar:
        st.markdown(f"### Hi, **{st.session_state.username}**")
        st.markdown("---")
        
        if st.button("Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        
        st.markdown("---")
        st.markdown("#### Quick Stats")
        
        uid = st.session_state.user_id
        cur.execute("SELECT COUNT(*) FROM interactions WHERE user_id=?", (uid,))
        interactions = cur.fetchone()[0]
        
        last_loc = get_preference(uid, 'last_location') or "Not set"
        
        st.write(f"**Searches:** {interactions}")
        st.write(f"**Last Area:** {last_loc}")

# ========================================
# PAGE: LOGIN
# ========================================
def page_login():
    st.title("Foodmandu Recommender")
    st.markdown("### Find the Best Restaurants in Your Area")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Login")
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                uid = login(u, p)
                if uid:
                    st.session_state.user_id = uid
                    st.session_state.username = u
                    ensure_preference_row(uid)  # Create row
                    st.rerun()
                else:
                    st.error("Invalid login")

    with col2:
        st.subheader("Register")
        with st.form("register"):
            u = st.text_input("Username", key="reg_u")
            p = st.text_input("Password", type="password", key="reg_p")
            e = st.text_input("Email", key="reg_e")
            if st.form_submit_button("Register"):
                if register(u, p, e):
                    st.success("Registered! Login now.")
                else:
                    st.error("Username taken")

# ========================================
# PAGE: DASHBOARD
# ========================================
def page_dashboard():
    st.title("Your Dashboard")
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
        st.write(f"**Favorite Area:** {pref_loc}")
        st.write(f"**Favorite Cuisine:** {pref_cuisine}")
        st.write(f"**Last Search:** {last_loc}")

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
            st.info("No activity yet. Start searching!")

# ========================================
# PAGE: MAIN (LOCATION FIRST)
# ========================================
def page_main():
    st.title("Find Restaurants")
    uid = st.session_state.user_id

    default_loc = get_preference(uid, 'last_location') or get_user_city()

    col1, col2 = st.columns([3, 1])
    with col1:
        location = st.text_input("Where are you?", value=default_loc, help="e.g., Putalisadak, Thamel")
    with col2:
        use_current = st.checkbox("Use my current location", value=True)

    if use_current:
        location = get_user_city()
        st.info(f"Detected: **{location}**")

    cuisine = st.selectbox("Preferred cuisine?", 
                           options=["Any"] + sorted(rest_metadata['Cuisine Type'].unique().tolist()))

    if st.button("Search Restaurants", type="primary"):
        set_preference(uid, last_location=location)
        
        cuisine_filter = cuisine if cuisine != "Any" else None
        recs = recommend_by_location(location, cuisine_filter)

        if recs.empty:
            st.warning(f"No restaurants found in **{location}**")
        else:
            st.success(f"Top {len(recs)} in **{location}**")
            for _, row in recs.iterrows():
                with st.expander(f"{row['Restaurant Name']} • {row['Cuisine Type']}"):
                    st.write(f"**Location:** {row['Location']}")
                    if st.button(f"Select This", key=row['Restaurant Name']):
                        cur.execute("INSERT INTO interactions (user_id, restaurant, action) VALUES (?, ?, ?)",
                                    (uid, row['Restaurant Name'], 'select'))
                        conn.commit()
                        st.success(f"Selected **{row['Restaurant Name']}**!")

# ========================================
# MAIN APP
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
