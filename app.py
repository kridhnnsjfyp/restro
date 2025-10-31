# app.py
# Restaurant Recommender – FYP EC3319
# Krish Chakradhar – 00020758
# FINAL: NO READONLY ERROR + REGISTER WORKS

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import hashlib
import os
from datetime import datetime

# ========================================
# CONFIG
# ========================================
st.set_page_config(page_title="Foodmandu Recommender", layout="centered", initial_sidebar_state="expanded")

# PROFESSIONAL STYLING
st.markdown("""
<style>
    .main {background-color: #f8f9fa; padding: 2rem;}
    .login-container {max-width: 420px; margin: 2rem auto; padding: 2.5rem; background: white; border-radius: 16px; box-shadow: 0 12px 35px rgba(0,0,0,0.1);}
    .title {font-size: 2.8rem; font-weight: 800; color: white; text-align: center; margin-bottom: 0.5rem; text-shadow: 0 2px 4px rgba(0,0,0,0.3);}
    .subtitle {text-align: center; color: #ddd; font-size: 1.1rem; margin-bottom: 2.5rem;}
    .stButton>button {background: #007bff; color: white; border-radius: 10px; font-weight: 600; padding: 0.7rem; width: 100%; border: none;}
    .stButton>button:hover {background: #0056b3;}
    .card {background: white; border-radius: 12px; padding: 1.2rem; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
    .tag {background: #007bff; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; display: inline-block; margin: 0.2rem;}
    .register-link {text-align: center; margin-top: 1rem;}
    .stButton > button[kind="secondary"] {background: #6c757d; color: white;}
    .stButton > button[kind="secondary"]:hover {background: #5a6268;}
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "recommender_model"
DB_PATH = "/tmp/restaurant_recommender.db"  # ALWAYS WRITABLE

# ========================================
# ENSURE DB EXISTS IN /tmp/
# ========================================
def init_db():
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

# Initialize DB
conn = init_db()
cur = conn.cursor()

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
    except: pass

# ========================================
# AUTH
# ========================================
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def register(u, p, e): 
    try:
        u = u.strip()
        if not u or not p or not e:
            return False
        cur.execute("SELECT id FROM users WHERE username=?", (u,))
        if cur.fetchone():
            return False
        cur.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                    (u, hash_pw(p), e))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return False

def login(u, p):
    try:
        cur.execute("SELECT id FROM users WHERE username=? AND password_hash=?", (u, hash_pw(p)))
        row = cur.fetchone()
        return row[0] if row and row[0] else None
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return None

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
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/5e/Nilai_University_Logo.png", width=100)
        st.markdown(f"### Hi, **{st.session_state.username}**")
        st.markdown("---")
        
        if st.button("Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        
        st.markdown("---")
        st.markdown("#### Your Stats")
        
        uid = st.session_state.user_id
        try:
            cur.execute("SELECT COUNT(*) FROM interactions WHERE user_id=?", (uid,))
            interactions = cur.fetchone()[0]
            search_count = get_preference(uid, 'search_count') or 0
        except:
            interactions = search_count = 0
        
        last_loc = get_preference(uid, 'last_location') or "Not set"
        
        st.write(f"**Searches:** `{search_count}`")
        st.write(f"**Selections:** `{interactions}`")
        st.write(f"**Last Area:** `{last_loc}`")

# ========================================
# AUTH PAGE – REGISTER ON BUTTON
# ========================================
def page_auth():
    st.markdown('<div class="title">Foodmandu Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Find the best restaurants near you</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)

        if 'show_register' not in st.session_state:
            st.session_state.show_register = False

        if st.session_state.show_register:
            st.markdown("### Create Account")
            with st.form("register_form"):
                reg_u = st.text_input("Username", placeholder="Choose username", key="reg_u")
                reg_p = st.text_input("Password", type="password", placeholder="Create password", key="reg_p")
                reg_e = st.text_input("Email", placeholder="Your email", key="reg_e")
                
                col1, col2 = st.columns(2)
                with col1:
                    reg_submit = st.form_submit_button("Create Account")
                with col2:
                    back = st.form_submit_button("Back to Login")

                if back:
                    st.session_state.show_register = False
                    st.rerun()

                if reg_submit:
                    if register(reg_u, reg_p, reg_e):
                        st.success("Account created! Please login.")
                        st.session_state.show_register = False
                        st.rerun()
                    else:
                        st.error("Username taken or invalid input.")

        else:
            st.markdown("### Login")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit = st.form_submit_button("Login", use_container_width=True)
                if submit:
                    uid = login(username, password)
                    if uid:
                        st.session_state.user_id = uid
                        st.session_state.username = username
                        ensure_preference_row(uid)
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

            if st.button("Register now", use_container_width=True, type="secondary"):
                st.session_state.show_register = True
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# ========================================
# MAIN PAGE
# ========================================
def page_main():
    st.markdown('<div class="title">Find Restaurants</div>', unsafe_allow_html=True)
    uid = st.session_state.user_id

    last_loc = get_preference(uid, 'last_location')
    default_loc = last_loc if last_loc and last_loc in all_locations else all_locations[0]

    col1, col2 = st.columns([2, 1])
    with col1:
        location = st.selectbox("Select your area:", options=all_locations, index=all_locations.index(default_loc))
    with col2:
        cuisine = st.selectbox("Cuisine?", options=["Any"] + sorted(rest_metadata['Cuisine Type'].unique().tolist()))

    if st.button("Search Restaurants", use_container_width=True):
        increment_search_count(uid)
        set_preference(uid, last_location=location)
        
        cuisine_filter = cuisine if cuisine != "Any" else None
        recs = recommend_by_location(location, cuisine_filter)

        if recs.empty:
            st.warning(f"No restaurants found in **{location}**")
        else:
            st.success(f"Top {len(recs)} in **{location}**")
            for _, row in recs.iterrows():
                st.markdown(f"""
                <div class="card">
                    <h4 style="margin:0; color:#007bff;">{row['Restaurant Name']}</h4>
                    <p style="margin:0.3rem 0;">
                        <span class="tag">{row['Cuisine Type']}</span>
                        <span class="tag">{row['Location']}</span>
                    </p>
                    <p><strong>Price:</strong> Rs. {row.get('avg_price', 'N/A'):.0f}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Select This", key=row['Restaurant Name']):
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
        try:
            cur.execute("SELECT username, email, created_at FROM users WHERE id=?", (uid,))
            user = cur.fetchone()
            st.write(f"**Name:** {user[0]}")
            st.write(f"**Email:** {user[1]}")
            st.write(f"**Member Since:** {user[2][:10]}")
        except:
            st.write("Profile data unavailable.")

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
        st.subheader("Recent Activity")
        try:
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
        except:
            st.info("No activity log.")

# ========================================
# MAIN ROUTING
# ========================================
if 'user_id' not in st.session_state:
    page_auth()
else:
    sidebar_profile()
    tab1, tab2 = st.tabs(["Search", "Dashboard"])
    with tab1:
        page_main()
    with tab2:
        page_dashboard()
