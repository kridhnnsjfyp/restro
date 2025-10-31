# app.py
# Restaurant Recommender – FYP EC3319
# Krish Chakradhar – 00020758
# FINAL: PRICE RANGE + SIMILARITY % + NO AVG PRICE

import streamlit as st
import pandas as pd
import pickle
import sqlite3
import hashlib
import os

# ========================================
# CONFIG
# ========================================
st.set_page_config(page_title="Foodmandu Recommender", layout="centered", initial_sidebar_state="expanded")

# DARK THEME + STYLING
st.markdown("""
<style>
    .main {background-color: #1a202c; padding: 0 !important; margin: 0 !important;}
    .block-container {padding-top: 1rem !important; padding-bottom: 2rem !important;}
    header, #MainMenu, footer, .stDeployButton, div[data-testid="stToolbar"] {display: none !important;}
    
    .stTextInput > div > div > input {
        background-color: #2d3748 !important; color: white !important; border-radius: 10px !important;
        border: 1px solid #4a5568 !important;
    }
    .stTextInput > label {color: #e2e8f0 !important; font-weight: 600 !important;}
    
    .login-container {
        max-width: 420px; margin: 2rem auto; padding: 2.5rem; background: #1a202c; 
        border-radius: 16px; box-shadow: 0 12px 35px rgba(0,0,0,0.3); color: white; border: 1px solid #2d3748;
    }
    .title {font-size: 2.8rem; font-weight: 800; color: #4299e1; text-align: center; margin: 1rem 0 0.5rem;}
    .subtitle {text-align: center; color: #a0aec0; font-size: 1.1rem; margin-bottom: 2rem;}
    
    .stButton>button {
        background: #4299e1; color: white; border-radius: 10px; font-weight: 600; 
        padding: 0.7rem; width: 100%; border: none;
    }
    .stButton>button:hover {background: #3182ce;}
    .stButton > button[type="secondary"] {background: #48bb78 !important;}
    .stButton > button[type="secondary"]:hover {background: #38a169 !important;}
    
    .card {
        background: #2d3748 !important; color: white !important;
        border-radius: 12px; padding: 1.2rem; margin: 1rem 0; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.2); border: 1px solid #4a5568;
    }
    .similar-card {
        background: #1a202c !important; color: white !important;
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0; 
        border: 1px dashed #4a5568; font-size: 0.9rem;
    }
    .tag {
        background: #4299e1; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; 
        font-size: 0.8rem; display: inline-block; margin: 0.2rem;
    }
    .similarity-badge {
        background: #48bb78; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; 
        font-size: 0.75rem; font-weight: 600; display: inline-block; margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "recommender_model"
DB_PATH = "/tmp/restaurant_recommender.db"

# ========================================
# INIT DB
# ========================================
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password_hash TEXT, email TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
    cur.execute('CREATE TABLE IF NOT EXISTS preferences (user_id INTEGER PRIMARY KEY, pref_location TEXT, pref_cuisine TEXT, last_location TEXT, search_count INTEGER DEFAULT 0)')
    cur.execute('CREATE TABLE IF NOT EXISTS favorites (user_id INTEGER, restaurant TEXT, UNIQUE(user_id, restaurant))')
    cur.execute('CREATE TABLE IF NOT EXISTS interactions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, restaurant TEXT, action TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP)')
    conn.commit()
    return conn

conn = init_db()
cur = conn.cursor()

# ========================================
# LOAD MODEL + FOOD PRICES
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
    
    # LOAD FOOD PRICES (ASSUMED IN SAME FOLDER)
    food_path = f"{MODEL_DIR}/food_prices.csv"  # You must include this
    if os.path.exists(food_path):
        food_df = pd.read_csv(food_path)
        food_df['Restaurant'] = food_df['Restaurant'].str.strip()
        food_df['Price'] = pd.to_numeric(food_df['Price'], errors='coerce')
        # Group by restaurant → min & max price
        price_range = food_df.groupby('Restaurant')['Price'].agg(['min', 'max']).reset_index()
        price_range['Range'] = price_range.apply(lambda x: f"Rs. {int(x['min'])} – Rs. {int(x['max'])}" if pd.notna(x['min']) and pd.notna(x['max']) else "N/A", axis=1)
        meta = meta.merge(price_range[['Restaurant', 'Range']], left_on='Restaurant Name', right_on='Restaurant', how='left')
        meta['Price Range'] = meta['Range'].fillna("N/A")
        meta.drop(columns=['Restaurant', 'Range'], inplace=True)
    else:
        meta['Price Range'] = "N/A"
    
    return sim_df, meta

similarity_df, rest_metadata = load_model()

# ========================================
# DB HELPERS
# ========================================
def ensure_preference_row(user_id):
    try: cur.execute("INSERT OR IGNORE INTO preferences (user_id, search_count) VALUES (?, 0)", (user_id,)); conn.commit()
    except: pass

def get_preference(user_id, field):
    ensure_preference_row(user_id)
    try:
        cur.execute(f"SELECT {field} FROM preferences WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else None
    except: return None

def increment_search_count(user_id):
    try: cur.execute("UPDATE preferences SET search_count = search_count + 1 WHERE user_id=?", (user_id,)); conn.commit()
    except: pass

def set_preference(user_id, **kwargs):
    ensure_preference_row(user_id)
    try:
        updates = ", ".join([f"{k}=?" for k in kwargs])
        values = list(kwargs.values()) + [user_id]
        cur.execute(f"UPDATE preferences SET {updates} WHERE user_id=?", values)
        conn.commit()
    except: pass

def is_favorite(user_id, rest_name):
    cur.execute("SELECT 1 FROM favorites WHERE user_id=? AND restaurant=?", (user_id, rest_name))
    return cur.fetchone() is not None

def toggle_favorite(user_id, rest_name):
    if is_favorite(user_id, rest_name):
        cur.execute("DELETE FROM favorites WHERE user_id=? AND restaurant=?", (user_id, rest_name))
    else:
        cur.execute("INSERT OR IGNORE INTO favorites (user_id, restaurant) VALUES (?, ?)", (user_id, rest_name))
    conn.commit()

def log_interaction(user_id, restaurant, action):
    try:
        cur.execute("INSERT INTO interactions (user_id, restaurant, action) VALUES (?, ?, ?)", (user_id, restaurant, action))
        conn.commit()
    except: pass

# ========================================
# AUTH
# ========================================
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def register(u, p, e): 
    try:
        u = u.strip()
        if not u or not p or not e: return False
        cur.execute("SELECT id FROM users WHERE username=?", (u,))
        if cur.fetchone(): return False
        cur.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)", (u, hash_pw(p), e))
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
# PAGES
# ========================================
def page_auth():
    st.markdown('<div class="title">Foodmandu Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Find the best restaurants near you</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        if 'show_register' not in st.session_state: st.session_state.show_register = False
        if st.session_state.show_register:
            st.markdown("### Create Account")
            with st.form("register_form"):
                reg_u = st.text_input("Username", placeholder="Enter username", key="reg_u")
                reg_p = st.text_input("Password", type="password", placeholder="Create password", key="reg_p")
                reg_e = st.text_input("Email", placeholder="your@email.com", key="reg_e")
                col1, col2 = st.columns(2)
                with col1: reg_submit = st.form_submit_button("Create Account")
                with col2: back = st.form_submit_button("Back to Login")
                if back: st.session_state.show_register = False; st.rerun()
                if reg_submit:
                    if register(reg_u, reg_p, reg_e):
                        st.success("Account created! Login now.")
                        st.session_state.show_register = False
                    else:
                        st.error("Username taken.")
        else:
            st.markdown("### Login")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit = st.form_submit_button("Login")
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
            if st.button("Create New Account", use_container_width=True, type="secondary"):
                st.session_state.show_register = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def get_similar_restaurants_with_score(rest_name, top_n=2):
    if rest_name not in similarity_df.index:
        return pd.DataFrame()
    sim_scores = similarity_df.loc[rest_name].sort_values(ascending=False).iloc[1:top_n+1]
    sim_scores = sim_scores.reset_index()
    sim_scores.columns = ['Restaurant Name', 'Similarity']
    sim_scores['Similarity %'] = (sim_scores['Similarity'] * 100).round(1)
    return rest_metadata[rest_metadata['Restaurant Name'].isin(sim_scores['Restaurant Name'])].merge(sim_scores, on='Restaurant Name')

def page_main():
    st.markdown('<div class="title">Find Restaurants</div>', unsafe_allow_html=True)
    uid = st.session_state.user_id
    all_locations = sorted(rest_metadata['Location'].unique().tolist())
    last_loc = get_preference(uid, 'last_location')
    default_loc = last_loc if last_loc and last_loc in all_locations else all_locations[0]

    col1, col2 = st.columns([2, 1])
    with col1:
        location = st.selectbox("Select your area:", options=all_locations, index=all_locations.index(default_loc))
    with col2:
        cuisine = st.selectbox("Cuisine?", options=["Any"] + sorted(rest_metadata['Cuisine Type'].unique().tolist()))

    if st.button("Search Restaurants", use_container_width=True):
        increment_search_count(uid)
        set_preference(uid, last_location=location, pref_cuisine=cuisine if cuisine != "Any" else None)
        cuisine_filter = cuisine if cuisine != "Any" else None
        recs = recommend_by_location(location, cuisine_filter)

        if recs.empty:
            st.warning(f"No restaurants found in **{location}**")
        else:
            st.success(f"Found {len(recs)} restaurant(s) in **{location}**")
            for _, row in recs.iterrows():
                rest_name = row['Restaurant Name']
                price_range = row.get('Price Range', 'N/A')
                rating_str = f"{row.get('rating', 'N/A')}/5" if pd.notna(row.get('rating')) else "N/A"

                with st.expander(f"**{rest_name}**", expanded=False):
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.markdown(f"""
                        <p style="margin:0;"><span class="tag">{row['Cuisine Type']}</span> <span class="tag">{row['Location']}</span></p>
                        <p style="margin:0.5rem 0 0 0;"><strong>Price Range:</strong> {price_range} | <strong>Rating:</strong> {rating_str}</p>
                        """, unsafe_allow_html=True)
                    with col_b:
                        fav_label = "Unfavorite" if is_favorite(uid, rest_name) else "Favorite"
                        if st.button(fav_label, key=f"fav_{rest_name}", use_container_width=True):
                            toggle_favorite(uid, rest_name)
                            log_interaction(uid, rest_name, "favorited")
                            st.rerun()

                    # SIMILAR RESTAURANTS WITH %
                    sim_recs = get_similar_restaurants_with_score(rest_name, top_n=2)
                    if not sim_recs.empty:
                        st.markdown("**You might also like:**")
                        for _, sim in sim_recs.iterrows():
                            sim_price = sim.get('Price Range', 'N/A')
                            sim_pct = sim.get('Similarity %', 0)
                            st.markdown(f"""
                            <div class="similar-card">
                                <strong>{sim['Restaurant Name']}</strong> 
                                <span class="similarity-badge">{sim_pct}% similar</span><br>
                                <small>{sim['Cuisine Type']} • {sim['Location']} • {sim_price}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No similar restaurants found.")

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
        st.subheader("Favorites")
        cur.execute("SELECT restaurant FROM favorites WHERE user_id=?", (uid,))
        favs = cur.fetchall()
        if favs:
            for (r,) in favs:
                st.button(r, key=f"fav_dash_{r}", use_container_width=True)
        else:
            st.info("No favorites yet.")
    with col2:
        st.subheader("Recent Activity")
        cur.execute("SELECT restaurant, action, timestamp FROM interactions WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (uid,))
        logs = cur.fetchall()
        if logs:
            log_df = pd.DataFrame(logs, columns=["Restaurant", "Action", "Time"])
            log_df['Time'] = pd.to_datetime(log_df['Time']).dt.strftime('%b %d, %H:%M')
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.info("No activity yet.")

def sidebar_profile():
    with st.sidebar:
        st.markdown(f"### Hi, **{st.session_state.username}**")
        st.markdown("---")
        if st.button("Home", use_container_width=True):
            st.rerun()
        if st.button("Logout", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
        st.markdown("---")
        uid = st.session_state.user_id
        search_count = get_preference(uid, 'search_count') or 0
        last_loc = get_preference(uid, 'last_location') or "Not set"
        pref_cuisine = get_preference(uid, 'pref_cuisine') or "Not set"
        st.write(f"**Searches:** `{search_count}`")
        st.write(f"**Last Area:** `{last_loc}`")
        st.write(f"**Pref Cuisine:** `{pref_cuisine}`")

# ========================================
# RECOMMEND
# ========================================
def recommend_by_location(location, cuisine=None, top_n=5):
    loc_clean = location.lower().strip()
    mask = rest_metadata['Location'].str.lower().str.contains(loc_clean, na=False)
    candidates = rest_metadata[mask]
    if cuisine and cuisine != "Any":
        candidates = candidates[candidates['Cuisine Type'].str.contains(cuisine, case=False, na=False)]
    if candidates.empty: return pd.DataFrame()
    scores = candidates['Cuisine Type'].apply(lambda x: 2.0 if cuisine and cuisine.lower() in x.lower() else 1.0)
    candidates = candidates.copy()
    candidates['Score'] = scores
    return candidates.sort_values('Score', ascending=False).head(top_n)

# ========================================
# MAIN ROUTING
# ========================================
if 'user_id' not in st.session_state:
    page_auth()
else:
    sidebar_profile()
    tab1, tab2 = st.tabs(["Search", "Dashboard"])
    with tab1: page_main()
    with tab2: page_dashboard()
