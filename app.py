# app.py
# Restaurant Recommender – FYP EC3319
# Krish Chakradhar – 00020758

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import hashlib
from datetime import datetime
import os

# ========================================
# CONFIG
# ========================================
st.set_page_config(page_title="Foodmandu Recommender", layout="wide")

MODEL_DIR = "recommender_model"
DB_PATH = "restaurant_recommender.db"
CSV_PATH = "foodmandu_data_clean.csv"

# Load model & metadata
@st.cache_resource
def load_model():
    with open(f"{MODEL_DIR}/similarity_matrix.pkl", 'rb') as f:
        sim_df = pickle.load(f)
    meta = pd.read_csv(f"{MODEL_DIR}/restaurant_metadata.csv")
    meta['Location'] = meta['Location'].str.title()
    meta['Cuisine Type'] = meta['Cuisine Type'].str.title()
    return sim_df, meta

similarity_df, rest_metadata = load_model()

# DB Connection
@st.cache_resource
def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

conn = get_db()
cur = conn.cursor()

# ========================================
# AUTH FUNCTIONS
# ========================================
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def register(username, password, email):
    try:
        cur.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                    (username, hash_pw(password), email))
        conn.commit()
        return True
    except:
        return False

def login(username, password):
    cur.execute("SELECT id FROM users WHERE username=? AND password_hash=?", 
                (username, hash_pw(password)))
    row = cur.fetchone()
    return row[0] if row else None

# ========================================
# PAGES
# ========================================
def page_login():
    st.title("Restaurant Recommender")
    st.markdown("### Login or Register")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                uid = login(u, p)
                if uid:
                    st.session_state.user_id = uid
                    st.session_state.username = u
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("register"):
            u = st.text_input("Username", key="reg_u")
            p = st.text_input("Password", type="password", key="reg_p")
            e = st.text_input("Email", key="reg_e")
            if st.form_submit_button("Register"):
                if register(u, p, e):
                    st.success("Registered! Now login.")
                else:
                    st.error("Username taken")

def page_home():
    st.title(f"Welcome, {st.session_state.username}!")
    st.markdown("### Your Personalized Restaurant Recommendations")
    
    uid = st.session_state.user_id
    
    # Get user reviews
    cur.execute("SELECT restaurant, rating FROM reviews WHERE user_id=? ORDER BY timestamp DESC", (uid,))
    reviews = cur.fetchall()
    
    if not reviews:
        st.info("Rate a restaurant to get recommendations!")
        restaurant = st.selectbox("Pick a restaurant", rest_metadata['Restaurant Name'])
        rating = st.slider("Your Rating", 1, 5, 3)
        if st.button("Submit Rating"):
            cur.execute("INSERT INTO reviews (user_id, restaurant, rating) VALUES (?, ?, ?)",
                        (uid, restaurant, rating))
            conn.commit()
            st.success("Rating saved!")
            st.rerun()
        return
    
    # Get preference
    cur.execute("SELECT pref_cuisine, pref_location FROM preferences WHERE user_id=?", (uid,))
    pref = cur.fetchone()
    pref_cuisine = pref[0].lower() if pref and pref[0] else None
    pref_location = pref[1].lower() if pref and pref[1] else None
    
    # Use last rated as seed
    seed = reviews[0][0]
    if seed not in similarity_df.index:
        st.error("Restaurant not in model.")
        return
    
    sims = similarity_df[seed].sort_values(ascending=False).iloc[1:20]
    recs = []
    
    for rest, score in sims.items():
        meta = rest_metadata[rest_metadata['Restaurant Name'] == rest].iloc[0]
        boost = 1.0
        if pref_cuisine and pref_cuisine in meta['Cuisine Type'].lower():
            boost += 0.6
        if pref_location and pref_location in meta['Location'].lower():
            boost += 1.2
        recs.append({
            'Restaurant': rest,
            'Location': meta['Location'],
            'Cuisine': meta['Cuisine Type'],
            'Score': round(score * boost, 3)
        })
    
    rec_df = pd.DataFrame(recs).sort_values('Score', ascending=False).head(5)
    
    st.success("Top 5 Recommendations for You:")
    for _, row in rec_df.iterrows():
        st.markdown(f"**{row['Restaurant']}**  \n{row['Location']} | {row['Cuisine']} | Score: {row['Score']}")
    
    # Review form
    st.markdown("---")
    st.subheader("Rate Another Restaurant")
    with st.form("review"):
        restaurant = st.selectbox("Restaurant", rest_metadata['Restaurant Name'], key="rev_rest")
        rating = st.slider("Rating", 1, 5, 3, key="rev_rate")
        if st.form_submit_button("Submit"):
            cur.execute("INSERT INTO reviews (user_id, restaurant, rating) VALUES (?, ?, ?)",
                        (uid, restaurant, rating))
            conn.commit()
            st.success("Thank you!")
            st.rerun()

# ========================================
# MAIN
# ========================================
if 'user_id' not in st.session_state:
    page_login()
else:
    page_home()
    if st.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
