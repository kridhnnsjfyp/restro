# restro/app.py
"""
Streamlit Restaurant Recommender App
Run this using:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import pickle
from sqlalchemy import create_engine, text
from pathlib import Path

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "restaurant_recommender.db"
MODEL_DIR = BASE_DIR / "recommender_model"

SIMILARITY_PKL = MODEL_DIR / "similarity_matrix.pkl"
RESTAURANT_META_CSV = MODEL_DIR / "restaurant_metadata.csv"
RATING_MATRIX_CSV = MODEL_DIR / "rating_matrix.csv"

# -----------------------------
# DATABASE SETUP
# -----------------------------
def get_db_connection():
    return create_engine(f"sqlite:///{DB_PATH}")

def init_db():
    """Create tables if they don't exist."""
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            password_hash TEXT,
            location TEXT,
            preferences TEXT
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ratings (
            rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            restaurant_id TEXT,
            rating INTEGER,
            review TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """))
    engine.dispose()

# -----------------------------
# PASSWORD HASHING
# -----------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data(show_spinner=False)
def load_restaurant_metadata():
    df = pd.read_csv(RESTAURANT_META_CSV)
    if "restaurant_id" not in df.columns:
        df.insert(0, "restaurant_id", df.index.astype(str))
    if "name" not in df.columns:
        df["name"] = df["restaurant_id"]
    if "rating" not in df.columns:
        df["rating"] = np.random.uniform(3, 5, len(df)).round(1)
    if "location" not in df.columns:
        df["location"] = "Kathmandu"
    return df

@st.cache_data(show_spinner=False)
def load_similarity_matrix():
    with open(SIMILARITY_PKL, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_rating_matrix():
    return pd.read_csv(RATING_MATRIX_CSV, index_col=0)

# -----------------------------
# USER DATABASE FUNCTIONS
# -----------------------------
def get_user(username):
    engine = get_db_connection()
    with engine.connect() as conn:
        user = conn.execute(text(
            "SELECT username, email, location, preferences FROM users WHERE username = :u"
        ), {"u": username}).fetchone()
    engine.dispose()
    return user

def create_user(username, email, password):
    """Create a new user if username is unique."""
    engine = get_db_connection()
    with engine.connect() as conn:
        existing = conn.execute(text("SELECT username FROM users WHERE username = :u"), {"u": username}).fetchone()
        if existing:
            engine.dispose()
            return False, "Username already taken. Please choose another."

        try:
            conn.execute(text("""
                INSERT INTO users (username, email, password_hash)
                VALUES (:u, :e, :p)
            """), {"u": username, "e": email, "p": hash_password(password)})
            engine.dispose()
            return True, "User created successfully!"
        except Exception as e:
            engine.dispose()
            return False, f"Database error: {str(e)}"

def verify_user(username, password):
    """Verify user credentials."""
    engine = get_db_connection()
    with engine.connect() as conn:
        user = conn.execute(text("SELECT password_hash FROM users WHERE username = :u"), {"u": username}).fetchone()
    engine.dispose()

    if not user:
        return False
    stored_hash = user[0]
    return stored_hash == hash_password(password)

def update_user_location(username, location):
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("UPDATE users SET location=:l WHERE username=:u"), {"l": location, "u": username})
    engine.dispose()

def update_user_preferences(username, prefs):
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("UPDATE users SET preferences=:p WHERE username=:u"), {"p": prefs, "u": username})
    engine.dispose()

def save_user_rating(username, restaurant_id, rating, review):
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO ratings (username, restaurant_id, rating, review)
            VALUES (:u, :r, :ra, :rev)
        """), {"u": username, "r": restaurant_id, "ra": rating, "rev": review})
    engine.dispose()

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def recommend_for_user(username, ratings_df, rest_meta, similarity, location=None, prefs=None, top_n=10):
    df = rest_meta.copy()
    if location:
        df = df[df["location"].astype(str).str.lower() == location.lower()]
    if prefs:
        for pref in str(prefs).split(","):
            df = df[df["cuisine"].astype(str).str.contains(pref.strip(), case=False, na=False)]
    return df.sort_values("rating", ascending=False).head(top_n)

# -----------------------------
# STREAMLIT APP START
# -----------------------------
def main():
    st.set_page_config(page_title="Kathmandu Restaurant Recommender", layout="wide")
    init_db()

    rest_meta = load_restaurant_metadata()
    sim = load_similarity_matrix()
    ratings_df = load_rating_matrix()

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # --------------------------------
    # LOGIN / SIGNUP CENTERED FORM
    # --------------------------------
    if not st.session_state["logged_in"]:
        st.markdown("<h1 style='text-align:center;'>🍽 Kathmandu Restaurant Recommender</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; font-size:18px;'>Get personalized restaurant recommendations in Kathmandu.</p>", unsafe_allow_html=True)
        st.write("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            tab_login, tab_signup, tab_guest = st.tabs(["🔐 Login", "🆕 Sign Up", "👀 Guest Mode"])

            # LOGIN TAB
            with tab_login:
                st.subheader("Login to your account")
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                if st.button("Login", use_container_width=True):
                    if verify_user(username, password):
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        user = get_user(username)
                        if user:
                            st.session_state["location"] = user["location"]
                            st.session_state["preferences"] = user["preferences"]
                        st.success(f"Welcome back, {username}!")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid username or password.")

            # SIGNUP TAB
            with tab_signup:
                st.subheader("Create a new account")
                new_username = st.text_input("Choose a username", key="signup_user")
                email = st.text_input("Email", key="signup_email")
                new_password = st.text_input("Choose a password", type="password", key="signup_pass")
                if st.button("Sign Up", use_container_width=True):
                    ok, msg = create_user(new_username, email, new_password)
                    if ok:
                        st.success("✅ Account created successfully! You can now log in.")
                    else:
                        st.error(f"❌ {msg}")

            # GUEST MODE
            with tab_guest:
                st.subheader("Continue as Guest")
                st.write("You can explore restaurants without creating an account.")
                if st.button("Continue as Guest", use_container_width=True):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = "guest"
                    st.session_state["location"] = None
                    st.session_state["preferences"] = None
                    st.experimental_rerun()

        st.write("---")
        st.markdown("<p style='text-align:center; color:gray;'>Developed as part of Final Year Project — AI-based Restaurant Recommender System</p>", unsafe_allow_html=True)
        return

    # --------------------------------
    # LOGGED IN SECTION
    # --------------------------------
    username = st.session_state.get("username", "guest")
    st.sidebar.title(f"Hi, {username}")
    page = st.sidebar.radio("Navigate", ["Home", "Set My Location", "Explore", "Preferences", "My Reviews", "Logout"])

    if page == "Logout":
        st.session_state.clear()
        st.success("You have been logged out.")
        st.experimental_rerun()

    elif page == "Home":
        st.title("🏠 Home — Recommendations")
        user_loc = st.session_state.get("location")
        prefs = st.session_state.get("preferences")
        if not user_loc:
            st.warning("Please set your location first (via sidebar → Set My Location).")
        recs = recommend_for_user(username, ratings_df, rest_meta, sim, location=user_loc, prefs=prefs)
        if recs.empty:
            st.write("No recommendations available yet.")
        else:
            for _, row in recs.iterrows():
                st.markdown(f"### 🍴 {row['name']}")
                st.text(f"Cuisine: {row.get('cuisine','N/A')} | Location: {row.get('location','N/A')} | ⭐ {row.get('rating','N/A')}")
                st.divider()

    elif page == "Set My Location":
        st.title("📍 Set My Location")
        all_locations = sorted(rest_meta["location"].dropna().unique().tolist())
        selected = st.selectbox("Select your area", all_locations)
        if st.button("Save Location"):
            st.session_state["location"] = selected
            if username != "guest":
                update_user_location(username, selected)
            st.success(f"Location saved: {selected}")

    elif page == "Explore":
        st.title("🍽 Explore Restaurants")
        search = st.text_input("Search by name or cuisine")
        df = rest_meta.copy()
        if search:
            df = df[df["name"].str.contains(search, case=False, na=False) | df["cuisine"].str.contains(search, case=False, na=False)]
        st.write(f"Found {len(df)} restaurants")
        for _, row in df.head(100).iterrows():
            st.markdown(f"**{row['name']}** — {row['cuisine']} — {row['location']}")
            st.text(f"Price: {row.get('price','N/A')} | Rating: ⭐ {row.get('rating','N/A')}")
            st.divider()

    elif page == "Preferences":
        st.title("❤️ My Preferences")
        cuisines = sorted(rest_meta["cuisine"].dropna().unique().tolist())
        selected = st.multiselect("Favorite cuisines", cuisines)
        if st.button("Save Preferences"):
            prefs_str = ",".join(selected)
            st.session_state["preferences"] = prefs_str
            if username != "guest":
                update_user_preferences(username, prefs_str)
            st.success("Preferences updated successfully!")

    elif page == "My Reviews":
        st.title("⭐ My Reviews & Feedback")
        restaurant = st.selectbox("Choose a restaurant", rest_meta["name"].tolist())
        rating = st.slider("Rate this restaurant", 1, 5, 4)
        review = st.text_area("Your review (optional)")
        if st.button("Submit Review"):
            rid = rest_meta.loc[rest_meta["name"] == restaurant, "restaurant_id"].iloc[0]
            save_user_rating(username, rid, rating, review)
            st.success("Review saved. Thank you for your feedback!")

# Run the app
if __name__ == "__main__":
    main()
