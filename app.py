# In "Reviews" page
elif page == "Reviews":
    st.header("Your Reviews")
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("""
            SELECT restaurant_id, review, created_at 
            FROM reviews 
            WHERE username=? AND review IS NOT NULL AND TRIM(review) != ''
            ORDER BY created_at DESC
        """, conn, params=(st.session_state.username,))
        conn.close()
        if df.empty:
            st.info("No reviews yet.")
        else:
            for _, r in df.iterrows():
                name = meta[meta['restaurant_id'] == r['restaurant_id']]['name'].iloc[0] if not meta[meta['restaurant_id'] == r['restaurant_id']].empty else "Unknown"
                st.markdown(f"**{name}** — *{r['created_at'][:10]}*")
                st.markdown(f"> {r['review']}")
                st.markdown("---")
    except Exception as e:
        st.error("Error loading reviews.")
