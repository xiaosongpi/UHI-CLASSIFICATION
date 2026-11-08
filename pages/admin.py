import pandas as pd
import streamlit as st
from src.services.user_service import delete_user, get_all_users

st.title("User")

user = st.session_state.get("user")
user = get_all_users(user["id"])

if user:
    df = pd.DataFrame(user)

    event = st.dataframe(
        df, 
        use_container_width = True, 
        hide_index = True, 
        on_select = "rerun", 
        selection_mode = "multi-row"
    )

    if event.selection.rows:
        target_id = df.iloc[event.selection.rows[0]]["id"]
        if st.button("Hapus User Terpilih", type = "primary"):
            delete_user(target_id)
            st.rerun()
else:
    st.info("Tidak ada data user.")