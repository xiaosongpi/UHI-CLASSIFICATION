import os
import streamlit as st

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

def get_supabase():
    if "supabase_client" not in st.session_state:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL atau SUPABASE_KEY belum diatur di .env"
            )

        st.session_state["supabase_client"] = create_client(url, key)

    return st.session_state["supabase_client"]